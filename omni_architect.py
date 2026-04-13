import os
import sys
import argparse
import re
from datetime import datetime
import torch
from PIL import Image, ImageDraw
from transformers import AutoModelForCausalLM
from deepseek_vl.models import VLChatProcessor, MultiModalityCausalLM
from deepseek_vl.utils.io import load_pil_images
from groq import Groq
from langgraph.graph import StateGraph, END
from typing import TypedDict
from dotenv import load_dotenv

# Ensure Unicode output works on Windows consoles (prevents emoji print crashes).
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Load environment variables (Make sure you have a .env file with GROQ_API_KEY)
load_dotenv()

# Check for API key
if not os.getenv("GROQ_API_KEY"):
    raise ValueError("GROQ_API_KEY not found in environment variables. Please add it to your .env file.")

# Define the state for LangGraph
class AgentState(TypedDict):
    image_path: str
    vision_description: str
    react_code: str
    _answers_md: str


def _extract_ui_spec_markdown(text: str) -> str:
    """
    Hard-trim to ONLY the UI_SPEC markdown block.
    Accepts model output with extra junk and returns just the spec.
    """
    if not text:
        return ""
    # Normalize line endings for reliable matching
    t = text.replace("\r\n", "\n").replace("\r", "\n")

    m = re.search(r"(?m)^\s*##\s*UI_SPEC\s*$", t)
    if not m:
        return ""
    start = m.start()

    # End at next top-level "## " heading (but not the UI_SPEC heading itself)
    rest = t[m.end() :]
    m2 = re.search(r"(?m)^\s*##\s+\S", rest)
    end = len(t) if not m2 else (m.end() + m2.start())

    spec = t[start:end].strip()
    # Ensure it starts with the correct heading and contains at least one section
    if not spec.startswith("## UI_SPEC"):
        return ""
    return spec


def _extract_latex(text: str) -> str:
    """
    Hard-trim to a LaTeX block if present.
    We accept either $$...$$ or \\[...\\] or \\begin{...}...\\end{...}.
    """
    if not text:
        return ""
    t = text.replace("\r\n", "\n").replace("\r", "\n").strip()

    # Prefer $$...$$
    m = re.search(r"\$\$(.+?)\$\$", t, flags=re.DOTALL)
    if m:
        return ("$$" + m.group(1).strip() + "$$").strip()

    # Then \[...\]
    m = re.search(r"\\\[(.+?)\\\]", t, flags=re.DOTALL)
    if m:
        return ("\\[" + m.group(1).strip() + "\\]").strip()

    # Then any \begin{...}...\end{...}
    m = re.search(r"(\\begin\{[^\}]+\}[\s\S]*?\\end\{[^\}]+\})", t)
    if m:
        return m.group(1).strip()

    return ""

# ---------------------------------------------------------
# NODE 1: Local Vision Analysis (DeepSeek-VL)
# ---------------------------------------------------------
def local_vision_agent(state: AgentState) -> AgentState:
    print(f"\n👁️ Loading DeepSeek-VL to analyze {state['image_path']}...")
    
    model_path = "deepseek-ai/deepseek-vl-1.3b-chat"
    
    # Use DeepSeek's custom processor
    vl_chat_processor: VLChatProcessor = VLChatProcessor.from_pretrained(model_path, use_fast=True)
    tokenizer = vl_chat_processor.tokenizer
    
    # THE ULTIMATE OVERRIDE: Force PyTorch to load directly to standard CPU RAM 
    # to completely block Hugging Face's 'meta' tensor optimizations
    print("⏳ Initializing weights in system RAM to bypass meta-tensor crash...")
    with torch.device('cpu'):
        vl_gpt: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(
            model_path, 
            trust_remote_code=True,
            low_cpu_mem_usage=False,
            _fast_init=False,
            device_map=None
        )
    
    # Once safely loaded into normal RAM, cast to bfloat16 and manually move to RTX 4070
    print("⚡ Moving model to GPU (RTX 4070)...")
    vl_gpt = vl_gpt.to(torch.bfloat16).cuda().eval()
    
    # DeepSeek's specific conversation format
    conversation = [
        {
            "role": "User", 
            "content": (
                "<image_placeholder>\n"
                "You are a strict visual-to-structure translator.\n\n"
                "Task:\n"
                "- If the image is a UI mockup/screenshot/sketch: output STRICT Markdown describing the UI.\n"
                "- If the image is math/logic/diagram: output STRICT LaTeX (not Markdown).\n\n"
                "Rules:\n"
                "- Output ONLY the requested format. No extra commentary, no Q&A, no follow-up questions.\n"
                "- Be precise. Prefer lists/tables.\n"
                "- Use hex colors when visible/inferrable.\n\n"
                "For UI images, use this exact Markdown template:\n"
                "## UI_SPEC\n"
                "### Canvas\n"
                "- **size**: <w>x<h> (approx)\n"
                "- **background**: <hex|name>\n"
                "### Layout\n"
                "- **root**: <flex/stack/grid> ...\n"
                "### Component Tree\n"
                "- <ComponentName>\n"
                "  - **type**: <button/text/input/nav/...>\n"
                "  - **position**: <top-left... or grid>\n"
                "  - **size**: <w>x<h> (approx)\n"
                "  - **styles**: <key: value; ...>\n"
                "  - **text**: <literal text or empty>\n"
                "### Colors\n"
                "| token | hex | usage |\n"
                "|---|---|---|\n"
                "| primary | #...... | ... |\n"
                "### Interactions\n"
                "- <event> -> <effect>\n"
            ),
            "images": [state["image_path"]]
        },
        {"role": "Assistant", "content": ""}
    ]
    
    # DeepSeek's specific image batching
    pil_images = load_pil_images(conversation)
    
    # Process inputs and send them to the GPU
    prepare_inputs = vl_chat_processor(
        conversations=conversation, 
        images=pil_images, 
        force_batchify=True
    ).to(vl_gpt.device)
    
    print("🧠 Generating description...")
    
    with torch.no_grad():
        inputs_embeds = vl_gpt.prepare_inputs_embeds(**prepare_inputs)
        
        outputs = vl_gpt.language_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=prepare_inputs.attention_mask,
            pad_token_id=tokenizer.eos_token_id,
            max_new_tokens=512,
            eos_token_id=tokenizer.eos_token_id,
            do_sample=False
        )
    
    raw = tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True)
    ui_spec = _extract_ui_spec_markdown(raw)
    latex = _extract_latex(raw)

    answer = ui_spec or latex or raw
    print("\n✅ Vision Description Complete:\n", answer)

    return {"vision_description": answer}


def local_interview_agent(state: AgentState, answers_path: str = "") -> AgentState:
    """
    Local, cost-efficient 'orchestrator' step:
    - DeepSeek asks clarifying questions
    - user answers locally
    - DeepSeek returns a cleaner UI_SPEC (or LaTeX)
    """
    print("\n🧭 Local Interview (DeepSeek): gathering missing context...")

    model_path = "deepseek-ai/deepseek-vl-1.3b-chat"
    vl_chat_processor: VLChatProcessor = VLChatProcessor.from_pretrained(model_path, use_fast=True)
    tokenizer = vl_chat_processor.tokenizer

    print("⏳ Initializing weights in system RAM to bypass meta-tensor crash...")
    with torch.device("cpu"):
        vl_gpt: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            low_cpu_mem_usage=False,
            _fast_init=False,
            device_map=None,
        )
    print("⚡ Moving model to GPU (RTX 4070)...")
    vl_gpt = vl_gpt.to(torch.bfloat16).cuda().eval()

    current_spec = (state.get("vision_description") or "").strip()
    if not current_spec:
        current_spec = "(none)"

    # Step A: ask questions (strict format so we can parse)
    question_prompt = (
        "<image_placeholder>\n"
        "You are Omni-Architect's LOCAL Orchestrator.\n"
        "Goal: ask ONLY the minimum clarifying questions needed to produce an accurate, implementable spec.\n\n"
        "Given:\n"
        "- The image\n"
        "- The current draft spec below\n\n"
        "Current draft spec:\n"
        "-----\n"
        f"{current_spec}\n"
        "-----\n\n"
        "Output format rules:\n"
        "- Output ONLY this Markdown section, nothing else:\n"
        "## QUESTIONS\n"
        "- Q1: ...\n"
        "- Q2: ...\n"
        "- Q3: ...\n"
        "- Stop at 6 questions max.\n"
    )

    conversation = [
        {"role": "User", "content": question_prompt, "images": [state["image_path"]]},
        {"role": "Assistant", "content": ""},
    ]
    pil_images = load_pil_images(conversation)
    prepare_inputs = vl_chat_processor(
        conversations=conversation,
        images=pil_images,
        force_batchify=True,
    ).to(vl_gpt.device)

    print("❓ Generating questions...")
    with torch.no_grad():
        inputs_embeds = vl_gpt.prepare_inputs_embeds(**prepare_inputs)
        outputs = vl_gpt.language_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=prepare_inputs.attention_mask,
            pad_token_id=tokenizer.eos_token_id,
            max_new_tokens=256,
            eos_token_id=tokenizer.eos_token_id,
            do_sample=False,
        )
    raw_q = tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True).strip()

    # Hard-trim questions block
    q_block = ""
    m = re.search(r"(?m)^\s*##\s*QUESTIONS\s*$", raw_q.replace("\r\n", "\n"))
    if m:
        q_block = raw_q[m.start() :].strip()
    else:
        q_block = "## QUESTIONS\n- Q1: What are the key requirements you want to confirm?"

    print("\n" + q_block + "\n")

    # Collect answers (from file if provided, else interactive, else save questions)
    answers_md = ""
    if answers_path:
        try:
            with open(answers_path, "r", encoding="utf-8") as f:
                answers_md = f.read().strip()
        except Exception as e:
            print(f"⚠️ Could not read answers file '{answers_path}': {e}")
            answers_md = ""

    if not answers_md:
        if not sys.stdin.isatty():
            # Non-interactive environment (e.g., IDE-run tools). Persist questions and return.
            os.makedirs("specs", exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            questions_path = os.path.join("specs", f"questions_{timestamp}.md")
            with open(questions_path, "w", encoding="utf-8") as f:
                f.write(q_block.strip() + "\n")
            # Write an empty answers template users can fill
            answers_template_path = os.path.join("specs", f"answers_{timestamp}.md")
            with open(answers_template_path, "w", encoding="utf-8") as f:
                f.write("## ANSWERS\n")
                for line in q_block.replace("\r\n", "\n").split("\n"):
                    if re.match(r"^\s*-\s*Q\d+\s*:", line):
                        f.write(line.strip() + "\n")
                        f.write("  - A: \n")
            print(f"📄 Saved questions to '{questions_path}' (non-interactive run).")
            print(f"📄 Saved answers template to '{answers_template_path}' (fill this, then re-run with --answers).")
            state["_answers_md"] = ""  # type: ignore[typeddict-item]
            return state

        answers_lines = []
        print("📝 Answer in the terminal (interactive).\n")
        # Extract question lines that start with "- Q"
        questions = []
        for line in q_block.replace("\r\n", "\n").split("\n"):
            if re.match(r"^\s*-\s*Q\d+\s*:", line):
                questions.append(line.strip())
        if not questions:
            questions = ["- Q1: What should I clarify about this UI/math?"]

        for q in questions:
            print(q)
            a = input("  A: ").strip()
            if not a:
                a = "(no answer)"
            answers_lines.append(f"{q}\n  - A: {a}")

        answers_md = "## ANSWERS\n" + "\n".join(answers_lines)

    # Step B: regenerate spec with answers, strictly
    spec_prompt = (
        "<image_placeholder>\n"
        "You are a strict visual-to-structure translator.\n\n"
        "Task:\n"
        "- If the image is UI: output ONLY a STRICT Markdown spec starting with '## UI_SPEC'.\n"
        "- If the image is math/logic: output ONLY STRICT LaTeX.\n\n"
        "Constraints:\n"
        "- Use the user's answers to resolve ambiguity.\n"
        "- Do NOT include QUESTIONS or ANSWERS in the output.\n\n"
        "User answers:\n"
        "-----\n"
        f"{answers_md}\n"
        "-----\n\n"
        "Current draft spec (may be wrong/incomplete):\n"
        "-----\n"
        f"{current_spec}\n"
        "-----\n"
    )

    conversation2 = [
        {"role": "User", "content": spec_prompt, "images": [state["image_path"]]},
        {"role": "Assistant", "content": ""},
    ]
    pil_images2 = load_pil_images(conversation2)
    prepare_inputs2 = vl_chat_processor(
        conversations=conversation2,
        images=pil_images2,
        force_batchify=True,
    ).to(vl_gpt.device)

    print("🧱 Regenerating structured spec...")
    with torch.no_grad():
        inputs_embeds2 = vl_gpt.prepare_inputs_embeds(**prepare_inputs2)
        outputs2 = vl_gpt.language_model.generate(
            inputs_embeds=inputs_embeds2,
            attention_mask=prepare_inputs2.attention_mask,
            pad_token_id=tokenizer.eos_token_id,
            max_new_tokens=768,
            eos_token_id=tokenizer.eos_token_id,
            do_sample=False,
        )
    raw_spec = tokenizer.decode(outputs2[0].cpu().tolist(), skip_special_tokens=True)
    ui_spec = _extract_ui_spec_markdown(raw_spec)
    latex = _extract_latex(raw_spec)
    final_spec = ui_spec or latex or raw_spec.strip()

    print("\n✅ Refined Spec:\n", final_spec)

    # Attach answers so we can persist them to disk later (stored inside vision_description for now)
    combined = final_spec.strip()
    state["vision_description"] = combined
    state["_answers_md"] = answers_md  # type: ignore[typeddict-item]
    return state

# ---------------------------------------------------------
# NODE 2: Cloud Code Generation (Groq + Kimi K2 Instruct)
# ---------------------------------------------------------
def groq_coder_agent(state: AgentState) -> AgentState:
    print("\n💻 Sending description to Groq (Kimi K2 Instruct) for coding...")

    client = Groq(api_key=os.getenv("GROQ_API_KEY", ""))

    system_prompt = (
        "You are an expert React and Tailwind CSS developer.\n"
        "Convert the UI_SPEC markdown below into a single, functional React component.\n"
        "Rules:\n"
        "- Output ONLY valid JSX (no markdown fences, no explanations, no imports of CSS files).\n"
        "- Use Tailwind utility classes for ALL styling.\n"
        "- Include `import React from 'react';` and a default export.\n"
        "- Do NOT add placeholder comments or TODOs—write real code.\n"
        "- Keep the component concise; avoid unnecessary wrappers.\n"
    )

    # Kimi K2 runs hot: lower temperature + tighter top_p keeps output focused.
    completion = client.chat.completions.create(
        model="moonshotai/kimi-k2-instruct-0905",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Build a React + Tailwind component from this spec:\n\n{state['vision_description']}"},
        ],
        temperature=0.3,
        top_p=0.85,
        max_completion_tokens=4096,
        stream=False,
        stop=None,
    )

    react_code = (completion.choices[0].message.content or "").strip()

    if react_code.startswith("```"):
        lines = react_code.split("\n")
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        react_code = "\n".join(lines).strip()

    print("\n✅ React Code Generated Successfully!")
    return {"react_code": react_code}

# ---------------------------------------------------------
# HELPER: Create a test image
# ---------------------------------------------------------
def create_test_image(filename="test_mockup.jpg"):
    print(f"🎨 Creating a temporary test mockup: {filename}")
    # Create a blank white image
    img = Image.new('RGB', (800, 600), color='white')
    d = ImageDraw.Draw(img)
    
    # Draw a header (blue)
    d.rectangle([0, 0, 800, 80], fill='#3b82f6')
    d.text((20, 30), "My App Header", fill="white")
    
    # Draw a sidebar (gray)
    d.rectangle([0, 80, 200, 600], fill='#f3f4f6')
    d.text((20, 100), "Menu Item 1", fill="black")
    d.text((20, 140), "Menu Item 2", fill="black")
    
    # Draw a main content area button (green)
    d.rectangle([300, 200, 500, 250], fill='#10b981', outline="black")
    d.text((350, 220), "Click Me Button", fill="white")
    
    img.save(filename)
    return filename

# ---------------------------------------------------------
# MAIN EXECUTION
# ---------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Omni-Architect: local vision -> cloud codegen")
    parser.add_argument("--image", dest="image_path", default="", help="Path to an input image (png/jpg).")
    parser.add_argument("--no-cloud", action="store_true", help="Do not call the cloud coding model.")
    parser.add_argument("--interview", action="store_true", help="Run a local Q&A interview to refine specs before cloud.")
    parser.add_argument("--answers", dest="answers_path", default="", help="Path to an answers_*.md file to refine spec without interactive typing.")
    args = parser.parse_args()

    # 1. Choose image
    if args.image_path and os.path.exists(args.image_path):
        test_image_path = args.image_path
    else:
        test_image_path = create_test_image("test_mockup.jpg")
    
    # 2. Run local vision (+ optional local interview)
    print("\n🚀 Starting Omni-Architect Local Pipeline...\n")
    state: AgentState = {"image_path": test_image_path}  # type: ignore[typeddict-item]
    state.update(local_vision_agent(state))
    if args.interview or args.answers_path:
        state = local_interview_agent(state, answers_path=args.answers_path)

    # 4. Persist local artifacts (spec files)
    os.makedirs("specs", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    spec_text = (state.get("vision_description") or "").strip()
    answers_md = (state.get("_answers_md") or "").strip()

    spec_path = os.path.join("specs", f"spec_{timestamp}.md")
    with open(spec_path, "w", encoding="utf-8") as f:
        f.write(spec_text + "\n")

    if answers_md:
        answers_path = os.path.join("specs", f"answers_{timestamp}.md")
        with open(answers_path, "w", encoding="utf-8") as f:
            f.write(answers_md + "\n")

    print(f"\n📄 Saved spec to '{spec_path}'")
    if answers_md:
        print(f"📄 Saved answers to '{answers_path}'")

    # Gate cloud usage (optional second act)
    if args.no_cloud:
        print("\n🛑 Skipping cloud coding step (--no-cloud).")
        return

    # If we ran interview, still confirm before cloud call
    ready = input("\n✅ Ready to send spec to the CLOUD coding model now? (y/N): ").strip().lower()
    if ready != "y":
        print("🛑 Not sending to cloud. You can re-run later when ready.")
        return

    # 3. Cloud coding (second act)
    cloud_input: AgentState = {**state, "vision_description": spec_text}  # type: ignore[assignment]
    final_state = groq_coder_agent(cloud_input)
    
    # 5. Save and print results
    print("\n==================================================")
    print("🎉 FINAL GENERATED REACT CODE")
    print("==================================================\n")
    print(final_state["react_code"])
    
    # Save the code to a file
    with open("GeneratedUI.jsx", "w") as f:
        f.write(final_state["react_code"])
    print("\n💾 Saved code to 'GeneratedUI.jsx'")

if __name__ == "__main__":
    main()