"use client";

import { useEffect, useRef, useState } from "react";
import {
  type FileRecord,
  type HeuristicHint,
  type SessionSummary,
  getSession,
  listSessions,
  login,
  startSession,
  refineSpec,
  shipToCloud,
  exportSessionZip,
} from "./api";

/* ------------------------------------------------------------------ */
/*  Primitives                                                         */
/* ------------------------------------------------------------------ */

function Card({ children, className = "" }: { children: React.ReactNode; className?: string }) {
  return (
    <div className={`rounded-2xl border border-zinc-800/60 bg-[#111] p-6 transition-colors hover:border-zinc-700/60 ${className}`}>
      {children}
    </div>
  );
}

function Btn({
  children, onClick, disabled, variant = "default", className = "",
}: {
  children: React.ReactNode;
  onClick?: () => void;
  disabled?: boolean;
  variant?: "default" | "primary" | "danger";
  className?: string;
}) {
  const base = "rounded-xl px-4 py-2.5 text-sm font-medium transition-all duration-200 disabled:opacity-30 disabled:cursor-not-allowed";
  const variants = {
    default: "border border-zinc-800 bg-zinc-900/80 text-zinc-300 hover:bg-zinc-800 hover:text-zinc-100 hover:border-zinc-700",
    primary: "bg-zinc-100 text-zinc-950 hover:bg-white shadow-sm shadow-white/5",
    danger: "bg-red-500/10 text-red-400 border border-red-500/20 hover:bg-red-500/20",
  };

  function onPointerDown(e: React.PointerEvent<HTMLButtonElement>) {
    const host = e.currentTarget;
    const rect = host.getBoundingClientRect();
    const wrapper = host.querySelector<HTMLDivElement>("[data-ripple]");
    if (!wrapper) return;
    const dot = document.createElement("span");
    dot.style.left = `${e.clientX - rect.left}px`;
    dot.style.top = `${e.clientY - rect.top}px`;
    wrapper.appendChild(dot);
    window.setTimeout(() => dot.remove(), 650);
  }

  return (
    <button
      className={`relative overflow-hidden ${base} ${variants[variant]} ${className}`}
      onClick={onClick}
      disabled={disabled}
      onPointerDown={onPointerDown}
    >
      <div className="omni-ripple" data-ripple />
      {children}
    </button>
  );
}

function StatusBar({ text, variant = "info" }: { text: string; variant?: "info" | "error" | "success" }) {
  if (!text) return null;
  const colors = {
    info: "border-zinc-800/40 bg-zinc-900/40 text-zinc-400",
    error: "border-red-500/20 bg-red-500/5 text-red-400",
    success: "border-emerald-500/20 bg-emerald-500/5 text-emerald-400",
  };
  return (
    <div className={`flex items-center gap-2 rounded-xl border px-4 py-2.5 ${colors[variant]}`}>
      <div className={`h-1.5 w-1.5 rounded-full animate-pulse ${
        variant === "error" ? "bg-red-500" : variant === "success" ? "bg-emerald-500" : "bg-zinc-500"
      }`} />
      <span className="text-xs">{text}</span>
    </div>
  );
}

function CopyBtn({ text, label = "Copy" }: { text: string; label?: string }) {
  const [copied, setCopied] = useState(false);
  return (
    <Btn
      className="!px-3 !py-1.5 !text-xs"
      disabled={!text}
      onClick={() => {
        navigator.clipboard.writeText(text);
        setCopied(true);
        setTimeout(() => setCopied(false), 1500);
      }}
    >
      {copied ? "Copied!" : label}
    </Btn>
  );
}

/* ------------------------------------------------------------------ */
/*  Live Thinking Timer                                                */
/* ------------------------------------------------------------------ */

function ThinkingTimer({ active, label }: { active: boolean; label: string }) {
  const [elapsed, setElapsed] = useState(0);
  const start = useRef(Date.now());

  useEffect(() => {
    if (!active) { setElapsed(0); return; }
    start.current = Date.now();
    const id = setInterval(() => setElapsed(Date.now() - start.current), 100);
    return () => clearInterval(id);
  }, [active]);

  if (!active) return null;
  return (
    <div className="flex items-center gap-3 rounded-xl border border-zinc-800/40 bg-zinc-900/40 px-4 py-3">
      <div className="h-2 w-2 rounded-full bg-amber-400 animate-pulse" />
      <span className="text-sm text-zinc-300">{label}</span>
      <span className="ml-auto font-mono text-sm text-amber-400">{(elapsed / 1000).toFixed(1)}s</span>
    </div>
  );
}

/* ------------------------------------------------------------------ */
/*  Perf Badge                                                         */
/* ------------------------------------------------------------------ */

function PerfBadge({ ms, tokens, label }: { ms?: number; tokens?: number; label: string }) {
  if (ms === undefined) return null;
  return (
    <span className="inline-flex items-center gap-1.5 rounded-lg border border-zinc-800/40 bg-zinc-900/60 px-2.5 py-1 text-[10px] text-zinc-500">
      <span className="text-zinc-400">{label}</span>
      <span className="font-mono text-emerald-400">{(ms / 1000).toFixed(1)}s</span>
      {tokens !== undefined && <span className="font-mono text-zinc-500">~{tokens} tok</span>}
    </span>
  );
}

/* ------------------------------------------------------------------ */
/*  File Chip / Drop Zone / Uploaded Summary                           */
/* ------------------------------------------------------------------ */

function FileChip({ name, size, type, onRemove }: { name: string; size: number; type: string; onRemove?: () => void }) {
  const sizeStr = size > 1_000_000 ? `${(size / 1_000_000).toFixed(1)} MB` : `${(size / 1_000).toFixed(0)} KB`;
  const icon = type === "pdf" ? <span className="text-red-400">PDF</span> : <span className="text-blue-400">IMG</span>;
  return (
    <div className="group flex items-center gap-2 rounded-lg border border-zinc-800/60 bg-zinc-900/60 px-3 py-2 text-xs">
      <span className="font-mono text-[10px] font-semibold uppercase">{icon}</span>
      <span className="max-w-[140px] truncate text-zinc-300" title={name}>{name}</span>
      <span className="text-zinc-600">{sizeStr}</span>
      {onRemove && (
        <button onClick={onRemove} className="ml-1 text-zinc-600 transition-colors hover:text-zinc-300" aria-label={`Remove ${name}`}>
          &times;
        </button>
      )}
    </div>
  );
}

function DropZone({ files, onFilesChange }: { files: File[]; onFilesChange: (files: File[]) => void }) {
  const [dragging, setDragging] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);
  function addFiles(list: FileList | null) { if (list) onFilesChange([...files, ...Array.from(list)]); }
  return (
    <div className="space-y-3">
      <div
        className={`relative cursor-pointer rounded-xl border-2 border-dashed px-6 py-8 text-center transition-colors ${
          dragging ? "border-zinc-500 bg-zinc-800/20" : "border-zinc-800 bg-zinc-950/40 hover:border-zinc-600"
        }`}
        onClick={() => inputRef.current?.click()}
        onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
        onDragLeave={() => setDragging(false)}
        onDrop={(e) => { e.preventDefault(); setDragging(false); addFiles(e.dataTransfer.files); }}
      >
        <div className="text-sm text-zinc-400">{dragging ? "Drop files here" : "Click or drag files here"}</div>
        <div className="mt-1 text-[11px] text-zinc-600">Images (PNG, JPG, WebP) and PDFs -- up to 20 MB each</div>
        <input ref={inputRef} type="file" multiple accept="image/png,image/jpeg,image/webp,image/gif,application/pdf" className="hidden" onChange={(e) => addFiles(e.target.files)} />
      </div>
      {files.length > 0 && (
        <div className="flex flex-wrap gap-2">
          {files.map((f, i) => (
            <FileChip key={`${f.name}-${i}`} name={f.name} size={f.size} type={f.name.toLowerCase().endsWith(".pdf") ? "pdf" : "image"} onRemove={() => onFilesChange(files.filter((_, j) => j !== i))} />
          ))}
        </div>
      )}
    </div>
  );
}

function UploadedFilesSummary({ files }: { files: FileRecord[] }) {
  if (!files.length) return null;
  return (
    <div className="flex flex-wrap gap-2">
      {files.map((f, i) => <FileChip key={i} name={f.original_name} size={f.size} type={f.type} />)}
    </div>
  );
}

/* ------------------------------------------------------------------ */
/*  TYPE_GUESS parsing                                                 */
/* ------------------------------------------------------------------ */

type TypeGuess = { kind: string; domain: string; confidence: string };

function parseTypeGuess(questionsRaw: string): { guess: TypeGuess | null; questionsOnly: string } {
  const lines = questionsRaw.split("\n");
  const tgStart = lines.findIndex((l) => /^##\s*TYPE_GUESS/.test(l.trim()));
  const qStart = lines.findIndex((l) => /^##\s*QUESTIONS/.test(l.trim()));

  if (tgStart === -1) return { guess: null, questionsOnly: questionsRaw };

  const tgEnd = qStart !== -1 ? qStart : lines.length;
  const tgBlock = lines.slice(tgStart + 1, tgEnd).join("\n");

  let kind = "unknown", domain = "unknown", confidence = "low";
  for (const line of tgBlock.split("\n")) {
    const m = line.match(/^-\s*kind:\s*(.+)/i);
    if (m) kind = m[1].trim();
    const m2 = line.match(/^-\s*language_or_domain:\s*(.+)/i);
    if (m2) domain = m2[1].trim();
    const m3 = line.match(/^-\s*confidence:\s*(.+)/i);
    if (m3) confidence = m3[1].trim();
  }

  const questionsOnly = qStart !== -1 ? lines.slice(qStart).join("\n").trim() : "";
  return { guess: { kind, domain, confidence }, questionsOnly };
}

function guessToShipDefault(guess: TypeGuess | null, heuristic?: HeuristicHint | null): string {
  const sources: string[] = [];
  if (guess) {
    sources.push(guess.kind.toLowerCase(), guess.domain.toLowerCase());
  }
  if (heuristic?.domain) {
    sources.push(heuristic.domain.toLowerCase());
  }
  const all = sources.join(" ");

  const mathSignals = ["math", "algebra", "calculus", "probability", "statistics", "equation", "homework", "problem set"];
  if (mathSignals.some((s) => all.includes(s)))
    return "Step-by-step solution with clear explanations. Show all work in LaTeX.";

  if (all.includes("physics"))
    return "Step-by-step solution with diagrams described, units in every step, and physical reasoning.";

  if (all.includes("chemistry"))
    return "Step-by-step solution with balanced equations, units, and proper nomenclature.";

  if (all.includes("biology"))
    return "Detailed explanation with proper terminology, diagrams described, and clear reasoning.";

  if (all.includes("discrete") || all.includes("combinatorics"))
    return "Step-by-step proof / solution with clear reasoning.";

  if (all.includes("essay") || all.includes("writing") || all.includes("english") || all.includes("literary"))
    return "Well-structured essay with strong thesis, evidence, and polished prose.";

  if (all.includes("history") || all.includes("social science"))
    return "Analytical response grounded in specific evidence, dates, and historical context.";

  if (all.includes("python")) return "Python code with docstrings and type hints";
  if (all.includes("rust")) return "Rust implementation with error handling";
  if (all.includes("ui") || all.includes("react") || all.includes("web") || all.includes("dashboard") || all.includes("design"))
    return "React + Tailwind component with accessibility";
  if (all.includes("api") || all.includes("backend")) return "Python FastAPI service with error handling";
  if (all.includes("programming") || all.includes("algorithm")) return "Clean code with edge case handling";

  if (heuristic?.domain) return `Help with: ${heuristic.domain}`;
  return "Detailed solution or implementation";
}

function TypeGuessCard({ guess }: { guess: TypeGuess }) {
  const confColor = guess.confidence === "high" ? "text-emerald-400" : guess.confidence === "medium" ? "text-amber-400" : "text-red-400";
  const isAuto = guess.kind.includes("auto-detected");
  return (
    <Card className="!p-4 !bg-zinc-900/80 border-zinc-700/40">
      <div className="mb-2 text-[11px] font-semibold uppercase tracking-widest text-zinc-500">
        {isAuto ? "Auto-detected classification (keyword analysis)" : "DeepSeek classification"}
      </div>
      <div className="grid grid-cols-3 gap-4 text-sm">
        <div>
          <div className="text-[10px] uppercase text-zinc-600 mb-0.5">Type</div>
          <div className="text-zinc-200 font-medium">{guess.kind}</div>
        </div>
        <div>
          <div className="text-[10px] uppercase text-zinc-600 mb-0.5">Domain / Language</div>
          <div className="text-zinc-200 font-medium">{guess.domain}</div>
        </div>
        <div>
          <div className="text-[10px] uppercase text-zinc-600 mb-0.5">Confidence</div>
          <div className={`font-medium ${confColor}`}>{guess.confidence}</div>
        </div>
      </div>
    </Card>
  );
}

/* ------------------------------------------------------------------ */
/*  Readable text renderer (lightweight inline markdown)               */
/* ------------------------------------------------------------------ */

function RenderedBlock({ text, className = "" }: { text: string; className?: string }) {
  if (!text) return null;
  const lines = text.split("\n");
  const elements: React.ReactNode[] = [];
  let inCode = false;
  let codeLang = "";
  let codeLines: string[] = [];

  function flushCode() {
    if (codeLines.length) {
      elements.push(
        <div key={elements.length} className="my-3 rounded-lg border border-zinc-800/60 bg-zinc-950/80 overflow-x-auto">
          {codeLang && <div className="px-3 py-1 text-[10px] uppercase tracking-wider text-zinc-600 border-b border-zinc-800/40">{codeLang}</div>}
          <pre className="px-4 py-3 text-[13px] leading-relaxed text-emerald-300/90 font-mono">{codeLines.join("\n")}</pre>
        </div>
      );
    }
    codeLines = [];
    codeLang = "";
  }

  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];

    if (line.trimStart().startsWith("```")) {
      if (inCode) { flushCode(); inCode = false; continue; }
      inCode = true;
      codeLang = line.trimStart().slice(3).trim();
      continue;
    }
    if (inCode) { codeLines.push(line); continue; }

    if (/^#{1,3}\s/.test(line)) {
      const level = (line.match(/^(#+)/) ?? ["", "#"])[1].length;
      const heading = line.replace(/^#+\s*/, "");
      const sizes = ["text-lg font-semibold", "text-base font-semibold", "text-sm font-medium"];
      elements.push(<div key={i} className={`${sizes[level - 1] ?? sizes[2]} text-zinc-100 mt-4 mb-1`}>{heading}</div>);
    } else if (/^[-*]\s/.test(line.trimStart())) {
      const content = line.replace(/^\s*[-*]\s/, "");
      elements.push(
        <div key={i} className="flex gap-2 pl-2 text-[13px] leading-relaxed text-zinc-300">
          <span className="text-zinc-600 select-none mt-0.5">•</span>
          <span>{content}</span>
        </div>
      );
    } else if (/^\d+\.\s/.test(line.trimStart())) {
      const m = line.match(/^(\s*\d+)\.\s(.*)/);
      elements.push(
        <div key={i} className="flex gap-2 pl-2 text-[13px] leading-relaxed text-zinc-300">
          <span className="text-zinc-500 select-none font-mono text-xs mt-0.5 min-w-[1.5em] text-right">{m?.[1] ?? ""}.</span>
          <span>{m?.[2] ?? line}</span>
        </div>
      );
    } else if (line.trim() === "") {
      elements.push(<div key={i} className="h-2" />);
    } else {
      elements.push(<div key={i} className="text-[13px] leading-relaxed text-zinc-300">{line}</div>);
    }
  }
  if (inCode) flushCode();

  return <div className={`space-y-0.5 ${className}`}>{elements}</div>;
}

/* ------------------------------------------------------------------ */
/*  Helpers                                                            */
/* ------------------------------------------------------------------ */

type Phase = "login" | "brief" | "refine" | "ship" | "done";

function downloadTextFile(filename: string, content: string) {
  if (!content) return;
  const blob = new Blob([content], { type: "text/plain;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
}

/* ------------------------------------------------------------------ */
/*  Main App                                                           */
/* ------------------------------------------------------------------ */

export function OmniApp() {
  const [phase, setPhase] = useState<Phase>("login");
  const [password, setPassword] = useState("");
  const [status, setStatus] = useState("");
  const [statusVariant, setStatusVariant] = useState<"info" | "error" | "success">("info");
  const [loading, setLoading] = useState(false);
  const [thinkingLabel, setThinkingLabel] = useState("");

  const [sessions, setSessions] = useState<SessionSummary[]>([]);
  const [sessionId, setSessionId] = useState("");
  const [uploadedFiles, setUploadedFiles] = useState<FileRecord[]>([]);
  const [metrics, setMetrics] = useState<Record<string, number> | null>(null);

  const [pendingFiles, setPendingFiles] = useState<File[]>([]);
  const [userPrompt, setUserPrompt] = useState("");

  const [draftSpec, setDraftSpec] = useState("");
  const [questions, setQuestions] = useState("");
  const [answers, setAnswers] = useState("");
  const [spec, setSpec] = useState("");

  const [reactCode, setReactCode] = useState("");
  const [outputType, setOutputType] = useState<"latex" | "code" | "text">("text");
  const [shipInstructions, setShipInstructions] = useState("Detailed solution or implementation");
  const [heuristicHint, setHeuristicHint] = useState<HeuristicHint | null>(null);

  function setError(msg: string) { setStatus(msg); setStatusVariant("error"); }
  function setInfo(msg: string) { setStatus(msg); setStatusVariant("info"); }
  function setSuccess(msg: string) { setStatus(msg); setStatusVariant("success"); }

  async function pullMetrics(sid: string) {
    try {
      const d = await getSession(sid);
      setMetrics((d.meta as any)?.metrics ?? null);
    } catch { /* ignore */ }
  }

  // ---------- Login ----------
  async function doLogin() {
    setLoading(true); setThinkingLabel("Authenticating...");
    try {
      await login(password);
      setStatus(""); setPhase("brief");
      const s = await listSessions(); setSessions(s);
    } catch (e: any) { setError(`Login failed: ${e.message ?? e}`); }
    finally { setLoading(false); setThinkingLabel(""); }
  }

  // ---------- Resume ----------
  async function doResumeSession(id: string) {
    setLoading(true); setThinkingLabel("Loading session...");
    try {
      const d = await getSession(id);
      setSessionId(id); setUploadedFiles(d.files ?? []);
      setDraftSpec(d.draft_spec ?? ""); setQuestions(d.questions_md ?? "");
      setAnswers(d.answers_md ?? ""); setSpec(d.spec_md ?? ""); setReactCode(d.react_code ?? "");
      setMetrics((d.meta as any)?.metrics ?? null);
      setOutputType(((d.meta as any)?.output_type as any) ?? "text");
      const h = d.heuristic ?? null;
      setHeuristicHint(h);
      const parsed = parseTypeGuess(d.questions_md ?? "");
      setShipInstructions(guessToShipDefault(parsed.guess, h));
      if (d.react_code) setPhase("done");
      else if (d.spec_md) setPhase("ship");
      else if (d.questions_md) setPhase("refine");
      else setPhase("brief");
      setStatus("");
    } catch (e: any) { setError(`Failed: ${e.message ?? e}`); }
    finally { setLoading(false); setThinkingLabel(""); }
  }

  // ---------- Start ----------
  async function doStart() {
    if (!pendingFiles.length && !userPrompt.trim()) { setError("Add files or describe what you want."); return; }
    setLoading(true); setThinkingLabel("DeepSeek-VL is analyzing your input...");
    try {
      const r = await startSession(pendingFiles, userPrompt);
      setSessionId(r.session_id); setUploadedFiles(r.files);
      setDraftSpec(r.draft_spec); setQuestions(r.questions_md);
      setAnswers(""); setSpec(""); setReactCode(""); setPendingFiles([]);
      await pullMetrics(r.session_id);
      const h = r.heuristic ?? null;
      setHeuristicHint(h);
      const parsed = parseTypeGuess(r.questions_md);
      setShipInstructions(guessToShipDefault(parsed.guess, h));
      setPhase("refine");
      setSuccess("Analysis complete. Review DeepSeek's classification and answer the questions.");
      const s = await listSessions(); setSessions(s);
    } catch (e: any) { setError(`Start failed: ${e.message ?? e}`); }
    finally { setLoading(false); setThinkingLabel(""); }
  }

  // ---------- Refine ----------
  async function doRefine() {
    if (!answers.trim()) { setError("Write your answers first."); return; }
    setLoading(true); setThinkingLabel("DeepSeek-VL is refining your spec...");
    try {
      const r = await refineSpec(sessionId, answers);
      setSpec(r.spec_md);
      await pullMetrics(sessionId);
      setPhase("ship");
      setSuccess("Spec refined. Review it, then ship when ready.");
    } catch (e: any) { setError(`Refine failed: ${e.message ?? e}`); }
    finally { setLoading(false); setThinkingLabel(""); }
  }

  // ---------- Ship ----------
  async function doShip() {
    const ok = window.confirm("Send spec to Kimi K2 via Groq? This will use API credits.");
    if (!ok) return;
    setLoading(true); setThinkingLabel("Kimi K2 is generating output...");
    try {
      const r = await shipToCloud(sessionId, "moonshotai/kimi-k2-instruct-0905", shipInstructions);
      setReactCode(r.react_code);
      setOutputType((r.output_type as any) ?? "text");
      await pullMetrics(sessionId);
      setPhase("done");
      setSuccess("Output generated successfully!");
    } catch (e: any) { setError(`Ship failed: ${e.message ?? e}`); }
    finally { setLoading(false); setThinkingLabel(""); }
  }

  // ---------- New ----------
  function doNewSession() {
    setSessionId(""); setUploadedFiles([]); setPendingFiles([]);
    setUserPrompt(""); setDraftSpec(""); setQuestions(""); setAnswers("");
    setSpec(""); setReactCode(""); setMetrics(null); setOutputType("text");
    setShipInstructions("Detailed solution or implementation");
    setHeuristicHint(null);
    setStatus(""); setThinkingLabel(""); setPhase("brief");
  }

  const { guess: typeGuess, questionsOnly } = parseTypeGuess(questions);

  // Build effective classification: prefer DeepSeek's TYPE_GUESS, fall back to heuristic
  const effectiveGuess: TypeGuess | null = typeGuess
    ?? (heuristicHint?.domain
      ? { kind: `${heuristicHint.domain} (auto-detected)`, domain: heuristicHint.domain, confidence: heuristicHint.confidence }
      : null);

  /* ==================================================================
     LOGIN
  ================================================================== */
  if (phase === "login") {
    return (
      <div className="relative flex min-h-[60vh] items-center justify-center omni-enter">
        <Card className="w-full max-w-md text-center">
          <div className="mb-6">
            <div className="mb-2 text-2xl font-semibold tracking-tight">Welcome back</div>
            <p className="text-sm text-zinc-500">Enter your password to continue.</p>
          </div>
          <div className="flex gap-3">
            <input className="flex-1 rounded-xl border border-zinc-800 bg-zinc-950 px-4 py-2.5 text-sm outline-none transition-colors placeholder:text-zinc-600 focus:border-zinc-600" type="password" value={password} onChange={(e) => setPassword(e.target.value)} onKeyDown={(e) => e.key === "Enter" && doLogin()} placeholder="Password" autoFocus />
            <Btn variant="primary" onClick={doLogin} disabled={loading}>Login</Btn>
          </div>
          <ThinkingTimer active={loading} label="Authenticating" />
          {status && <div className="mt-4"><StatusBar text={status} variant={statusVariant} /></div>}
        </Card>
      </div>
    );
  }

  /* ==================================================================
     BRIEF
  ================================================================== */
  if (phase === "brief") {
    return (
      <div className="relative space-y-6 omni-enter">
        {sessions.length > 0 && (
          <Card>
            <div className="flex flex-wrap items-center gap-3">
              <span className="text-xs font-semibold uppercase tracking-widest text-zinc-500">Resume session</span>
              <select className="rounded-xl border border-zinc-800 bg-zinc-950 px-3 py-2 text-sm text-zinc-300 outline-none" value="" onChange={(e) => e.target.value && doResumeSession(e.target.value)}>
                <option value="">Select a previous session...</option>
                {sessions.map((s) => <option key={s.session_id} value={s.session_id}>{s.title}</option>)}
              </select>
            </div>
          </Card>
        )}

        <Card>
          <h2 className="mb-1 text-lg font-semibold tracking-tight">What are we building?</h2>
          <p className="mb-5 text-sm text-zinc-500">Upload sketches, PDFs, or reference files and describe what you want. DeepSeek-VL will classify the problem and ask clarifying questions.</p>
          <div className="space-y-5">
            <DropZone files={pendingFiles} onFilesChange={setPendingFiles} />
            <div className="flex flex-col gap-2">
              <label className="text-[11px] font-semibold uppercase tracking-widest text-zinc-500">Your prompt / description</label>
              <textarea rows={4} className="w-full resize-y rounded-xl border border-zinc-800/60 bg-zinc-950/60 px-4 py-3 text-[13px] leading-relaxed text-zinc-200 placeholder:text-zinc-600 outline-none transition-colors focus:border-zinc-600 focus:bg-zinc-950" value={userPrompt} onChange={(e) => setUserPrompt(e.target.value)} placeholder="e.g. Solve these linear algebra problems from the attached PDF..." />
            </div>
            <div className="flex items-center gap-4">
              <Btn variant="primary" onClick={doStart} disabled={loading}>{loading ? "Analyzing..." : "Start Analysis"}</Btn>
              <span className="text-[11px] text-zinc-600">Runs locally on your RTX 4070</span>
            </div>
          </div>
        </Card>

        <ThinkingTimer active={loading} label={thinkingLabel} />
        {status && <StatusBar text={status} variant={statusVariant} />}
      </div>
    );
  }

  /* ==================================================================
     REFINE
  ================================================================== */
  if (phase === "refine") {
    return (
      <div className="relative space-y-6 omni-enter">
        <div className="flex items-center justify-between">
          <h2 className="text-lg font-semibold tracking-tight">Refine your spec</h2>
          <div className="flex items-center gap-2">
            <Btn className="!px-3 !py-1.5 !text-xs" onClick={() => exportSessionZip(sessionId)} disabled={!sessionId}>Export (.zip)</Btn>
            <Btn onClick={doNewSession}>+ New</Btn>
          </div>
        </div>

        {uploadedFiles.length > 0 && (
          <Card className="!p-4">
            <div className="mb-2 text-[11px] font-semibold uppercase tracking-widest text-zinc-500">Uploaded files</div>
            <UploadedFilesSummary files={uploadedFiles} />
          </Card>
        )}

        {/* DeepSeek's classification (or heuristic fallback) */}
        {effectiveGuess && <TypeGuessCard guess={effectiveGuess} />}

        {/* Perf badges */}
        <div className="flex flex-wrap gap-2">
          <PerfBadge label="Draft spec" ms={metrics?.deepseek_initial_ms} tokens={metrics?.deepseek_initial_tokens_est} />
          <PerfBadge label="Questions" ms={metrics?.deepseek_questions_ms} tokens={metrics?.deepseek_questions_tokens_est} />
          {metrics?.deepseek_refine_ms !== undefined && <PerfBadge label="Last refine" ms={metrics.deepseek_refine_ms} tokens={metrics.deepseek_refine_tokens_est} />}
        </div>

        {/* Draft spec (collapsible) */}
        {draftSpec && (
          <details className="group">
            <summary className="cursor-pointer text-xs font-semibold uppercase tracking-widest text-zinc-500 hover:text-zinc-300 transition-colors">Draft spec (from DeepSeek-VL)</summary>
            <Card className="mt-2">
              <div className="flex justify-end mb-2"><CopyBtn text={draftSpec} label="Copy spec" /></div>
              <RenderedBlock text={draftSpec} />
            </Card>
          </details>
        )}

        {/* Questions */}
        <Card>
          <div className="mb-3 flex items-center justify-between">
            <span className="text-[11px] font-semibold uppercase tracking-widest text-zinc-500">DeepSeek-VL asks</span>
            <CopyBtn text={questionsOnly || questions} label="Copy questions" />
          </div>
          <RenderedBlock text={questionsOnly || questions || "No questions generated yet."} />
        </Card>

        {/* Answers */}
        <Card>
          <div className="space-y-4">
            <div className="flex flex-col gap-2">
              <label className="text-[11px] font-semibold uppercase tracking-widest text-zinc-500">Your answers</label>
              <textarea rows={8} className="w-full resize-y rounded-xl border border-zinc-800/60 bg-zinc-950/60 px-4 py-3 text-[13px] leading-relaxed text-zinc-200 placeholder:text-zinc-600 outline-none transition-colors focus:border-zinc-600 focus:bg-zinc-950 font-mono" value={answers} onChange={(e) => setAnswers(e.target.value)} placeholder={"Answer each question:\n- Q1: ...\n  A: your answer\n- Q2: ...\n  A: your answer"} />
            </div>
            <div className="flex items-center justify-between gap-4">
              <Btn variant="primary" onClick={doRefine} disabled={loading}>{loading ? "Refining..." : "Refine Spec"}</Btn>
              <CopyBtn text={answers} label="Copy answers" />
            </div>
          </div>
        </Card>

        <ThinkingTimer active={loading} label={thinkingLabel} />
        {status && <StatusBar text={status} variant={statusVariant} />}
      </div>
    );
  }

  /* ==================================================================
     SHIP
  ================================================================== */
  if (phase === "ship") {
    return (
      <div className="relative space-y-6 omni-enter">
        <div className="flex items-center justify-between">
          <h2 className="text-lg font-semibold tracking-tight">Review & Ship</h2>
          <div className="flex items-center gap-2">
            <Btn className="!px-3 !py-1.5 !text-xs" onClick={() => exportSessionZip(sessionId)} disabled={!sessionId}>Export (.zip)</Btn>
            <Btn onClick={doNewSession}>+ New</Btn>
          </div>
        </div>

        {effectiveGuess && <TypeGuessCard guess={effectiveGuess} />}

        <div className="flex flex-wrap gap-2">
          <PerfBadge label="Draft spec" ms={metrics?.deepseek_initial_ms} tokens={metrics?.deepseek_initial_tokens_est} />
          <PerfBadge label="Refine" ms={metrics?.deepseek_refine_ms} tokens={metrics?.deepseek_refine_tokens_est} />
        </div>

        <Card>
          <div className="mb-3 flex items-center justify-between">
            <span className="text-[11px] font-semibold uppercase tracking-widest text-zinc-500">Final spec</span>
            <CopyBtn text={spec} label="Copy spec" />
          </div>
          <div className="max-h-[400px] overflow-y-auto">
            <RenderedBlock text={spec} />
          </div>
        </Card>

        <div className="space-y-4">
          <div className="flex flex-col gap-2">
            <label className="text-[11px] font-semibold uppercase tracking-widest text-zinc-500">What should Kimi output?</label>
            <input className="w-full rounded-xl border border-zinc-800 bg-zinc-950 px-4 py-2.5 text-sm outline-none transition-colors placeholder:text-zinc-600 focus:border-zinc-600" value={shipInstructions} onChange={(e) => setShipInstructions(e.target.value)} placeholder="e.g. Step-by-step solution, Python code, React component" />
            {effectiveGuess && (
              <div className="text-[11px] text-zinc-600">
                Auto-filled from classification: <span className="text-zinc-400">{effectiveGuess.kind}</span>
              </div>
            )}
          </div>
          <div className="flex items-center gap-4">
            <Btn onClick={() => setPhase("refine")}>Back to refine</Btn>
            <Btn variant="primary" onClick={doShip} disabled={loading}>{loading ? "Asking Kimi..." : "Ship to Kimi K2 (Groq)"}</Btn>
            <Btn className="!px-3 !py-1.5 !text-xs" onClick={() => downloadTextFile("omni-spec.txt", spec || draftSpec)} disabled={!spec && !draftSpec}>Download spec (.txt)</Btn>
          </div>
        </div>

        <ThinkingTimer active={loading} label={thinkingLabel} />
        {status && <StatusBar text={status} variant={statusVariant} />}
      </div>
    );
  }

  /* ==================================================================
     DONE
  ================================================================== */
  return (
    <div className="relative space-y-6 omni-enter">
      <div className="flex items-center justify-between">
        <h2 className="text-lg font-semibold tracking-tight">Output</h2>
        <div className="flex items-center gap-2">
          <Btn className="!px-3 !py-1.5 !text-xs" onClick={() => exportSessionZip(sessionId)} disabled={!sessionId}>Export (.zip)</Btn>
          <Btn onClick={doNewSession}>+ New Session</Btn>
        </div>
      </div>

      <div className="flex flex-wrap gap-2">
        <PerfBadge label="DeepSeek analysis" ms={metrics?.deepseek_initial_ms} tokens={metrics?.deepseek_initial_tokens_est} />
        <PerfBadge label="DeepSeek refine" ms={metrics?.deepseek_refine_ms} tokens={metrics?.deepseek_refine_tokens_est} />
        <PerfBadge label="Kimi K2" ms={metrics?.kimi_ms} tokens={metrics?.kimi_tokens_est} />
      </div>

      <Card>
        <div className="mb-3 flex items-center justify-between gap-3">
          <span className="text-[11px] font-semibold uppercase tracking-widest text-zinc-500">
            Kimi K2 output
            {outputType === "latex" && <span className="ml-2 rounded bg-amber-500/10 px-1.5 py-0.5 text-[9px] text-amber-400 font-mono">LaTeX</span>}
            {outputType === "code" && <span className="ml-2 rounded bg-blue-500/10 px-1.5 py-0.5 text-[9px] text-blue-400 font-mono">Code</span>}
            {outputType === "text" && <span className="ml-2 rounded bg-zinc-500/10 px-1.5 py-0.5 text-[9px] text-zinc-400 font-mono">Text</span>}
          </span>
          <div className="flex gap-2">
            <CopyBtn text={reactCode} />
            {outputType === "latex" && (
              <Btn className="!px-3 !py-1.5 !text-xs" onClick={() => downloadTextFile("omni-solution.tex", reactCode)} disabled={!reactCode}>Download (.tex)</Btn>
            )}
            {outputType === "code" && (
              <Btn className="!px-3 !py-1.5 !text-xs" onClick={() => downloadTextFile("omni-output.txt", reactCode)} disabled={!reactCode}>Download (.txt)</Btn>
            )}
            {outputType === "text" && (
              <Btn className="!px-3 !py-1.5 !text-xs" onClick={() => downloadTextFile("omni-output.txt", reactCode)} disabled={!reactCode}>Download (.txt)</Btn>
            )}
          </div>
        </div>
        <div className="max-h-[600px] overflow-y-auto">
          {outputType === "latex" ? (
            <div className="rounded-lg border border-zinc-800/60 bg-zinc-950/80 overflow-x-auto">
              <div className="px-3 py-1 text-[10px] uppercase tracking-wider text-amber-500/70 border-b border-zinc-800/40">LaTeX — compile with pdflatex to get PDF</div>
              <pre className="px-4 py-3 text-[13px] leading-relaxed text-emerald-300/90 font-mono whitespace-pre-wrap">{reactCode}</pre>
            </div>
          ) : outputType === "code" ? (
            <div className="rounded-lg border border-zinc-800/60 bg-zinc-950/80 overflow-x-auto">
              <div className="px-3 py-1 text-[10px] uppercase tracking-wider text-blue-500/70 border-b border-zinc-800/40">Source code</div>
              <pre className="px-4 py-3 text-[13px] leading-relaxed text-emerald-300/90 font-mono whitespace-pre-wrap">{reactCode}</pre>
            </div>
          ) : (
            <RenderedBlock text={reactCode} />
          )}
        </div>
      </Card>

      {spec && (
        <details>
          <summary className="cursor-pointer text-xs font-semibold uppercase tracking-widest text-zinc-500 hover:text-zinc-300 transition-colors">View final spec</summary>
          <Card className="mt-2 space-y-3">
            <div className="flex items-center justify-between">
              <span className="text-[11px] font-semibold uppercase tracking-widest text-zinc-500">Final spec</span>
              <CopyBtn text={spec} label="Copy spec" />
            </div>
            <RenderedBlock text={spec} />
          </Card>
        </details>
      )}

      {status && <StatusBar text={status} variant={statusVariant} />}
    </div>
  );
}
