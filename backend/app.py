import os
import re
import json
import uuid
import time
import secrets
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional, Dict, Any, List
from io import BytesIO
import zipfile

import fitz  # PyMuPDF
import jwt
from dotenv import load_dotenv
from fastapi import (
    FastAPI, UploadFile, File, Form,
    HTTPException, Depends, Response, Request,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

import torch
from transformers import AutoModelForCausalLM
from deepseek_vl.models import VLChatProcessor, MultiModalityCausalLM
from deepseek_vl.utils.io import load_pil_images

from groq import Groq

# ---------------------------------------------------------------------------
#  Config
# ---------------------------------------------------------------------------

_project_root = Path(__file__).resolve().parent.parent
load_dotenv(_project_root / ".env")

APP_TITLE = "Omni-Architect Backend"
SESSIONS_DIR = _project_root / "specs" / "sessions"
SESSIONS_DIR.mkdir(parents=True, exist_ok=True)

JWT_SECRET = os.getenv("OMNI_JWT_SECRET", "")
OMNI_PASSWORD = os.getenv("OMNI_PASSWORD", "")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
FRONTEND_ORIGIN = os.getenv("OMNI_FRONTEND_ORIGIN", "http://localhost:3003")

if not JWT_SECRET:
    JWT_SECRET = "dev-only-change-me"


def _normalize_env_secret(value: str) -> str:
    v = (value or "").strip()
    if len(v) >= 2 and ((v[0] == "(" and v[-1] == ")") or (v[0] == v[-1] and v[0] in ("'", '"'))):
        v = v[1:-1].strip()
    return v


OMNI_PASSWORD = _normalize_env_secret(OMNI_PASSWORD)
if not OMNI_PASSWORD:
    OMNI_PASSWORD = ""

print(f"[boot] OMNI_PASSWORD loaded: {'YES (' + str(len(OMNI_PASSWORD)) + ' chars)' if OMNI_PASSWORD else 'NO (auth disabled)'}")

JWT_ALG = "HS256"
AUTH_COOKIE_NAME = "omni_auth"
CSRF_COOKIE_NAME = "omni_csrf"
TOKEN_TTL_HOURS = 12

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".gif"}
PDF_EXTENSIONS = {".pdf"}
ALLOWED_EXTENSIONS = IMAGE_EXTENSIONS | PDF_EXTENSIONS
MAX_FILE_SIZE = 20 * 1024 * 1024  # 20 MB per file
MAX_FILES_PER_SESSION = 15

security = HTTPBearer(auto_error=False)

# ---------------------------------------------------------------------------
#  Helpers
# ---------------------------------------------------------------------------

def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _session_path(session_id: str) -> Path:
    if not re.fullmatch(r"[a-f0-9\-]{10,64}", session_id):
        raise HTTPException(status_code=400, detail="Invalid session id")
    return SESSIONS_DIR / session_id


def _write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _update_session_meta(session_path: Path, updates: Dict[str, Any]) -> Dict[str, Any]:
    meta_path = session_path / "meta.json"
    meta = _read_json(meta_path) if meta_path.exists() else {}
    # Nested metrics dict to avoid clobbering unrelated meta fields
    metrics = meta.get("metrics", {})
    metrics.update(updates.get("metrics", {}))
    meta["metrics"] = metrics
    for k, v in updates.items():
        if k == "metrics":
            continue
        meta[k] = v
    _write_json(meta_path, meta)
    return meta

# ---------------------------------------------------------------------------
#  Auth / Security
# ---------------------------------------------------------------------------

def _jwt_issue() -> str:
    exp = _now_utc() + timedelta(hours=TOKEN_TTL_HOURS)
    payload = {
        "exp": int(exp.timestamp()),
        "iat": int(_now_utc().timestamp()),
        "role": "admin",
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALG)


def _jwt_verify(token: str) -> None:
    try:
        jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALG])
    except Exception:
        raise HTTPException(status_code=401, detail="Unauthorized")


def _require_csrf(request: Request) -> None:
    if not OMNI_PASSWORD:
        return
    cookie_token = request.cookies.get(CSRF_COOKIE_NAME, "")
    header_token = request.headers.get("x-csrf-token", "")
    if not cookie_token or not header_token or cookie_token != header_token:
        raise HTTPException(status_code=403, detail="CSRF check failed")


class _RateLimiter:
    def __init__(self) -> None:
        self._hits: Dict[str, List[float]] = {}

    def check(self, key: str, limit: int, window_sec: int) -> None:
        now = time.time()
        window_start = now - window_sec
        q = self._hits.get(key, [])
        q = [t for t in q if t >= window_start]
        if len(q) >= limit:
            raise HTTPException(status_code=429, detail="Rate limit exceeded")
        q.append(now)
        self._hits[key] = q


rate_limiter = _RateLimiter()


def require_auth(
    request: Request,
    creds: Optional[HTTPAuthorizationCredentials] = Depends(security),
) -> None:
    if not OMNI_PASSWORD:
        return
    cookie_token = request.cookies.get(AUTH_COOKIE_NAME, "")
    header_token = creds.credentials if creds else ""
    token = cookie_token or header_token
    if not token:
        raise HTTPException(status_code=401, detail="Unauthorized")
    _jwt_verify(token)


def require_auth_and_csrf(
    request: Request,
    creds: Optional[HTTPAuthorizationCredentials] = Depends(security),
) -> None:
    require_auth(request, creds)
    _require_csrf(request)

# ---------------------------------------------------------------------------
#  Request / Response Models
# ---------------------------------------------------------------------------

class LoginRequest(BaseModel):
    password: str = Field(min_length=1, max_length=256)


class RefineSpecRequest(BaseModel):
    session_id: str = Field(min_length=10, max_length=64, pattern=r"^[a-f0-9\-]+$")
    answers_md: str = Field(max_length=50_000)


class ShipRequest(BaseModel):
    session_id: str = Field(min_length=10, max_length=64, pattern=r"^[a-f0-9\-]+$")
    model_name: str = Field(default="moonshotai/kimi-k2-instruct-0905", max_length=128)
    instructions: str = Field(
        default="React + Tailwind component",
        max_length=400,
        description="What Kimi should output (language / framework / format).",
    )


class SessionSummary(BaseModel):
    session_id: str
    created_at: str
    title: str

# ---------------------------------------------------------------------------
#  PDF Text Extraction
# ---------------------------------------------------------------------------

def _extract_pdf_text(pdf_path: Path, max_pages: int = 50) -> str:
    doc = fitz.open(str(pdf_path))
    pages = []
    for i, page in enumerate(doc):
        if i >= max_pages:
            pages.append(f"\n[... truncated after {max_pages} pages ...]")
            break
        pages.append(page.get_text())
    doc.close()
    return "\n".join(pages).strip()

# ---------------------------------------------------------------------------
#  File Storage
# ---------------------------------------------------------------------------

def _sanitize_filename(name: str) -> str:
    name = re.sub(r'[<>:"/\\|?*\x00-\x1f]', '_', name)
    if len(name) > 100:
        stem = Path(name).stem[:80]
        ext = Path(name).suffix
        name = stem + ext
    return name


async def _save_uploaded_files(
    session_path: Path,
    files: List[UploadFile],
    existing_files: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    files_dir = session_path / "files"
    files_dir.mkdir(parents=True, exist_ok=True)

    total = len(existing_files)
    new_records: List[Dict[str, Any]] = []

    for f in files:
        if total >= MAX_FILES_PER_SESSION:
            break
        if not f.filename:
            continue

        ext = Path(f.filename).suffix.lower()
        if ext not in ALLOWED_EXTENSIONS:
            continue

        content = await f.read()
        if len(content) > MAX_FILE_SIZE:
            continue

        safe_name = _sanitize_filename(f.filename)
        stored_name = f"{uuid.uuid4().hex[:8]}_{safe_name}"
        dest = files_dir / stored_name
        dest.write_bytes(content)

        file_type = "pdf" if ext in PDF_EXTENSIONS else "image"
        record: Dict[str, Any] = {
            "original_name": f.filename,
            "stored_name": stored_name,
            "type": file_type,
            "size": len(content),
        }

        if file_type == "pdf":
            try:
                pdf_text = _extract_pdf_text(dest)
                txt_name = stored_name + ".txt"
                (files_dir / txt_name).write_text(pdf_text, encoding="utf-8")
                record["text_file"] = txt_name
                record["page_count"] = fitz.open(str(dest)).page_count
            except Exception as exc:
                record["extraction_error"] = str(exc)

        new_records.append(record)
        total += 1

    return new_records

# ---------------------------------------------------------------------------
#  DeepSeek-VL Model Loading & Generation
# ---------------------------------------------------------------------------

def _load_local_models() -> Dict[str, Any]:
    if hasattr(_load_local_models, "_cache"):
        return getattr(_load_local_models, "_cache")

    model_path = "deepseek-ai/deepseek-vl-1.3b-chat"
    processor: VLChatProcessor = VLChatProcessor.from_pretrained(model_path, use_fast=True)
    tokenizer = processor.tokenizer
    with torch.device("cpu"):
        vl_gpt: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            low_cpu_mem_usage=False,
            _fast_init=False,
            device_map=None,
        )
    vl_gpt = vl_gpt.to(torch.bfloat16).cuda().eval()

    cache = {"model_path": model_path, "processor": processor, "tokenizer": tokenizer, "vl_gpt": vl_gpt}
    setattr(_load_local_models, "_cache", cache)
    return cache


def _deepseek_vision_generate(
    image_path: Path, prompt: str, max_new_tokens: int,
    temperature: float = 0.0, top_p: float = 1.0,
) -> str:
    models = _load_local_models()
    processor: VLChatProcessor = models["processor"]
    tokenizer = models["tokenizer"]
    vl_gpt: MultiModalityCausalLM = models["vl_gpt"]

    conversation = [
        {"role": "User", "content": prompt, "images": [str(image_path)]},
        {"role": "Assistant", "content": ""},
    ]
    pil_images = load_pil_images(conversation)
    prepare_inputs = processor(conversations=conversation, images=pil_images, force_batchify=True).to(vl_gpt.device)

    sampling = temperature > 0
    gen_kwargs: Dict[str, Any] = {
        "inputs_embeds": vl_gpt.prepare_inputs_embeds(**prepare_inputs),
        "attention_mask": prepare_inputs.attention_mask,
        "pad_token_id": tokenizer.eos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "max_new_tokens": max_new_tokens,
        "do_sample": sampling,
    }
    if sampling:
        gen_kwargs["temperature"] = temperature
        gen_kwargs["top_p"] = top_p

    with torch.no_grad():
        outputs = vl_gpt.language_model.generate(**gen_kwargs)
    return tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True)


def _deepseek_text_generate(
    prompt: str, max_new_tokens: int,
    temperature: float = 0.0, top_p: float = 1.0,
) -> str:
    models = _load_local_models()
    tokenizer = models["tokenizer"]
    vl_gpt: MultiModalityCausalLM = models["vl_gpt"]

    enc = tokenizer(prompt, return_tensors="pt")
    input_ids = enc.input_ids.to(vl_gpt.device)
    attention_mask = enc.attention_mask.to(vl_gpt.device)

    sampling = temperature > 0
    gen_kwargs: Dict[str, Any] = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "max_new_tokens": max_new_tokens,
        "do_sample": sampling,
        "pad_token_id": tokenizer.eos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if sampling:
        gen_kwargs["temperature"] = temperature
        gen_kwargs["top_p"] = top_p

    with torch.no_grad():
        outputs = vl_gpt.language_model.generate(**gen_kwargs)
    new_tokens = outputs[0][input_ids.shape[1]:]
    return tokenizer.decode(new_tokens.cpu().tolist(), skip_special_tokens=True)


def _build_context(session_path: Path) -> Dict[str, Any]:
    """Gather all file context for a session: images list, aggregated PDF text, user prompt."""
    meta = _read_json(session_path / "meta.json")
    files_info = meta.get("files", [])
    user_prompt = meta.get("user_prompt", "")

    images = [f for f in files_info if f["type"] == "image"]
    pdfs = [f for f in files_info if f["type"] == "pdf"]

    pdf_text_parts: List[str] = []
    for pdf in pdfs:
        txt_file = pdf.get("text_file")
        if txt_file:
            txt_path = session_path / "files" / txt_file
            if txt_path.exists():
                text = txt_path.read_text(encoding="utf-8").strip()
                if text:
                    pdf_text_parts.append(f"--- Document: {pdf['original_name']} ---\n{text}\n---")

    return {
        "images": images,
        "pdfs": pdfs,
        "pdf_context": "\n\n".join(pdf_text_parts),
        "user_prompt": user_prompt,
        "files_dir": session_path / "files",
    }


# ---------------------------------------------------------------------------
#  Heuristic domain classifier
#  Each entry: (domain_label, [weighted_markers])
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
#  Heuristic domain classifier
#  Each entry: (domain_label, [weighted_markers])
#  A marker is either a plain string (weight=1) or a (string, weight) tuple.
#  We score every domain and return the one with the highest score if it
#  clears a minimum threshold, along with a rough confidence level.
# ---------------------------------------------------------------------------

_DOMAIN_RULES: List[tuple] = [
    (
        "linear algebra",
        [
            ("basis", 3), ("subspace", 3), ("vector space", 3),
            ("null space", 3), ("column space", 3), ("row space", 3),
            ("eigenvalue", 4), ("eigenvector", 4),
            ("determinant", 3), ("orthogonal", 2), ("orthonormal", 2),
            ("diagonaliz", 3), ("characteristic polynomial", 4),
            ("linear combination", 3), ("linearly independent", 3),
            ("linearly dependent", 3), ("span", 2),
            ("matrix", 2), ("matrices", 2),
            ("r^2", 2), ("r^3", 2), ("r^4", 2), ("r^n", 3),
            ("rank", 2), ("nullity", 3),
            ("inner product", 2), ("dot product", 2), ("cross product", 2),
            ("projection", 2), ("gram-schmidt", 4),
            ("augmented matrix", 3), ("row reduce", 3), ("row echelon", 3),
            ("reduced echelon", 3), ("pivot", 2),
            ("system of equations", 2), ("gaussian elimination", 3),
            ("least squares", 3), ("singular value", 3), ("svd", 3),
            ("lu decomposition", 3), ("qr decomposition", 3),
            ("trace", 1), ("transpose", 2), ("invertible", 2), ("inverse", 1),
        ],
    ),
    (
        "calculus",
        [
            ("derivative", 4), ("differentiat", 4), ("antiderivative", 4),
            ("integral", 4), ("integrat", 3),
            ("d/dx", 4), (" dx", 2), (" dy", 2),
            ("limit", 3), ("lim ", 3), ("lim(", 3),
            ("chain rule", 4), ("product rule", 4), ("quotient rule", 4),
            ("taylor series", 4), ("maclaurin", 4), ("power series", 3),
            ("partial derivative", 4), ("gradient", 2), ("divergence", 3), ("curl", 2),
            ("fundamental theorem", 4),
            ("riemann sum", 4), ("improper integral", 3),
            ("l'hopital", 4), ("l'hôpital", 4),
            ("critical point", 3), ("inflection point", 3),
            ("concave", 2), ("convex", 1),
            ("related rates", 3), ("optimization", 2),
            ("differential equation", 4), ("ode", 3), ("pde", 3),
        ],
    ),
    (
        "probability / statistics",
        [
            ("probability", 4), ("p(", 2), ("p(a", 2), ("p(b", 2),
            ("random variable", 4), ("expected value", 4), ("expectation", 3),
            ("variance", 3), ("standard deviation", 3), ("covariance", 3),
            ("distribution", 3), ("pmf", 4), ("pdf", 2), ("cdf", 3),
            ("binomial", 4), ("poisson", 4), ("geometric", 2),
            ("normal distribution", 4), ("gaussian", 3), ("uniform distribution", 3),
            ("exponential distribution", 3), ("chi-squared", 3), ("t-distribution", 3),
            ("z-score", 3), ("confidence interval", 3), ("hypothesis test", 4),
            ("null hypothesis", 4), ("p-value", 4), ("significance level", 3),
            ("bayes", 4), ("conditional probability", 4), ("independence", 2),
            ("law of large numbers", 4), ("central limit theorem", 4),
            ("moment generating", 4), ("markov", 3),
            ("regression", 2), ("correlation", 2),
        ],
    ),
    (
        "discrete math / combinatorics",
        [
            ("permutation", 4), ("combination", 3), ("factorial", 3),
            ("n choose k", 4), ("binomial coefficient", 4),
            ("graph theory", 4), ("vertex", 2), ("vertices", 2), ("edge", 1), ("edges", 1),
            ("bipartite", 4), ("isomorphic", 3), ("adjacency matrix", 4),
            ("tree", 2), ("spanning tree", 4), ("planar graph", 4),
            ("euler", 2), ("hamiltonian", 3),
            ("recurrence relation", 4), ("closed form", 2),
            ("big-o", 3), ("asymptotic", 3),
            ("boolean", 2), ("logic gate", 3), ("truth table", 4),
            ("proof by induction", 4), ("induction", 2),
            ("pigeonhole", 4), ("inclusion-exclusion", 4),
            ("set theory", 3), ("power set", 3), ("cardinality", 3),
        ],
    ),
    (
        "programming / algorithms",
        [
            ("def ", 3), ("function(", 2), ("class ", 2), ("return ", 2),
            ("if __name__", 4), ("import ", 2), ("from ", 1),
            ("python", 3), ("java", 2), ("javascript", 3), ("typescript", 3),
            ("rust", 2), ("c++", 3), ("golang", 3), ("ruby", 2), ("swift", 2),
            ("for loop", 3), ("while loop", 3), ("for (", 2), ("while (", 2),
            ("array", 2), ("linked list", 3), ("binary tree", 3), ("binary search", 3),
            ("sort", 2), ("merge sort", 3), ("quicksort", 3),
            ("hash map", 3), ("hash table", 3), ("dictionary", 2),
            ("recursion", 3), ("dynamic programming", 4), ("memoization", 3),
            ("time complexity", 3), ("space complexity", 3),
            ("api", 2), ("http", 2), ("rest", 2), ("json", 2),
            ("sql", 3), ("database", 2), ("query", 2),
            ("react", 3), ("component", 2), ("typescript", 3), ("tailwind", 3),
        ],
    ),
    (
        "physics",
        [
            ("velocity", 3), ("acceleration", 3), ("force", 2), ("momentum", 3),
            ("newton", 3), ("friction", 2), ("torque", 3), ("angular", 2),
            ("kinetic energy", 3), ("potential energy", 3), ("work-energy", 3),
            ("electric field", 3), ("magnetic field", 3), ("coulomb", 3),
            ("circuit", 2), ("voltage", 2), ("current", 2), ("resistance", 2),
            ("ohm's law", 4), ("kirchhoff", 4),
        ],
    ),
    (
        "ui / product design",
        [
            ("component", 2), ("layout", 2), ("navbar", 3), ("sidebar", 3),
            ("button", 2), ("modal", 3), ("dropdown", 3), ("form", 2),
            ("responsive", 3), ("mobile", 2), ("desktop", 2), ("wireframe", 4),
            ("mockup", 4), ("prototype", 3), ("user flow", 4),
            ("typography", 3), ("color palette", 3), ("hex", 2), ("spacing", 2),
            ("grid", 2), ("flexbox", 3), ("css", 3), ("tailwind", 3),
            ("dark mode", 3), ("light mode", 3), ("theme", 2),
            ("ux", 3), ("ui", 2), ("design system", 4),
            ("figma", 4), ("sketch", 3), ("adobe xd", 4),
        ],
    ),
    (
        "chemistry",
        [
            ("mole", 3), ("molar mass", 4), ("molarity", 4), ("stoichiometry", 4),
            ("chemical equation", 4), ("balanced equation", 4), ("reaction", 2),
            ("oxidation", 3), ("reduction", 3), ("redox", 4),
            ("acid", 2), ("base", 2), ("ph", 3), ("poh", 3), ("buffer", 2),
            ("titration", 4), ("equilibrium constant", 4), ("keq", 3), ("ka", 2), ("kb", 2),
            ("enthalpy", 4), ("entropy", 4), ("gibbs", 4), ("thermodynamics", 3),
            ("electron configuration", 4), ("orbital", 3), ("bond", 2), ("covalent", 3),
            ("ionic", 3), ("lewis structure", 4), ("vsepr", 4), ("hybridization", 4),
            ("periodic table", 3), ("element", 1), ("isotope", 3),
            ("gas law", 4), ("ideal gas", 4), ("pv=nrt", 4), ("dalton", 3),
            ("organic chemistry", 4), ("functional group", 4), ("nomenclature", 3),
            ("electrochemistry", 4), ("galvanic", 4), ("electrolysis", 4),
        ],
    ),
    (
        "biology",
        [
            ("cell", 2), ("mitosis", 4), ("meiosis", 4), ("dna", 3), ("rna", 3),
            ("protein", 2), ("amino acid", 3), ("enzyme", 3), ("substrate", 2),
            ("gene", 3), ("allele", 4), ("genotype", 4), ("phenotype", 4),
            ("chromosome", 3), ("mutation", 3), ("transcription", 3), ("translation", 3),
            ("photosynthesis", 4), ("cellular respiration", 4), ("atp", 3),
            ("krebs cycle", 4), ("glycolysis", 4), ("electron transport", 4),
            ("evolution", 3), ("natural selection", 4), ("speciation", 4),
            ("taxonomy", 3), ("phylogen", 3), ("cladogram", 4),
            ("ecology", 3), ("ecosystem", 3), ("food chain", 3), ("trophic", 3),
            ("homeostasis", 4), ("nervous system", 3), ("endocrine", 3),
            ("immune system", 3), ("antibody", 3), ("antigen", 3),
        ],
    ),
    (
        "writing / english",
        [
            ("essay", 4), ("thesis statement", 4), ("topic sentence", 4),
            ("paragraph", 2), ("introduction", 2), ("conclusion", 2), ("body paragraph", 3),
            ("argumentative", 4), ("persuasive", 4), ("expository", 4), ("narrative", 3),
            ("rhetorical", 4), ("rhetoric", 3), ("ethos", 4), ("pathos", 4), ("logos", 4),
            ("literary analysis", 4), ("theme", 2), ("motif", 3), ("symbolism", 3),
            ("metaphor", 3), ("simile", 3), ("irony", 3), ("allegory", 4),
            ("tone", 2), ("mood", 2), ("diction", 4), ("syntax", 2),
            ("citation", 3), ("mla", 4), ("apa", 4), ("chicago style", 4),
            ("bibliography", 3), ("works cited", 4), ("annotated bibliography", 4),
            ("creative writing", 4), ("fiction", 3), ("nonfiction", 3),
            ("poetry", 3), ("stanza", 4), ("verse", 2), ("sonnet", 4),
            ("protagonist", 3), ("antagonist", 3), ("plot", 2), ("character development", 4),
            ("point of view", 3), ("first person", 3), ("third person", 3),
            ("compare and contrast", 4), ("cause and effect", 3),
            ("grammar", 3), ("punctuation", 2), ("sentence structure", 3),
            ("write", 2), ("draft", 2), ("revise", 3), ("proofread", 3),
        ],
    ),
    (
        "history / social science",
        [
            ("century", 2), ("era", 2), ("civilization", 3), ("empire", 2),
            ("revolution", 3), ("war", 2), ("treaty", 3), ("constitution", 3),
            ("amendment", 3), ("democracy", 3), ("monarchy", 3), ("feudal", 3),
            ("colonialism", 4), ("imperialism", 4), ("industrialization", 4),
            ("primary source", 4), ("secondary source", 4), ("historiography", 4),
            ("cause and effect", 3), ("turning point", 3),
            ("economics", 3), ("supply and demand", 4), ("gdp", 3), ("inflation", 3),
            ("political science", 4), ("ideology", 3), ("sovereignty", 3),
            ("sociology", 3), ("anthropology", 3), ("psychology", 3),
            ("classical conditioning", 4), ("operant conditioning", 4),
            ("cognitive", 2), ("behavioral", 2),
        ],
    ),
]


def _score_domain(text_lower: str, markers: List) -> int:
    score = 0
    for m in markers:
        if isinstance(m, tuple):
            token, weight = m
        else:
            token, weight = m, 1
        if token in text_lower:
            score += weight
    return score


def _heuristic_domain_hint(session_path: Path) -> dict:
    """
    Returns a dict with:
      - domain (str): best-guess domain label, or "" if nothing scores above threshold
      - scores (dict): all domain scores for observability
      - confidence (str): "high" | "medium" | "low" based on margin over second-best
    """
    ctx = _build_context(session_path)
    text = (ctx.get("pdf_context", "") or "") + "\n" + (ctx.get("user_prompt", "") or "")
    text_lower = text.lower()

    if not text_lower.strip():
        return {"domain": "", "scores": {}, "confidence": "low"}

    scores: Dict[str, int] = {}
    for domain_label, markers in _DOMAIN_RULES:
        scores[domain_label] = _score_domain(text_lower, markers)

    # Sort by score descending
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    best_domain, best_score = ranked[0]
    second_score = ranked[1][1] if len(ranked) > 1 else 0

    # Minimum threshold to claim a domain; avoids low-signal noise
    THRESHOLD = 4
    if best_score < THRESHOLD:
        return {"domain": "", "scores": scores, "confidence": "low"}

    margin = best_score - second_score
    if margin >= 8 or (best_score >= 12 and margin >= 4):
        confidence = "high"
    elif margin >= 4:
        confidence = "medium"
    else:
        confidence = "low"

    return {"domain": best_domain, "scores": scores, "confidence": confidence}


def _math_index_is_sane(text: str) -> bool:
    """
    Very coarse check that a supposed math problem index is not obviously junk.
    - Requires at least one '## Problem' line.
    - Rejects outputs that are overwhelmingly a small set of tokens (e.g. '1'/'2').
    """
    t = (text or "").strip()
    if not t:
        return False

    # Must mention at least one problem heading
    if re.search(r"^##\\s*Problem\\b", t, flags=re.MULTILINE) is None:
        return False

    tokens = re.findall(r"\\S+", t)
    if not tokens:
        return False
    unique = set(tokens)
    # If it's long but has very few unique tokens, it's probably garbage
    if len(tokens) > 50 and len(unique) < 8:
        return False
    return True


def _make_math_fallback_index(source: str) -> str:
    """
    Deterministic fallback when DeepSeek fails to produce a problem index.
    We try to split the source text into Problem blocks and create a minimal
    index that Kimi can still use.
    """
    src = (source or "").replace("\r\n", "\n").replace("\r", "\n").strip()
    if not src:
        return (
            "# Math Problems – Unknown Count\n\n"
            "## Problem 1 – (unparsed)\n"
            "- Type: unknown\n"
            "- Given: (see original document text)\n"
            "- Find: (see original document text)\n"
            "- Tags: math\n"
            "- Subparts: unknown\n"
        )

    # Remove common boilerplate lines
    cleaned_lines = []
    for line in src.split("\n"):
        s = line.strip()
        if not s:
            continue
        if "webwork" in s.lower() or "mathematical association of america" in s.lower():
            continue
        if s.lower().startswith("generated by"):
            continue
        cleaned_lines.append(s)
    cleaned = "\n".join(cleaned_lines)

    # Split on "Problem N" style
    parts = re.split(r"(?im)^\s*problem\s+(\d+)\b.*$", cleaned)
    # re.split yields: [preamble, n1, block1, n2, block2, ...]
    problems = []
    if len(parts) >= 3:
        i = 1
        while i + 1 < len(parts):
            num = parts[i].strip()
            block = parts[i + 1].strip()
            problems.append((num, block))
            i += 2

    out = []
    out.append("# Math Problems – Parsed Index")
    out.append("")
    if problems:
        for num, block in problems[:30]:
            snippet = re.sub(r"\s+", " ", block)[:180].strip()
            out.append(f"## Problem {num} – (auto-parsed)")
            out.append("- Type: unknown")
            out.append(f"- Given: {snippet if snippet else '(see original document text)'}")
            out.append("- Find: (see original document text)")
            out.append("- Tags: math")
            out.append("- Subparts: unknown")
            out.append("")
    else:
        # If we couldn't detect problem numbers, create a single entry with a snippet.
        snippet = re.sub(r"\s+", " ", cleaned)[:220].strip()
        out.append("## Problem 1 – (auto-parsed)")
        out.append("- Type: unknown")
        out.append(f"- Given: {snippet if snippet else '(see original document text)'}")
        out.append("- Find: (see original document text)")
        out.append("- Tags: math")
        out.append("- Subparts: unknown")
        out.append("")

    out.append("## User Preferences")
    out.append("- Solutions: all problems")
    out.append("- Format: full step-by-step LaTeX")
    out.append("- Constraints: none specified")
    out.append("")
    return "\n".join(out).strip() + "\n"


def _spec_should_be_rejected(text: str) -> bool:
    """
    Generic garbage detector for any spec output.
    Reject if it's empty or extremely low-entropy.
    """
    t = (text or "").strip()
    if not t:
        return True
    tokens = re.findall(r"\S+", t)
    if len(tokens) >= 80 and len(set(tokens)) < 10:
        return True
    return False


def _code_spec_is_sane(text: str) -> bool:
    """
    For code tasks, DeepSeek must produce a *spec*, not code.
    Reject if code fences are present or it looks like source code.
    """
    t = (text or "").strip()
    if _spec_should_be_rejected(t):
        return False
    if "```" in t:
        return False
    # Heuristic: too many typical code tokens
    codeish = sum(1 for s in ["def ", "class ", "import ", "{", "};", "=>", "from fastapi"] if s in t.lower())
    if codeish >= 2:
        return False
    # Needs at least some structure
    if not (re.search(r"(?m)^#{1,3}\s+", t) or re.search(r"(?m)^\s*[-*]\s+", t)):
        return False
    return True


def _ui_spec_is_sane(text: str) -> bool:
    t = (text or "").lstrip()
    if _spec_should_be_rejected(t):
        return False
    return t.startswith("## UI_SPEC")


def _make_code_fallback_spec(user_prompt: str) -> str:
    up = (user_prompt or "").strip()
    return (
        "# CODE_SPEC\n\n"
        "## Goal\n"
        f"- {up if up else 'Implement the requested code task.'}\n\n"
        "## Requirements\n"
        "- Follow the requested language/framework.\n"
        "- Validate inputs and handle errors.\n"
        "- Include tests if requested.\n\n"
        "## Deliverables\n"
        "- Source code\n"
        "- Tests\n"
    )


def _generate_with_context(
    session_path: Path,
    task_instructions: str,
    max_new_tokens: int,
    extra_context: str = "",
    temperature: float = 0.0,
    top_p: float = 1.0,
) -> str:
    ctx = _build_context(session_path)

    prompt_parts: List[str] = [task_instructions]

    if ctx["user_prompt"]:
        prompt_parts.append(f"User's request:\n{ctx['user_prompt']}")

    if ctx["pdf_context"]:
        prompt_parts.append(f"Extracted document text:\n{ctx['pdf_context']}")

    if extra_context:
        prompt_parts.append(extra_context)

    full_prompt = "\n\n".join(prompt_parts)

    if ctx["images"]:
        image_path = ctx["files_dir"] / ctx["images"][0]["stored_name"]
        return _deepseek_vision_generate(
            image_path,
            "<image_placeholder>\n" + full_prompt,
            max_new_tokens,
            temperature=temperature,
            top_p=top_p,
        )
    else:
        return _deepseek_text_generate(
            full_prompt, max_new_tokens,
            temperature=temperature,
            top_p=top_p,
        )


def _extract_ui_spec_markdown(text: str) -> str:
    if not text:
        return ""
    t = text.replace("\r\n", "\n").replace("\r", "\n")
    m = re.search(r"(?m)^\s*##\s*UI_SPEC\s*$", t)
    if not m:
        return ""
    start = m.start()
    rest = t[m.end():]
    m2 = re.search(r"(?m)^\s*##\s+\S", rest)
    end = len(t) if not m2 else (m.end() + m2.start())
    spec = t[start:end].strip()
    return spec if spec.startswith("## UI_SPEC") else ""

# ---------------------------------------------------------------------------
#  FastAPI App
# ---------------------------------------------------------------------------

app = FastAPI(title=APP_TITLE)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[FRONTEND_ORIGIN],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> Dict[str, Any]:
    return {
        "ok": True,
        "auth_required": bool(OMNI_PASSWORD),
        "sessions_dir": str(SESSIONS_DIR),
    }


@app.post("/auth/login")
def login_route(body: LoginRequest, response: Response, request: Request) -> Dict[str, Any]:
    ip = "unknown"
    try:
        ip = request.client.host if request.client else "unknown"
    except Exception:
        pass
    rate_limiter.check(f"login:{ip}", limit=10, window_sec=60)

    if not OMNI_PASSWORD:
        token = _jwt_issue()
    else:
        if body.password != OMNI_PASSWORD:
            raise HTTPException(status_code=401, detail="Invalid password")
        token = _jwt_issue()

    csrf = secrets.token_urlsafe(32)
    response.set_cookie(
        AUTH_COOKIE_NAME, token,
        httponly=True, secure=False, samesite="lax",
        max_age=TOKEN_TTL_HOURS * 3600, path="/",
    )
    response.set_cookie(
        CSRF_COOKIE_NAME, csrf,
        httponly=False, secure=False, samesite="lax",
        max_age=TOKEN_TTL_HOURS * 3600, path="/",
    )
    return {"ok": True}


@app.post("/auth/logout")
def logout_route(response: Response) -> Dict[str, Any]:
    response.delete_cookie(AUTH_COOKIE_NAME, path="/")
    response.delete_cookie(CSRF_COOKIE_NAME, path="/")
    return {"ok": True}


# ---- Sessions ----

@app.get("/sessions", dependencies=[Depends(require_auth)])
def list_sessions() -> List[SessionSummary]:
    out: List[SessionSummary] = []
    if not SESSIONS_DIR.exists():
        return out
    for d in sorted(SESSIONS_DIR.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
        meta_path = d / "meta.json"
        if not meta_path.exists():
            continue
        meta = _read_json(meta_path)
        out.append(SessionSummary(
            session_id=d.name,
            created_at=meta.get("created_at", ""),
            title=meta.get("title", "Untitled"),
        ))
    return out


@app.post("/sessions/start", dependencies=[Depends(require_auth_and_csrf)])
async def start_session(
    user_prompt: str = Form(""),
    files: List[UploadFile] = File(default=[]),
) -> Dict[str, Any]:
    """
    All-in-one: create session, upload files (images + PDFs), and run initial
    DeepSeek-VL analysis. Returns session_id, file info, draft spec, and
    clarifying questions.
    """
    session_id = str(uuid.uuid4())
    sp = _session_path(session_id)
    sp.mkdir(parents=True, exist_ok=True)

    file_records = await _save_uploaded_files(sp, files, [])

    title = user_prompt[:60].strip() if user_prompt else f"Session {datetime.now().strftime('%Y-%m-%d %H:%M')}"
    meta: Dict[str, Any] = {
        "session_id": session_id,
        "created_at": _now_utc().isoformat(),
        "title": title,
        "user_prompt": user_prompt.strip(),
        "files": file_records,
    }
    _write_json(sp / "meta.json", meta)

    has_content = bool(file_records) or bool(user_prompt.strip())
    draft_spec = ""
    questions_md = ""

    if has_content:
        # Run heuristic early so spec instructions can adapt
        pre_hint = _heuristic_domain_hint(sp)
        pre_domain = (pre_hint.get("domain") or "").lower()

        is_math_content = any(s in pre_domain for s in
            ["algebra", "calculus", "probability", "statistics", "physics", "math",
             "discrete", "chemistry", "biology"])

        t0 = time.time()
        if is_math_content:
            spec_instructions = (
                "You will see raw text extracted from a math homework PDF.\n"
                "It may contain noisy formatting, page numbers, and boilerplate.\n\n"
                "YOUR JOB: build a CLEAN PROBLEM INDEX. This is NOT a copy of the text.\n"
                "You must ANALYZE the content and COMPRESS each problem into a short entry.\n\n"
                "STRICT OUTPUT FORMAT (follow exactly):\n"
                "# [Course / Subject] - [N] Problems\n\n"
                "## Problem 1 - [short descriptive name, e.g. 'Basis test in R^3']\n"
                "- Type: [category, e.g. basis determination, linear independence, coordinate vectors]\n"
                "- Given: [compact summary of data, e.g. 'Vectors v1=[1,0,0], v2=[1,1,0], v3=[1,1,1]']\n"
                "- Find: [what to determine, e.g. 'Is {v1,v2,v3} a basis for R^3?']\n"
                "- Has sub-parts: [yes/no, and how many]\n\n"
                "## Problem 2 - [short name]\n"
                "...\n\n"
                "ABSOLUTE RULES:\n"
                "- Do NOT copy the original question text verbatim.\n"
                "- Do NOT include boilerplate like 'Generated by WeBWorK' or page numbers.\n"
                "- Do NOT paste raw matrix columns as separate lines. Write them inline: [1,0,0].\n"
                "- Do NOT solve any problem.\n"
                "- EVERY problem must appear. Count them carefully.\n"
                "- Keep each entry to 3-5 lines max.\n"
                "\n"
                "GEOMETRY/PHYSICS PROBLEMS — extra detail required:\n"
                "- If a problem involves objects at specific positions (e.g. masses at corners of a rectangle),\n"
                "  list the EXACT coordinates or relative positions of every object AND the target point.\n"
                "- Note any obvious symmetries (e.g. 'target is at midpoint of bottom edge, so bottom-corner\n"
                "  masses are equidistant left and right → their horizontal forces cancel').\n"
                "- For banked-curve or friction problems, list ALL given speeds separately with labels\n"
                "  (e.g. 'design speed v1 = X m/s, actual speed v2 = Y m/s') so the solver can track subscripts.\n"
            )
        else:
            spec_instructions = (
                "You are a strict visual-to-structure translator.\n\n"
                "If the input is UI: output STRICT Markdown starting with '## UI_SPEC' "
                "using a component tree and colors table.\n"
                "If the input is a document: summarize the key requirements as a structured spec.\n"
                "If the input is code: output a STRUCTURED spec (NOT code) with clear headings and bullets.\n"
                "Never output code fences or source code.\n"
                "Rules: output ONLY the spec. No Q&A, no conversational filler.\n"
            )
        spec_max_tokens = 900 if is_math_content else 700
        raw_spec = _generate_with_context(sp, spec_instructions, max_new_tokens=spec_max_tokens)
        t1 = time.time()

        if is_math_content:
            # For math: keep the raw output as-is, but only if it looks like a real index
            candidate = raw_spec.strip()
            if _math_index_is_sane(candidate) and not _spec_should_be_rejected(candidate):
                draft_spec = candidate
            else:
                # Fallback: build a deterministic index from original context
                ctx = _build_context(sp)
                src = (ctx.get("pdf_context") or "") + "\n" + (ctx.get("user_prompt") or "")
                draft_spec = _make_math_fallback_index(src)
        else:
            # For UI: try to extract a clean UI_SPEC block; fall back to raw
            candidate = _extract_ui_spec_markdown(raw_spec) or raw_spec.strip()
            # If it looks like code or garbage for code tasks, fall back to a deterministic code spec.
            if (("python" in pre_domain) or ("programming" in pre_domain) or ("algorithm" in pre_domain)) and not _code_spec_is_sane(candidate):
                draft_spec = _make_code_fallback_spec(user_prompt)
            else:
                draft_spec = candidate

        (sp / "draft_spec.md").write_text(draft_spec + "\n", encoding="utf-8")

        hint = pre_hint
        domain_label = hint["domain"]
        domain_conf = hint["confidence"]

        # Build a crisp one-line hint sentence for DeepSeek to anchor on
        if domain_label:
            hint_sentence = (
                f"Pre-analysis (keyword scoring, confidence={domain_conf}): "
                f"this content is most likely '{domain_label}'. "
                "Use this as your starting guess in TYPE_GUESS; only override if clearly wrong."
            )
        else:
            hint_sentence = ""

        q_instructions = (
            "You are Omni-Architect's LOCAL Orchestrator.\n"
            "The user has uploaded content (a PDF, image, or text) and wants YOUR HELP with it.\n"
            "They are NOT asking you to repeat the content back. They want you to help them\n"
            "solve, build, or understand it.\n\n"
            "Your job has two parts:\n"
            "  1. Classify what kind of content this is (TYPE_GUESS).\n"
            "  2. Ask a FEW short clarifying questions about HOW the user wants help.\n\n"
            "IMPORTANT for QUESTIONS:\n"
            "- Do NOT repeat or restate the problems from the document.\n"
            "- Do NOT ask the user to answer the math/code problems themselves.\n"
            "- Instead ask things like:\n"
            "  - 'Do you want full step-by-step solutions or just final answers?'\n"
            "  - 'Should I show my work using LaTeX notation?'\n"
            "  - 'Are there specific problems you want me to focus on, or all of them?'\n"
            "  - 'What format do you want the output in (PDF-ready LaTeX, plain text, code)?'\n"
            "  - 'Is there a specific method or theorem you must use?'\n"
            "  - For code: 'What language? Any specific framework?'\n"
            "  - For UI: 'What screen size? Dark/light mode?'\n\n"
            "Classification hints:\n"
            "- 'basis', 'subspace', 'eigenvalue', 'matrix', 'R^n' -> linear algebra\n"
            "- 'derivative', 'integral', 'limit', 'dx' -> calculus\n"
            "- 'probability', 'random variable', 'distribution' -> probability/statistics\n"
            "- code syntax or language names -> programming\n"
            "- 'component', 'layout', 'wireframe', 'mockup' -> UI design\n"
            "ALWAYS make a best-effort guess. Never say 'unknown'.\n\n"
            "Your output MUST follow this EXACT structure:\n"
            "## TYPE_GUESS\n"
            "- kind: <e.g. 'linear algebra homework (8 problems)', 'React dashboard', 'calculus exam'>\n"
            "- language_or_domain: <e.g. 'linear algebra', 'Python', 'calculus'>\n"
            "- confidence: <low|medium|high>\n\n"
            "## QUESTIONS\n"
            "- Q1: <clarifying question about how the user wants help>\n"
            "- Q2: <clarifying question about format/method preferences>\n"
            "- (2-4 questions max, keep them short)\n"
        )
        t2 = time.time()
        raw_q = _generate_with_context(
            sp, q_instructions, max_new_tokens=380,
            extra_context=hint_sentence,
        )
        t3 = time.time()
        normalized = raw_q.replace("\r\n", "\n")
        m = re.search(r"(?m)^\s*##\s*TYPE_GUESS\s*$", normalized)
        n = re.search(r"(?m)^\s*##\s*QUESTIONS\s*$", normalized)
        if m and n:
            questions_md = normalized[m.start():].strip()
        elif n:
            questions_md = normalized[n.start():].strip()
        else:
            # Fallback: build TYPE_GUESS from heuristic + generic questions
            fb_kind = domain_label if domain_label else "general"
            fb_conf = domain_conf if domain_label else "low"
            questions_md = (
                "## TYPE_GUESS\n"
                f"- kind: {fb_kind} problem\n"
                f"- language_or_domain: {domain_label or 'unknown'}\n"
                f"- confidence: {fb_conf}\n\n"
                "## QUESTIONS\n"
                "- Q1: What specific aspect do you need help with?\n"
                "- Q2: Are there any constraints or special requirements?\n"
            )
        (sp / "questions.md").write_text(questions_md + "\n", encoding="utf-8")
        (sp / "answers.md").write_text("## ANSWERS\n", encoding="utf-8")

        # Update timing metrics + heuristic scores for observability
        _update_session_meta(
            sp,
            {
                "heuristic_domain": domain_label,
                "heuristic_confidence": domain_conf,
                "heuristic_scores": hint["scores"],
                "metrics": {
                    "deepseek_initial_ms": int((t1 - t0) * 1000),
                    "deepseek_questions_ms": int((t3 - t2) * 1000),
                    "deepseek_initial_tokens_est": len(draft_spec.split()),
                    "deepseek_questions_tokens_est": len(questions_md.split()),
                },
            },
        )

    heuristic_out = hint if has_content else {"domain": "", "confidence": "low"}

    return {
        "session_id": session_id,
        "files": file_records,
        "draft_spec": draft_spec,
        "questions_md": questions_md,
        "heuristic": {
            "domain": heuristic_out.get("domain", ""),
            "confidence": heuristic_out.get("confidence", "low"),
        },
    }


@app.post("/sessions/{session_id}/files", dependencies=[Depends(require_auth_and_csrf)])
async def add_files(session_id: str, files: List[UploadFile] = File(...)) -> Dict[str, Any]:
    """Add more files to an existing session."""
    sp = _session_path(session_id)
    meta_path = sp / "meta.json"
    if not meta_path.exists():
        raise HTTPException(status_code=404, detail="Session not found")
    meta = _read_json(meta_path)
    existing = meta.get("files", [])

    new_records = await _save_uploaded_files(sp, files, existing)
    meta["files"] = existing + new_records
    _write_json(meta_path, meta)

    return {"ok": True, "new_files": new_records, "total_files": meta["files"]}


@app.post("/sessions/refine", dependencies=[Depends(require_auth_and_csrf)])
def refine_spec(body: RefineSpecRequest) -> Dict[str, Any]:
    sp = _session_path(body.session_id)
    meta_path = sp / "meta.json"
    if not meta_path.exists():
        raise HTTPException(status_code=404, detail="Session not found")

    answers_md = body.answers_md.strip() if body.answers_md else ""
    (sp / "answers.md").write_text(answers_md + "\n", encoding="utf-8")

    meta = _read_json(meta_path)
    h_domain = (meta.get("heuristic_domain") or "").lower()
    is_math_refine = any(s in h_domain for s in
        ["algebra", "calculus", "probability", "statistics", "physics", "math",
         "discrete", "chemistry", "biology"])

    draft_spec = ""
    draft_path = sp / "draft_spec.md"
    if draft_path.exists():
        draft_spec = draft_path.read_text(encoding="utf-8").strip()

    if is_math_refine:
        spec_instructions = (
            "You are refining a PROBLEM INDEX for a math homework.\n"
            "You have two inputs:\n"
            "  1. The current draft index (may be incomplete or wrong).\n"
            "  2. The user's clarifying answers about what they need.\n\n"
            "YOUR JOB: produce an IMPROVED problem index + a USER PREFERENCES section.\n\n"
            "OUTPUT FORMAT:\n"
            "# [Course / Subject] - [N] Problems\n\n"
            "## Problem 1 - [short name]\n"
            "- Type: [category]\n"
            "- Given: [compact data summary]\n"
            "- Find: [what to solve]\n\n"
            "## Problem 2 - [short name]\n"
            "...\n\n"
            "## User Preferences\n"
            "- [Summarize what the user wants: full solutions? specific problems? format?]\n"
            "- [Include any constraints they mentioned: one attempt, careful work, etc.]\n\n"
            "RULES:\n"
            "- Keep the same compact format. Do NOT copy verbatim from the original document.\n"
            "- Fix any errors in the draft index based on what the user said.\n"
            "- EVERY problem must appear. Do NOT skip any.\n"
            "- Do NOT solve any problem. Do NOT include raw Q&A.\n"
        )
    else:
        spec_instructions = (
            "You are a strict visual-to-structure translator.\n\n"
            "Task:\n"
            "- If the content is UI: output ONLY a STRICT Markdown spec starting with '## UI_SPEC'.\n"
            "- If the content is a document: output a structured requirements spec.\n\n"
            "Constraints:\n"
            "- Use the user's answers to resolve ambiguity.\n"
            "- Do NOT include QUESTIONS or ANSWERS in the output.\n"
        )

    extra = (
        f"User answers:\n-----\n{answers_md}\n-----\n\n"
        f"Current draft spec (may be wrong/incomplete):\n-----\n{draft_spec}\n-----\n"
    )

    t0 = time.time()
    raw = _generate_with_context(sp, spec_instructions, max_new_tokens=900, extra_context=extra)
    t1 = time.time()

    if is_math_refine:
        # Only accept the refine output if it looks like a sane math index.
        candidate = raw.strip()
        if _math_index_is_sane(candidate) and not _spec_should_be_rejected(candidate):
            refined_spec = candidate
        else:
            # Fall back to the previous draft_spec instead of overwriting with junk
            refined_spec = draft_spec or _make_math_fallback_index(candidate)
    else:
        candidate = _extract_ui_spec_markdown(raw) or raw.strip()
        # If this was a code task and DeepSeek returned code, fall back to deterministic spec.
        h_domain = (meta.get("heuristic_domain") or "").lower()
        if ("programming" in h_domain or "python" in h_domain or "algorithms" in h_domain) and not _code_spec_is_sane(candidate):
            refined_spec = _make_code_fallback_spec(meta.get("user_prompt", ""))
        elif ("ui" in h_domain or "design" in h_domain or "product" in h_domain) and not _ui_spec_is_sane(candidate):
            # Don't overwrite a good UI draft with generic filler
            refined_spec = draft_spec or candidate
        else:
            refined_spec = candidate

    (sp / "spec.md").write_text(refined_spec + "\n", encoding="utf-8")
    _update_session_meta(
        sp,
        {
                "metrics": {
                    "deepseek_refine_ms": int((t1 - t0) * 1000),
                    "deepseek_refine_tokens_est": len(refined_spec.split()),
                }
        },
    )
    return {"spec_md": refined_spec}


@app.post("/sessions/ship", dependencies=[Depends(require_auth_and_csrf)])
def ship_to_cloud(body: ShipRequest) -> Dict[str, Any]:
    if not GROQ_API_KEY:
        raise HTTPException(status_code=400, detail="GROQ_API_KEY not configured")

    sp = _session_path(body.session_id)
    spec_path = sp / "spec.md"
    if not spec_path.exists():
        raise HTTPException(status_code=400, detail="Run refine first (spec.md missing)")
    spec_md = spec_path.read_text(encoding="utf-8").strip()

    rate_limiter.check(f"ship:{body.session_id}", limit=5, window_sec=60)

    allowed_models = {"moonshotai/kimi-k2-instruct-0905"}
    if body.model_name not in allowed_models:
        raise HTTPException(status_code=400, detail="Unsupported model")

    groq_client = Groq(api_key=GROQ_API_KEY)

    # Read heuristic to determine output format
    meta = _read_json(sp / "meta.json") if (sp / "meta.json").exists() else {}
    h_domain = (meta.get("heuristic_domain") or "").lower()
    instr_lower = body.instructions.lower()

    # ── Fine-grained domain detection ──────────────────────────────
    all_text = h_domain + " " + instr_lower

    pure_math_signals = ["math", "algebra", "calculus", "probability", "statistics",
                         "equation", "latex", "step-by-step", "solve", "proof",
                         "discrete", "linear algebra", "combinatorics"]
    science_signals   = ["physics", "chemistry", "biology", "newton", "force",
                         "velocity", "acceleration", "mole", "reaction", "cell",
                         "gene", "thermodynamic", "circuit", "enthalpy", "kepler",
                         "friction", "momentum", "energy", "torque", "gravity",
                         "kinematic", "orbital", "equilibrium"]
    code_signals      = ["code", "python", "rust", "javascript", "typescript", "java",
                         "react", "tailwind", "fastapi", "component", "function",
                         "implementation", "c++", "golang", "algorithm"]
    writing_signals   = ["essay", "writing", "english", "creative", "narrative",
                         "rhetorical", "literary", "fiction", "poetry", "draft",
                         "persuasive", "argumentative", "expository", "analysis",
                         "compare and contrast", "thesis"]
    history_signals   = ["history", "social science", "civilization", "revolution",
                         "political", "economics", "sociology", "psychology",
                         "primary source", "historiography"]

    is_pure_math = any(s in all_text for s in pure_math_signals)
    is_science   = any(s in all_text for s in science_signals)
    is_code      = any(s in all_text for s in code_signals)
    is_writing   = any(s in all_text for s in writing_signals)
    is_history   = any(s in all_text for s in history_signals)

    is_stem = is_pure_math or is_science

    # Gather the original PDF/document text so Kimi has full context
    ctx = _build_context(sp)
    original_content = ""
    if ctx.get("pdf_context"):
        original_content = ctx["pdf_context"]
    if ctx.get("user_prompt"):
        original_content = (ctx["user_prompt"] + "\n\n" + original_content).strip()

    # Also read user answers to include as preferences for Kimi
    answers_path = sp / "answers.md"
    user_answers = answers_path.read_text(encoding="utf-8").strip() if answers_path.exists() else ""

    # ── Per-domain: system_prompt, user_content, output_type, params ─

    if is_stem and not is_code:
        # ========== MATH / SCIENCE (grounded, precise) ==========
        output_type = "latex"
        science_extra = ""
        if is_science:
            science_extra = (
                "\nSCIENCE-SPECIFIC rules:\n"
                "- Always state which physical law or principle you are applying before using it.\n"
                "- Include units in EVERY intermediate step, not just the final answer.\n"
                "- Draw or describe free-body diagrams, circuit diagrams, or molecular structures where applicable.\n"
                "- For physics: clearly define your coordinate system and sign conventions before solving.\n"
                "- For chemistry: balance all equations, show oxidation states when relevant.\n"
                "- For biology: use proper nomenclature (binomial names, IUPAC, etc.).\n"
                "- Cross-check: does the magnitude and sign of your answer make physical/chemical sense?\n"
                "  (e.g. a friction coefficient should be 0 < mu < ~1.5; energies should have correct signs)\n"
            )
        system_prompt = (
            "You are an expert mathematician and scientist.\n"
            "You will receive THREE sections:\n"
            "  1. ORIGINAL DOCUMENT TEXT — the raw PDF extraction with exact problem wording.\n"
            "  2. STRUCTURED SPEC — a clean problem index summarizing each problem's type, data, and goal.\n"
            "  3. USER PREFERENCES — how the user wants the output formatted.\n\n"
            "You MUST solve EVERY SINGLE problem listed in the spec. Do not skip any.\n"
            "Use the ORIGINAL TEXT for exact data (matrices, vectors, numbers, units, etc.).\n"
            "Use the SPEC for guidance on what each problem is asking.\n"
            "Use the USER PREFERENCES for output format requirements.\n\n"
            "Your output MUST be a complete, compilable LaTeX document.\n"
            "Rules:\n"
            "- Start with \\documentclass{article} and proper preamble (amsmath, amssymb, physics, siunitx, etc.).\n"
            "- Use \\section{} for each problem, \\subsection{} for sub-parts.\n"
            "- Show all work step by step. Use align or align* environments for equations.\n"
            "- Box the final answer for each problem using \\boxed{}.\n"
            "- End with \\end{document}.\n"
            "- Output ONLY the LaTeX code. No markdown, no commentary outside the document.\n"
            f"{science_extra}\n"
            "CRITICAL - Verification checklist (apply to EVERY problem before finalizing):\n"
            "\n"
            "1. ALGEBRA — Variable isolation:\n"
            "   - When isolating a variable from combined equations, track EVERY subscript.\n"
            "   - After cross-multiplication, verify each term's subscript matches the original equation it came from.\n"
            "   - Common LLM error: swapping subscripts (e.g. using v_1 where v_2 belongs). Re-read the derivation line by line.\n"
            "   - Sanity-check: does your isolated formula reduce to the known special case when friction = 0 or angle = 0?\n"
            "\n"
            "2. GEOMETRY — Force diagrams and spatial reasoning:\n"
            "   - Before computing forces, DRAW the geometry mentally: where is each object relative to the target point?\n"
            "   - Identify the DIRECTION of each force vector (x-component, y-component) from the actual positions.\n"
            "   - If two objects are symmetric about the target point along one axis, their force components along that axis CANCEL to zero.\n"
            "   - Never assume a force has a vertical (or horizontal) component without checking the actual angle from the geometry.\n"
            "   - Common LLM error: hallucinating force components that are actually zero due to symmetry. Verify every trig ratio against the actual triangle.\n"
            "\n"
            "3. ARITHMETIC — Numerical checks:\n"
            "   - Double-check all substitutions, matrix-vector multiplications, row reductions, and scalar calculations.\n"
            "   - Verify that your final answers satisfy the original equations when you substitute them back in.\n"
            "   - Check units and orders of magnitude — does the answer make physical sense?\n"
            "   - LLMs often make simple arithmetic errors; take care to avoid them.\n"
        )
        prefs_block = user_answers if user_answers else "Solve all problems with full step-by-step work."
        user_content = (
            f"=== ORIGINAL DOCUMENT TEXT ===\n{original_content}\n\n"
            f"=== STRUCTURED SPEC (DeepSeek analysis) ===\n{spec_md}\n\n"
            f"=== USER PREFERENCES ===\n{prefs_block}\n\n"
            f"Additional instructions: {body.instructions!r}\n\n"
            "Solve every problem. Output a complete LaTeX document."
        )
        max_tokens = 8192
        temp = 0.1
        top_p_val = 0.7
        freq_penalty = 0.0

    elif is_code:
        # ========== CODE (precise but flexible) ==========
        output_type = "code"
        system_prompt = (
            "You are an expert software engineer who writes production-quality code.\n"
            "The user will provide a spec. Write clean, complete, working code.\n"
            "Rules:\n"
            "- Follow the user's language/framework instructions exactly.\n"
            "- Include all imports, type definitions, and exports.\n"
            "- Handle edge cases: null/undefined inputs, empty arrays, boundary values.\n"
            "- Add proper error handling (try/catch, Result types, etc.) where appropriate.\n"
            "- Use strong typing where the language supports it.\n"
            "- Add brief inline comments ONLY for non-obvious logic or tricky algorithms.\n"
            "- If building a UI component: make it accessible (ARIA labels, keyboard nav, semantic HTML).\n"
            "- Output ONLY the code. No markdown wrappers, no explanatory text.\n"
        )
        extra_ctx = f"\n\nOriginal context:\n{original_content}" if original_content else ""
        user_content = (
            f"Here is the spec:\n\n{spec_md}{extra_ctx}\n\n"
            f"Output instructions: {body.instructions!r}\n\n"
            "Produce ONLY the code."
        )
        max_tokens = 6144
        temp = 0.25
        top_p_val = 0.85
        freq_penalty = 0.0

    elif is_writing:
        # ========== WRITING / ENGLISH (creative, expressive) ==========
        output_type = "text"
        system_prompt = (
            "You are a skilled writer, essayist, and literary analyst.\n"
            "You write with clarity, style, and strong rhetorical structure.\n\n"
            "Rules:\n"
            "- Vary your sentence length and structure for natural rhythm.\n"
            "- Use vivid, concrete language over vague abstractions.\n"
            "- For essays: strong thesis statement, well-developed body paragraphs with evidence, compelling conclusion.\n"
            "- For creative writing: show don't tell, use sensory details, develop voice.\n"
            "- For literary analysis: cite specific passages, analyze technique (not just plot), connect to theme.\n"
            "- Maintain the appropriate tone for the genre (formal for academic, engaging for creative).\n"
            "- Avoid clichés and filler phrases ('In today's society...', 'Since the dawn of time...').\n"
            "- If a citation format is specified (MLA, APA, Chicago), follow it precisely.\n"
            "- Output clean text with clear paragraph breaks. No markdown code fences.\n"
        )
        extra_ctx = f"\n\nOriginal context:\n{original_content}" if original_content else ""
        user_content = (
            f"Here is the assignment/topic:\n\n{spec_md}{extra_ctx}\n\n"
            f"User preferences: {user_answers}\n" if user_answers else
            f"Here is the assignment/topic:\n\n{spec_md}{extra_ctx}\n\n"
            f"Output instructions: {body.instructions!r}\n"
        )
        max_tokens = 6144
        temp = 0.7
        top_p_val = 0.92
        freq_penalty = 0.3

    elif is_history:
        # ========== HISTORY / SOCIAL SCIENCE (balanced) ==========
        output_type = "text"
        system_prompt = (
            "You are a knowledgeable historian and social scientist.\n"
            "You write with analytical rigor and clear argumentation.\n\n"
            "Rules:\n"
            "- Ground every claim in evidence: dates, names, specific events.\n"
            "- Distinguish between primary and secondary sources when relevant.\n"
            "- Present multiple perspectives where historians disagree.\n"
            "- Use precise terminology (e.g. 'mercantilism', 'détente', not vague synonyms).\n"
            "- For essay-style answers: clear thesis, chronological or thematic structure, strong conclusion.\n"
            "- For short answers: be concise but specific — no vague generalizations.\n"
            "- Acknowledge causality carefully (correlation vs. causation, proximate vs. root causes).\n"
            "- Output clean text with clear section breaks. No markdown code fences.\n"
        )
        extra_ctx = f"\n\nOriginal context:\n{original_content}" if original_content else ""
        user_content = (
            f"Here is the content:\n\n{spec_md}{extra_ctx}\n\n"
            f"User preferences: {user_answers}\n" if user_answers else
            f"Here is the content:\n\n{spec_md}{extra_ctx}\n\n"
            f"Output instructions: {body.instructions!r}\n"
        )
        max_tokens = 6144
        temp = 0.4
        top_p_val = 0.88
        freq_penalty = 0.15

    else:
        # ========== GENERAL / FALLBACK (balanced) ==========
        output_type = "text"
        system_prompt = (
            "You are a clear, thorough writer and problem solver.\n"
            "The user will provide a topic or question set.\n"
            "Write a well-structured, detailed response.\n"
            "Rules:\n"
            "- Use clear headings, numbered sections, and bullet points.\n"
            "- Be thorough: address every question or sub-part.\n"
            "- Keep language precise and readable.\n"
            "- Support claims with reasoning or evidence.\n"
            "- Output clean text. No markdown code fences.\n"
        )
        extra_ctx = f"\n\nOriginal context:\n{original_content}" if original_content else ""
        user_content = (
            f"Here is the content:\n\n{spec_md}{extra_ctx}\n\n"
            f"Output instructions: {body.instructions!r}\n"
        )
        max_tokens = 6144
        temp = 0.4
        top_p_val = 0.88
        freq_penalty = 0.1

    t0 = time.time()
    completion = groq_client.chat.completions.create(
        model=body.model_name,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        temperature=temp,
        top_p=top_p_val,
        max_completion_tokens=max_tokens,
        frequency_penalty=freq_penalty,
        stream=False,
        stop=None,
    )

    t1 = time.time()
    output_text = (completion.choices[0].message.content or "").strip()

    # Strip wrapping code fences if present
    if output_text.startswith("```"):
        lines = output_text.split("\n")
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        output_text = "\n".join(lines).strip()

    # Save with appropriate extension
    ext_map = {"latex": ".tex", "code": ".txt", "text": ".txt"}
    out_filename = f"output{ext_map.get(output_type, '.txt')}"
    (sp / out_filename).write_text(output_text + "\n", encoding="utf-8")
    # Also keep GeneratedUI.jsx for backward compat
    (sp / "GeneratedUI.jsx").write_text(output_text + "\n", encoding="utf-8")

    _update_session_meta(
        sp,
        {
            "output_type": output_type,
            "metrics": {
                "kimi_ms": int((t1 - t0) * 1000),
                "kimi_tokens_est": len(output_text.split()),
            },
        },
    )
    return {"react_code": output_text, "output_type": output_type}


@app.get("/sessions/{session_id}", dependencies=[Depends(require_auth)])
def get_session(session_id: str) -> Dict[str, Any]:
    sp = _session_path(session_id)
    meta_path = sp / "meta.json"
    if not meta_path.exists():
        raise HTTPException(status_code=404, detail="Session not found")
    meta = _read_json(meta_path)

    def read_if_exists(name: str) -> str:
        p = sp / name
        return p.read_text(encoding="utf-8") if p.exists() else ""

    return {
        "meta": meta,
        "files": meta.get("files", []),
        "draft_spec": read_if_exists("draft_spec.md"),
        "questions_md": read_if_exists("questions.md"),
        "answers_md": read_if_exists("answers.md"),
        "spec_md": read_if_exists("spec.md"),
        "react_code": read_if_exists("GeneratedUI.jsx"),
        "heuristic": {
            "domain": meta.get("heuristic_domain", ""),
            "confidence": meta.get("heuristic_confidence", "low"),
        },
    }


@app.get("/sessions/{session_id}/export", dependencies=[Depends(require_auth)])
def export_session(session_id: str) -> StreamingResponse:
    sp = _session_path(session_id)
    if not sp.exists():
        raise HTTPException(status_code=404, detail="Session not found")

    mem = BytesIO()
    with zipfile.ZipFile(mem, "w", compression=zipfile.ZIP_DEFLATED) as z:
        # Core JSON/meta
        for name in ("meta.json",):
            p = sp / name
            if p.exists():
                z.write(p, arcname=f"{session_id}/{name}")
        # Text artifacts
        for name in ("draft_spec.md", "questions.md", "answers.md", "spec.md", "GeneratedUI.jsx"):
            p = sp / name
            if p.exists():
                z.write(p, arcname=f"{session_id}/{name}")
        # Uploaded files
        files_dir = sp / "files"
        if files_dir.exists():
            for root, _, files in os.walk(files_dir):
                for f in files:
                    fp = Path(root) / f
                    rel = fp.relative_to(sp)
                    z.write(fp, arcname=f"{session_id}/{rel}")

    mem.seek(0)
    filename = f"omni-session-{session_id}.zip"
    return StreamingResponse(
        mem,
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )
