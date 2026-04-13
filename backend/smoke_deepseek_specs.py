"""
End-to-end smoke test for DeepSeek spec generation via the FastAPI backend.

Runs three scenarios (math, UI, code) without file uploads to validate:
  - /auth/login works
  - /sessions/start returns non-empty draft_spec + questions_md
  - /sessions/refine returns a sane spec for each scenario

Usage:
  python backend/smoke_deepseek_specs.py
"""

from __future__ import annotations

import os
import re
import sys
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import requests


API_BASE = os.environ.get("OMNI_API_BASE", "http://127.0.0.1:8000").rstrip("/")


def _load_env_password() -> str:
    # Prefer process env; fall back to repo .env
    p = os.environ.get("OMNI_PASSWORD", "").strip()
    if p:
        return p
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    env_path = os.path.join(root, ".env")
    try:
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                if k.strip() == "OMNI_PASSWORD":
                    return v.strip().strip("\"'() ")
    except Exception:
        pass
    return ""


PASSWORD = _load_env_password()


@dataclass
class Case:
    name: str
    user_prompt: str
    answers: str


def _must(cond: bool, msg: str) -> None:
    if not cond:
        raise RuntimeError(msg)


def _is_low_entropy_numeric_garbage(text: str) -> bool:
    t = (text or "").strip()
    if not t:
        return True
    tokens = re.findall(r"\S+", t)
    if len(tokens) < 30:
        return False
    unique = set(tokens)
    if len(unique) < 8:
        # looks like "1 1 2 2 2 ..."
        return True
    return False


def _check_math_spec(text: str) -> Tuple[bool, str]:
    if not text.strip():
        return False, "empty"
    if _is_low_entropy_numeric_garbage(text):
        return False, "low-entropy numeric garbage"
    if re.search(r"^##\s*Problem\b", text, flags=re.MULTILINE) is None:
        return False, "missing '## Problem' headings"
    if "Given:" not in text or "Find:" not in text:
        return False, "missing Given/Find fields"
    return True, "ok"


def _check_ui_spec(text: str) -> Tuple[bool, str]:
    if not text.strip():
        return False, "empty"
    if _is_low_entropy_numeric_garbage(text):
        return False, "low-entropy numeric garbage"
    if not text.lstrip().startswith("## UI_SPEC"):
        return False, "missing ## UI_SPEC header"
    return True, "ok"


def _check_code_spec(text: str) -> Tuple[bool, str]:
    if not text.strip():
        return False, "empty"
    if _is_low_entropy_numeric_garbage(text):
        return False, "low-entropy numeric garbage"
    # Very loose: ensure it looks like a structured spec, not just repeated junk.
    has_bullets = bool(re.search(r"(?m)^\s*[-*]\s+", text))
    has_sections = bool(re.search(r"(?m)^#{1,3}\s+", text))
    if not (has_bullets or has_sections):
        return False, "missing obvious structure (no bullets/headers)"
    return True, "ok"


def main() -> None:
    if not PASSWORD:
        print("ERROR: OMNI_PASSWORD env var not set.")
        sys.exit(2)

    s = requests.Session()

    # login (captures cookies: omni_auth, omni_csrf)
    r = s.post(f"{API_BASE}/auth/login", json={"password": PASSWORD}, timeout=30)
    _must(r.ok, f"login failed: {r.status_code} {r.text}")

    csrf = s.cookies.get("omni_csrf", "")
    _must(bool(csrf), "missing omni_csrf cookie after login")
    csrf_headers = {"X-CSRF-Token": csrf}

    cases = [
        Case(
            name="math",
            user_prompt=(
                "Solve this linear algebra homework. It has multiple problems.\n"
                "Problem 1: Determine whether each Wi is a basis for R^3 given sets of vectors.\n"
                "Problem 2: Determine whether a given set is a basis for a plane defined by 2x1+2x2+x3=0.\n"
                "Problem 3: Find a basis for a subspace of R^4 spanned by given vectors.\n"
                "Problem 4: Coordinate vector relative to a basis B in R^2.\n"
                "I want all problems solved carefully.\n"
            ),
            answers=(
                "## ANSWERS\n"
                "- Q1: Please solve ALL problems.\n"
                "- Q2: Provide full step-by-step work.\n"
                "- Q3: Output a compilable LaTeX document.\n"
            ),
        ),
        Case(
            name="ui",
            user_prompt=(
                "Design a mobile login screen for Omni-Architect.\n"
                "Dark theme, centered card, email + password fields, 'Login' button,\n"
                "error state, loading state, and a small 'Forgot password?' link.\n"
                "Use a modern glassy aesthetic.\n"
            ),
            answers=(
                "## ANSWERS\n"
                "- Q1: Target screen size iPhone 15 Pro.\n"
                "- Q2: Use dark mode only.\n"
                "- Q3: Prefer a single screen with a modal error.\n"
            ),
        ),
        Case(
            name="code",
            user_prompt=(
                "Build a Python FastAPI endpoint POST /sum that accepts JSON {a:int, b:int}\n"
                "and returns {sum:int}. Validate inputs, return 400 on bad input, and include tests.\n"
            ),
            answers=(
                "## ANSWERS\n"
                "- Q1: Use pytest for tests.\n"
                "- Q2: Keep it minimal and production-ish.\n"
            ),
        ),
    ]

    results: Dict[str, Dict[str, Any]] = {}

    for c in cases:
        # start session
        r = s.post(
            f"{API_BASE}/sessions/start",
            data={"user_prompt": c.user_prompt},
            files=[],
            headers=csrf_headers,
            timeout=600,
        )
        _must(r.ok, f"{c.name}: start failed: {r.status_code} {r.text}")
        start = r.json()
        sid = start["session_id"]
        draft = (start.get("draft_spec") or "").strip()
        questions = (start.get("questions_md") or "").strip()

        # refine
        rr = s.post(
            f"{API_BASE}/sessions/refine",
            json={"session_id": sid, "answers_md": c.answers},
            headers=csrf_headers,
            timeout=600,
        )
        _must(rr.ok, f"{c.name}: refine failed: {rr.status_code} {rr.text}")
        refined = (rr.json().get("spec_md") or "").strip()

        if c.name == "math":
            ok_d, why_d = _check_math_spec(draft)
            ok_r, why_r = _check_math_spec(refined)
        elif c.name == "ui":
            ok_d, why_d = _check_ui_spec(draft)
            ok_r, why_r = _check_ui_spec(refined)
        else:
            ok_d, why_d = _check_code_spec(draft)
            ok_r, why_r = _check_code_spec(refined)

        results[c.name] = {
            "session_id": sid,
            "draft_ok": ok_d,
            "draft_why": why_d,
            "refined_ok": ok_r,
            "refined_why": why_r,
            "draft_preview": draft[:400],
            "refined_preview": refined[:400],
            "questions_preview": questions[:300],
        }

    print("\nDeepSeek spec smoke test results\n" + "=" * 34)
    for name, info in results.items():
        print(f"\n[{name}] session={info['session_id']}")
        print(f"  draft:   {'OK' if info['draft_ok'] else 'FAIL'} ({info['draft_why']})")
        print(f"  refined: {'OK' if info['refined_ok'] else 'FAIL'} ({info['refined_why']})")
        if not info["draft_ok"]:
            print("  draft preview:\n" + info["draft_preview"])
        if not info["refined_ok"]:
            print("  refined preview:\n" + info["refined_preview"])
        print("  questions preview:\n" + info["questions_preview"])


if __name__ == "__main__":
    main()

