"""
Quick standalone test for the heuristic domain scorer.

Usage:
  python backend/test_heuristic.py path/to/file.pdf
  python backend/test_heuristic.py path/to/file.pdf path/to/another.pdf
  python backend/test_heuristic.py --text "solve eigenvalues of this matrix"

No backend or GPU needed.
"""

import sys
import os

# Allow running from repo root
sys.path.insert(0, os.path.dirname(__file__))

import argparse
import re
from pathlib import Path
from typing import Dict, List

# ---- copy of _DOMAIN_RULES and scoring from app.py (no FastAPI imports needed) ----

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
]

THRESHOLD = 4


def score_domain(text_lower: str, markers: list) -> int:
    score = 0
    for m in markers:
        token, weight = m if isinstance(m, tuple) else (m, 1)
        if token in text_lower:
            score += weight
    return score


def classify(text: str) -> dict:
    text_lower = text.lower()
    scores: Dict[str, int] = {}
    for domain_label, markers in _DOMAIN_RULES:
        scores[domain_label] = score_domain(text_lower, markers)

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    best_domain, best_score = ranked[0]
    second_score = ranked[1][1] if len(ranked) > 1 else 0

    if best_score < THRESHOLD:
        return {"domain": "unknown", "confidence": "low", "scores": scores}

    margin = best_score - second_score
    if margin >= 8 or (best_score >= 12 and margin >= 4):
        confidence = "high"
    elif margin >= 4:
        confidence = "medium"
    else:
        confidence = "low"

    return {"domain": best_domain, "confidence": confidence, "scores": scores}


def extract_pdf_text(path: Path, max_pages: int = 50) -> str:
    try:
        import fitz
        doc = fitz.open(str(path))
        pages = []
        for i, page in enumerate(doc):
            if i >= max_pages:
                break
            pages.append(page.get_text())
        doc.close()
        return "\n".join(pages).strip()
    except ImportError:
        print("PyMuPDF (fitz) not installed. Falling back to empty text for PDF.")
        return ""


def print_result(label: str, result: dict):
    print(f"\n{'='*60}")
    print(f"  Input : {label}")
    print(f"  Domain: {result['domain']}")
    print(f"  Conf  : {result['confidence']}")
    print(f"\n  Scores (top 5):")
    sorted_scores = sorted(result["scores"].items(), key=lambda x: x[1], reverse=True)
    for domain, score in sorted_scores[:5]:
        bar = "#" * min(score, 40)
        print(f"    {domain:<35} {score:>3}  {bar}")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(description="Test the heuristic domain scorer.")
    parser.add_argument("inputs", nargs="*", help="PDF file paths or plain text snippets")
    parser.add_argument("--text", "-t", help="Inline text to classify")
    args = parser.parse_args()

    if args.text:
        result = classify(args.text)
        print_result(f'"{args.text[:60]}..."', result)

    for inp in args.inputs:
        p = Path(inp)
        if p.suffix.lower() == ".pdf":
            text = extract_pdf_text(p)
            if not text:
                print(f"WARNING: No text extracted from {p.name}")
            result = classify(text)
            print_result(p.name, result)
        else:
            # Treat as raw text file or inline text
            if p.exists():
                text = p.read_text(encoding="utf-8", errors="replace")
            else:
                text = inp
            result = classify(text)
            print_result(str(p)[:60], result)

    if not args.inputs and not args.text:
        print("Running built-in smoke tests...\n")

        smoke_tests = [
            ("linear algebra homework",
             "Find a basis for the null space of the matrix. Determine whether W1 is a subspace of R^3. Eigenvalues of the given matrix."),
            ("calculus problem set",
             "Compute the derivative using the chain rule. Evaluate the integral from 0 to pi of sin(x) dx. Find the limit as x approaches 0."),
            ("probability exam",
             "A random variable X follows a binomial distribution. Find the expected value and variance. Compute P(A|B) using Bayes theorem."),
            ("discrete math",
             "Prove by induction that the sum formula holds. How many permutations are there? Construct a bipartite graph."),
            ("python code review",
             "def binary_search(arr, target): for loop over the array. import numpy as np. Time complexity O(log n)."),
            ("UI wireframe",
             "Design a navbar with a dropdown. The mockup shows a button and modal. Use tailwind CSS flexbox grid dark mode."),
            ("ambiguous short text",
             "Solve this problem and show your work."),
        ]

        for label, text in smoke_tests:
            result = classify(text)
            status = "OK" if label.split()[0].lower() in result["domain"].lower() or result["domain"] == "unknown" else "??"
            print(f"  [{status}] {label:<35} -> {result['domain']} ({result['confidence']})")


if __name__ == "__main__":
    main()
