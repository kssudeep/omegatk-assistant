"""
test_prompts.py
---------------
Automated test suite for the OmegaTK RAG assistant covering all official
OmegaTK prompt categories observed in the original notebook plus edge cases.

Run with:
    python test_prompts.py

Requires GOOGLE_API_KEY in environment or a .env file.
Results are written to test_results.md.
"""

import os
import time
import textwrap
from dataclasses import dataclass, field
from typing import Literal
from dotenv import load_dotenv, find_dotenv

# Load .env from project directory (works in VS Code and local Jupyter)
load_dotenv(find_dotenv(usecwd=True))

os.environ.setdefault("USER_AGENT", "OmegaTK-RAG-Assistant/1.0")

# ---------------------------------------------------------------------------
# Test case definitions
# ---------------------------------------------------------------------------

@dataclass
class TestCase:
    id: str
    category: str
    prompt: str
    # What the response MUST contain (case-insensitive substring checks)
    must_contain: list[str] = field(default_factory=list)
    # What the response must NOT contain
    must_not_contain: list[str] = field(default_factory=list)
    # "code" | "clarify" | "decline" | "any"
    expected_type: Literal["code", "clarify", "decline", "any"] = "any"
    # For multi-turn tests: list of prior (human, ai) string pairs
    history: list[tuple[str, str]] = field(default_factory=list)


TEST_CASES: list[TestCase] = [

    # ------------------------------------------------------------------
    # 1. BASIC CONFORMER GENERATION (from official examples)
    # ------------------------------------------------------------------
    TestCase(
        id="T01",
        category="Basic conformer generation",
        prompt="How do I generate conformers for a molecule from a SMILES string using OmegaTK?",
        must_contain=["OEOmega", "OEMol", "oechem", "oeomega"],
        expected_type="code",
    ),
    TestCase(
        id="T02",
        category="Basic conformer generation",
        prompt=(
            "Generate conformers for a molecule with SMILES input, "
            "save output to SDF file, maximum 100 conformers"
        ),
        must_contain=["OEOmega", "SetMaxConfs", "oemolostream", "OEWriteMolecule"],
        expected_type="code",
    ),

    # ------------------------------------------------------------------
    # 2. OMEGA OPTIONS / CONFIGURATION
    # ------------------------------------------------------------------
    TestCase(
        id="T03",
        category="OmegaOptions configuration",
        prompt=(
            "How do I set the maximum number of conformers to 50 "
            "and the energy window to 10 in OmegaTK?"
        ),
        must_contain=["OEOmegaOptions", "SetMaxConfs"],
        expected_type="code",
    ),
    TestCase(
        id="T04",
        category="OmegaOptions configuration",
        prompt="How do I set the RMSD cutoff for conformer filtering in OmegaTK?",
        must_contain=["OEOmegaOptions"],
        expected_type="code",
    ),
    TestCase(
        id="T05",
        category="OmegaOptions configuration",
        prompt="How do I use densely sampled conformers in OmegaTK?",
        must_contain=["OEOmega", "oeomega"],
        expected_type="code",
    ),

    # ------------------------------------------------------------------
    # 3. BATCH / MULTI-MOLECULE PROCESSING
    # ------------------------------------------------------------------
    TestCase(
        id="T06",
        category="Batch processing",
        prompt=(
            "How do I generate conformers for multiple molecules "
            "from an SDF file using OmegaTK?"
        ),
        must_contain=["OEOmega", "oemolistream"],
        expected_type="code",
    ),
    TestCase(
        id="T07",
        category="Batch processing",
        prompt=(
            "Read molecules from a mol2 file, generate conformers with OmegaTK, "
            "and write results to an SDF file."
        ),
        must_contain=["OEOmega", "oemolistream", "oemolostream"],
        expected_type="code",
    ),

    # ------------------------------------------------------------------
    # 4. MACROCYCLE CONFORMERS
    # ------------------------------------------------------------------
    TestCase(
        id="T08",
        category="Macrocycle conformers",
        prompt="How do I generate conformers for a macrocycle molecule in OmegaTK?",
        must_contain=["oeomega"],
        expected_type="code",
    ),

    # ------------------------------------------------------------------
    # 5. STEREO / FLIPPER INTEGRATION
    # ------------------------------------------------------------------
    TestCase(
        id="T09",
        category="Stereochemistry",
        prompt=(
            "How do I enumerate stereoisomers and generate conformers "
            "for each one using OmegaTK?"
        ),
        must_contain=["OEOmega", "oeomega"],
        expected_type="code",
    ),

    # ------------------------------------------------------------------
    # 6. MULTI-TURN: clarification then code
    # ------------------------------------------------------------------
    TestCase(
        id="T10",
        category="Multi-turn: clarification -> code",
        prompt="SMILES input, save to SDF, max 50 conformers",
        must_contain=["OEOmega", "SetMaxConfs"],
        expected_type="code",
        history=[
            (
                "Generate conformers",
                (
                    "To generate conformers, I need a bit more information. "
                    "What is the input molecule? How should conformers be stored?"
                ),
            )
        ],
    ),

    # ------------------------------------------------------------------
    # 7. MULTI-TURN: follow-up modification
    # ------------------------------------------------------------------
    TestCase(
        id="T11",
        category="Multi-turn: follow-up modification",
        prompt="Now add error handling to that code",
        must_not_contain=["which code", "what code", "please clarify"],
        expected_type="code",
        history=[
            (
                "Generate conformers for a SMILES string and save to SDF",
                (
                    "```python\n"
                    "from openeye import oechem, oeomega\n"
                    "mol = oechem.OEMol()\n"
                    "oechem.OESmilesToMol(mol, smiles)\n"
                    "omega = oeomega.OEOmega()\n"
                    "omega.Build(mol)\n"
                    "```"
                ),
            )
        ],
    ),

    # ------------------------------------------------------------------
    # 8. VAGUE PROMPTS → must ask for clarification
    # ------------------------------------------------------------------
    TestCase(
        id="T12",
        category="Clarification required",
        prompt="Generate conformers",
        must_contain=["?"],
        must_not_contain=["```python"],
        expected_type="clarify",
    ),
    TestCase(
        id="T13",
        category="Clarification required",
        prompt="Write some OmegaTK code",
        must_contain=["?"],
        must_not_contain=["```python"],
        expected_type="clarify",
    ),

    # ------------------------------------------------------------------
    # 9. OUT-OF-SCOPE → must decline
    # ------------------------------------------------------------------
    TestCase(
        id="T14",
        category="Out of scope",
        prompt="How do I sort a list in Python?",
        must_not_contain=["```python", "def sort"],
        expected_type="decline",
    ),
    TestCase(
        id="T15",
        category="Out of scope",
        prompt="What is the capital of France?",
        must_not_contain=["```python"],
        expected_type="decline",
    ),
]


# ---------------------------------------------------------------------------
# RAG chain (reuses rag/ modules)
# ---------------------------------------------------------------------------

def build_rag_chain():
    """Build the RAG chain using the shared rag/ modules."""
    from rag.indexer import get_or_build_index
    from rag.chain import build_llm, build_rag_chain as _build

    print("Setting up RAG chain for tests...")
    vectorstore = get_or_build_index()
    llm = build_llm()
    return _build(vectorstore, llm)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@dataclass
class TestResult:
    tc: TestCase
    response: str
    passed: bool
    failures: list[str]
    elapsed: float


def evaluate(tc: TestCase, response: str, elapsed: float) -> TestResult:
    failures = []
    r_lower = response.lower()

    for term in tc.must_contain:
        if term.lower() not in r_lower:
            failures.append(f"must_contain '{term}' not found")

    for term in tc.must_not_contain:
        if term.lower() in r_lower:
            failures.append(f"must_not_contain '{term}' was found")

    if tc.expected_type == "code" and "```python" not in response:
        failures.append("expected_type=code but no ```python block found")
    elif tc.expected_type == "clarify" and "?" not in response:
        failures.append("expected_type=clarify but no question mark found")
    elif tc.expected_type == "decline" and "```python" in response:
        failures.append("expected_type=decline but code block was generated")

    return TestResult(
        tc=tc,
        response=response,
        passed=len(failures) == 0,
        failures=failures,
        elapsed=elapsed,
    )


# ---------------------------------------------------------------------------
# Test runner
# ---------------------------------------------------------------------------

def run_tests() -> list[TestResult]:
    from langchain_core.messages import HumanMessage, AIMessage

    rag_chain = build_rag_chain()
    results: list[TestResult] = []

    for tc in TEST_CASES:
        print(f"\n{'='*60}")
        print(f"[{tc.id}] {tc.category}")
        print(f"Prompt: {tc.prompt[:80]}...")

        history = [
            msg
            for human, ai in tc.history
            for msg in (HumanMessage(content=human), AIMessage(content=ai))
        ]

        t0 = time.time()
        try:
            result = rag_chain.invoke({
                "input": tc.prompt,
                "chat_history": history,
            })
            response = result.get("answer", "")
        except Exception as e:
            response = f"ERROR: {e}"
        elapsed = time.time() - t0

        result = evaluate(tc, response, elapsed)
        results.append(result)

        status = "PASS" if result.passed else "FAIL"
        print(f"Status : {status}  ({elapsed:.1f}s)")
        if result.failures:
            for f in result.failures:
                print(f"  ✗ {f}")
        else:
            print("  ✓ All checks passed")

    return results


# ---------------------------------------------------------------------------
# Report writer
# ---------------------------------------------------------------------------

def write_report(results: list[TestResult], path: str = "test_results.md"):
    passed = sum(1 for r in results if r.passed)
    total = len(results)

    lines = [
        "# OmegaTK RAG Assistant — Test Results\n",
        f"**{passed}/{total} tests passed**\n",
        "",
        "| ID | Category | Status | Time | Failures |",
        "|----|----------|--------|------|----------|",
    ]
    for r in results:
        status = "✅ PASS" if r.passed else "❌ FAIL"
        failures = "; ".join(r.failures) if r.failures else "—"
        lines.append(
            f"| {r.tc.id} | {r.tc.category} | {status} | {r.elapsed:.1f}s | {failures} |"
        )

    lines += ["", "---", ""]
    for r in results:
        lines += [
            f"## {r.tc.id} — {r.tc.category}",
            f"**Prompt:** {r.tc.prompt}",
            "",
            f"**Status:** {'PASS' if r.passed else 'FAIL'}  |  **Time:** {r.elapsed:.1f}s",
            "",
        ]
        if r.failures:
            lines.append("**Failures:**")
            for f in r.failures:
                lines.append(f"- {f}")
            lines.append("")
        lines += [
            "**Response:**",
            "```",
            textwrap.shorten(r.response, width=800, placeholder="... [truncated]"),
            "```",
            "",
        ]

    with open(path, "w") as f:
        f.write("\n".join(lines))
    print(f"\nReport written to {path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    results = run_tests()
    write_report(results)
    passed = sum(1 for r in results if r.passed)
    print(f"\n{'='*60}")
    print(f"FINAL: {passed}/{len(results)} tests passed")
