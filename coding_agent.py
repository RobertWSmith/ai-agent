"""A small LangGraph coding agent that generates validated Python scripts.

Usage:
    python coding_agent.py "Create a CLI that counts words in a text file" --out word_count.py

The generated script is validated with Python's compiler, but it is not
executed. That keeps the validation loop useful without running arbitrary code.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path
from typing import Any, Callable, TypedDict

from dotenv import load_dotenv

load_dotenv()

DEFAULT_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
DEFAULT_MAX_ATTEMPTS = 3


class CodingState(TypedDict):
    """State passed between LangGraph nodes."""

    prompt: str
    script: str
    validation_error: str
    attempts: int
    max_attempts: int
    validated: bool


def _message_text(message: Any) -> str:
    """Extract plain text from common LangChain chat message shapes."""

    content = getattr(message, "content", message)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and isinstance(item.get("text"), str):
                parts.append(item["text"])
        return "\n".join(parts)
    return str(content)


def _strip_markdown_fence(text: str) -> str:
    """Return the first fenced Python block, or the full response if unfenced."""

    text = text.strip()
    fenced_match = re.search(
        r"```(?:python|py)?\s*(.*?)```", text, re.IGNORECASE | re.DOTALL
    )
    if fenced_match:
        return fenced_match.group(1).strip()
    return text


def validate_python_script(script: str) -> tuple[bool, str]:
    """Validate generated Python source without executing it."""

    if not script.strip():
        return False, "The generated script is empty."
    if "```" in script:
        return False, "The script still contains Markdown code fences."

    try:
        compile(script, "<generated_script>", "exec")
    except SyntaxError as exc:
        line = exc.lineno or "unknown"
        offset = exc.offset or "unknown"
        return False, f"SyntaxError on line {line}, offset {offset}: {exc.msg}"

    return True, ""


def build_coding_graph(llm: Any) -> Any:
    """Create and compile the LangGraph validation loop."""

    try:
        from langchain_core.messages import HumanMessage, SystemMessage
        from langgraph.graph import END, StateGraph
    except ImportError as exc:
        raise RuntimeError(
            "Missing LangGraph dependencies. Install them with: "
            "pip install langgraph langchain-openai"
        ) from exc

    def generate(state: CodingState) -> CodingState:
        response = llm.invoke(
            [
                SystemMessage(
                    content=(
                        "You are a careful Python coding agent. "
                        "Return exactly one complete Python script. "
                        "Do not include Markdown fences, prose, or explanations. "
                        "Prefer standard-library code unless the user asks otherwise."
                    )
                ),
                HumanMessage(content=state["prompt"]),
            ]
        )
        return {
            **state,
            "script": _strip_markdown_fence(_message_text(response)),
            "validation_error": "",
            "attempts": state["attempts"] + 1,
            "validated": False,
        }

    def validate(state: CodingState) -> CodingState:
        valid, error = validate_python_script(state["script"])
        return {**state, "validated": valid, "validation_error": error}

    def repair(state: CodingState) -> CodingState:
        response = llm.invoke(
            [
                SystemMessage(
                    content=(
                        "You repair Python scripts. Return exactly one complete "
                        "Python script with no Markdown fences or explanation."
                    )
                ),
                HumanMessage(
                    content=(
                        "Original user request:\n"
                        f"{state['prompt']}\n\n"
                        "Validation error:\n"
                        f"{state['validation_error']}\n\n"
                        "Invalid script:\n"
                        f"{state['script']}\n\n"
                        "Return a corrected Python script."
                    )
                ),
            ]
        )
        return {
            **state,
            "script": _strip_markdown_fence(_message_text(response)),
            "validation_error": "",
            "attempts": state["attempts"] + 1,
            "validated": False,
        }

    def route_after_validation(state: CodingState) -> str:
        if state["validated"] or state["attempts"] >= state["max_attempts"]:
            return "end"
        return "repair"

    workflow = StateGraph(CodingState)
    workflow.add_node("generate", generate)
    workflow.add_node("validate", validate)
    workflow.add_node("repair", repair)

    workflow.set_entry_point("generate")
    workflow.add_edge("generate", "validate")
    workflow.add_conditional_edges(
        "validate", route_after_validation, {"repair": "repair", "end": END}
    )
    workflow.add_edge("repair", "validate")

    return workflow.compile()


def generate_validated_script(
    prompt: str,
    *,
    model: str = DEFAULT_MODEL,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    llm_factory: Callable[[str], Any] | None = None,
) -> str:
    """Generate a Python script, repairing until it compiles or attempts run out."""

    if max_attempts < 1:
        raise ValueError("max_attempts must be at least 1.")

    if llm_factory is None:
        try:
            from langchain_openai import ChatOpenAI
        except ImportError as exc:
            raise RuntimeError(
                "Missing OpenAI chat dependency. Install it with: pip install langchain-openai"
            ) from exc

        def llm_factory(model_name: str) -> Any:
            return ChatOpenAI(model=model_name, temperature=0)

    graph = build_coding_graph(llm_factory(model))
    recursion_limit = max(6, max_attempts * 3 + 2)
    result = graph.invoke(
        {
            "prompt": prompt,
            "script": "",
            "validation_error": "",
            "attempts": 0,
            "max_attempts": max_attempts,
            "validated": False,
        },
        config={"recursion_limit": recursion_limit},
    )

    if not result["validated"]:
        raise RuntimeError(
            "Could not produce a valid Python script within "
            f"{max_attempts} attempt(s). Last error: {result['validation_error']}"
        )

    return result["script"]


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate validated Python scripts with LangGraph."
    )
    parser.add_argument(
        "prompt", nargs="*", help="Prompt describing the Python script to generate."
    )
    parser.add_argument(
        "--out", type=Path, help="Optional file path to write the validated script."
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Chat model to use. Default: {DEFAULT_MODEL}",
    )
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=DEFAULT_MAX_ATTEMPTS,
        help=f"Maximum generation/repair attempts. Default: {DEFAULT_MAX_ATTEMPTS}",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    prompt = " ".join(args.prompt).strip()

    if not prompt and not sys.stdin.isatty():
        prompt = sys.stdin.read().strip()
    if not prompt:
        print("Provide a prompt as an argument or through stdin.", file=sys.stderr)
        return 2

    try:
        script = generate_validated_script(
            prompt,
            model=args.model,
            max_attempts=args.max_attempts,
        )
    except Exception as exc:
        print(f"Agent failed: {exc}", file=sys.stderr)
        return 1

    if args.out:
        args.out.write_text(script + "\n", encoding="utf-8")
        print(f"Wrote validated script to {args.out}")
    else:
        print(script)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
