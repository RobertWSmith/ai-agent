"""A small LangGraph coding agent that generates validated Python scripts.

Usage:
    python coding_agent.py "Create a CLI that counts words in a text file" --out word_count.py

The generated script is validated with Python's compiler, but it is not
executed. That keeps the validation loop useful without running arbitrary code.

The module is intentionally compact so it can be used as either:
    * a command-line script for generating one file at a time, or
    * an importable helper via ``generate_validated_script``.
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
    """Shared state passed between LangGraph nodes.

    Attributes:
        prompt: The original natural-language request from the user.
        script: The latest Python source produced by the model.
        validation_error: The most recent validation error, if any.
        attempts: Number of model generation or repair calls already made.
        max_attempts: Hard cap for generation plus repair attempts.
        validated: Whether the latest script passed validation.
    """

    # The original request stays unchanged so repair prompts keep full context.
    prompt: str
    # The candidate Python source is overwritten on every generation or repair.
    script: str
    # Validation failures are fed back to the repair node.
    validation_error: str
    # Attempts counts model calls, not graph node transitions.
    attempts: int
    # The graph stops repairing once attempts reaches this limit.
    max_attempts: int
    # The routing function uses this flag to decide whether to finish.
    validated: bool


def _message_text(message: Any) -> str:
    """Extract plain text from common LangChain chat message shapes.

    Args:
        message: A LangChain message object, a raw string, or a list-style
            multimodal message payload.

    Returns:
        A best-effort plain-text representation that can be treated as source
        text by the rest of the agent.
    """

    content = getattr(message, "content", message)
    if isinstance(content, str):
        return content

    # Some providers return content as a list of typed blocks. Keep only the
    # text blocks because this agent only knows how to validate Python source.
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
    """Remove Markdown code fences from model output.

    Args:
        text: Raw text returned by the model.

    Returns:
        The first fenced Python block when one is present. If the model already
        returned plain source code, the stripped original text is returned.
    """

    text = text.strip()
    fenced_match = re.search(
        r"```(?:python|py)?\s*(.*?)```", text, re.IGNORECASE | re.DOTALL
    )
    if fenced_match:
        return fenced_match.group(1).strip()
    return text


def validate_python_script(script: str) -> tuple[bool, str]:
    """Validate generated Python source without executing it.

    Args:
        script: Candidate Python source code.

    Returns:
        A ``(valid, error)`` tuple. ``valid`` is ``True`` when the script can be
        compiled by Python. ``error`` is empty on success and contains a concise
        validation message on failure.
    """

    if not script.strip():
        return False, "The generated script is empty."
    if "```" in script:
        return False, "The script still contains Markdown code fences."

    try:
        # ``compile`` catches syntax errors without importing modules or running
        # arbitrary generated code.
        compile(script, "<generated_script>", "exec")
    except SyntaxError as exc:
        line = exc.lineno or "unknown"
        offset = exc.offset or "unknown"
        return False, f"SyntaxError on line {line}, offset {offset}: {exc.msg}"

    return True, ""


def build_coding_graph(llm: Any) -> Any:
    """Create and compile the LangGraph validation loop.

    Args:
        llm: A chat model object with an ``invoke(messages)`` method, such as
            ``langchain_openai.ChatOpenAI``.

    Returns:
        A compiled LangGraph app. Invoking it produces a final ``CodingState``.

    Raises:
        RuntimeError: If the LangGraph or LangChain message dependencies are
            not installed in the active Python environment.
    """

    try:
        from langchain_core.messages import HumanMessage, SystemMessage
        from langgraph.graph import END, StateGraph
    except ImportError as exc:
        raise RuntimeError(
            "Missing LangGraph dependencies. Install them with: "
            "pip install langgraph langchain-openai"
        ) from exc

    def generate(state: CodingState) -> CodingState:
        """Generate the first candidate script from the user's prompt.

        Args:
            state: Current graph state. The ``prompt`` field must be populated.

        Returns:
            Updated graph state with a candidate script and incremented attempt
            count.
        """

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
        """Validate the current candidate script.

        Args:
            state: Current graph state containing the latest candidate script.

        Returns:
            Updated graph state with ``validated`` and ``validation_error`` set
            from the compiler check.
        """

        valid, error = validate_python_script(state["script"])
        return {**state, "validated": valid, "validation_error": error}

    def repair(state: CodingState) -> CodingState:
        """Ask the model to repair a script that failed validation.

        Args:
            state: Current graph state including the invalid script and the
                compiler error that should guide the repair.

        Returns:
            Updated graph state with a replacement candidate script and
            incremented attempt count.
        """

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
        """Choose the next graph edge after validation.

        Args:
            state: Current graph state after the validation node has run.

        Returns:
            ``"end"`` when the script is valid or the attempt cap is reached;
            otherwise ``"repair"`` to continue the validation loop.
        """

        if state["validated"] or state["attempts"] >= state["max_attempts"]:
            return "end"
        return "repair"

    # The graph shape is deliberately small:
    # generate once, validate, then repair and re-validate until success or cap.
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
    """Generate a Python script, repairing until it compiles or attempts run out.

    Args:
        prompt: Natural-language request describing the desired Python script.
        model: Chat model name passed to the default OpenAI client.
        max_attempts: Maximum number of model calls across generation and
            repairs.
        llm_factory: Optional factory used by tests or callers that want to
            provide a custom LangChain-compatible chat model.

    Returns:
        Validated Python source code.

    Raises:
        ValueError: If ``max_attempts`` is less than one.
        RuntimeError: If dependencies are missing or validation never succeeds.
    """

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
            """Build the default deterministic OpenAI chat model.

            Args:
                model_name: Name of the chat model to instantiate.

            Returns:
                A ``ChatOpenAI`` instance configured for repeatable generation.
            """

            return ChatOpenAI(model=model_name, temperature=0)

    graph = build_coding_graph(llm_factory(model))
    # LangGraph has its own recursion protection. Keep it slightly above the
    # expected node transitions for the configured attempt count.
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
    """Parse command-line arguments for the coding agent.

    Args:
        argv: Raw argument list, excluding the executable name.

    Returns:
        Parsed CLI options.
    """

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
    """Run the command-line interface.

    Args:
        argv: Optional argument list for tests or programmatic callers. When
            omitted, arguments are read from ``sys.argv``.

    Returns:
        Process-style exit code: ``0`` on success, ``1`` for agent failures, and
        ``2`` when no prompt was provided.
    """

    args = parse_args(sys.argv[1:] if argv is None else argv)
    prompt = " ".join(args.prompt).strip()

    # Stdin support makes the CLI easy to use with prompt files or shell pipes.
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

    # If no output path is provided, emit the generated script to stdout so the
    # command can be composed with shell redirection or inspection tools.
    if args.out:
        args.out.write_text(script + "\n", encoding="utf-8")
        print(f"Wrote validated script to {args.out}")
    else:
        print(script)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
