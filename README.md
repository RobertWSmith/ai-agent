# LangGraph Coding Agent

A simple LangGraph-based coding agent that accepts a natural-language prompt and
generates a validated Python script.

The agent uses a small graph:

1. Generate a candidate Python script from the prompt.
2. Validate the script with Python's compiler.
3. If validation fails, ask the model to repair the script using the compiler
   error.
4. Repeat until the script validates or the maximum attempt limit is reached.

Validation uses `compile(...)`, so generated code is checked for Python syntax
without being executed.

## Requirements

- Python 3.10 through 3.13
- An OpenAI API key
- The pinned packages listed in `pyproject.toml`

## Setup

Create and activate a virtual environment, then install dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip setuptools wheel
.\.venv\Scripts\python.exe -m pip install -e .
```

For development tools such as `black`, install the `dev` extra:

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
```

Set your API key in the shell:

```powershell
$env:OPENAI_API_KEY = "your-api-key"
```

You can also create a `.env` file because `coding_agent.py` loads environment
variables with `python-dotenv`:

```text
OPENAI_API_KEY=your-api-key
OPENAI_MODEL=gpt-4o-mini
```

## Usage

Generate a script and print it to the terminal:

```powershell
.\.venv\Scripts\python.exe coding_agent.py "Create a CLI that counts words in a text file"
```

Generate a script and write it to a file:

```powershell
.\.venv\Scripts\python.exe coding_agent.py "Create a CSV sorter by date" --out generated_script.py
```

Limit the validation loop:

```powershell
.\.venv\Scripts\python.exe coding_agent.py "Create a backup utility" --max-attempts 2
```

Choose a model:

```powershell
.\.venv\Scripts\python.exe coding_agent.py "Create a JSON pretty-printer" --model gpt-4o-mini
```

You can also pipe prompts through stdin:

```powershell
Get-Content prompt.txt | .\.venv\Scripts\python.exe coding_agent.py --out generated_script.py
```

## Code Structure

- `CodingState`: Typed state shared between LangGraph nodes.
- `_message_text`: Converts LangChain message responses into plain text.
- `_strip_markdown_fence`: Removes Markdown code fences from model output.
- `validate_python_script`: Compiles generated code without executing it.
- `build_coding_graph`: Creates the generate, validate, and repair graph.
- `generate_validated_script`: Public helper that runs the graph and returns
  validated source code.
- `parse_args`: Defines the command-line interface.
- `main`: Runs the CLI and handles output.

## Validation And Limits

The validation loop has two safeguards:

- `--max-attempts` limits model calls across the first generation and all
  repairs.
- LangGraph's `recursion_limit` is set from the attempt limit so the graph cannot
  run indefinitely.

The current validator catches empty responses, leftover Markdown fences, and
Python syntax errors. It does not execute generated code, import generated
modules, install packages, or verify runtime behavior.

## Development Checks

Compile the agent:

```powershell
.\.venv\Scripts\python.exe -m py_compile coding_agent.py
```

Format the code:

```powershell
.\.venv\Scripts\python.exe -m black coding_agent.py
```
