# LongMem-RAT: Long-Term Memory for Cross-Attack Red Teaming

a multi-agent
red-teaming framework equipped with long term memory that accumulates, consolidates,
and reuses insights from past attacks across
sessions.

## Project Structure

```
src/mmm_attack/
├── __init__.py
├── __main__.py          # Entry point for module execution
├── main.py              # Main execution script
├── agents/              # Agent classes and state definitions
│   ├── __init__.py
│   ├── state.py         # AgentState definition
│   └── output_schemas.py # Output schema definitions
├── config/              # Configuration and prompts
│   ├── __init__.py
│   ├── prompts.py       # All prompt templates
│   └── settings.py      # System settings and parameters
├── llms/                # LLM implementations
│   ├── __init__.py
│   ├── claude_llm.py    # Anthropic Claude interface
│   ├── gemini_llm.py    # Google Gemini interface
│   ├── openai_llm.py    # OpenAI interface
│   ├── local_llm.py     # Local model implementations
│   ├── judge_llm.py     # Judge model for safety evaluation
│   └── runnable_llm.py  # LangChain runnable wrapper
├── utils/               # Utility functions
│   ├── __init__.py
│   ├── refusal_detection.py # Refusal detection utilities
│   └── memory_utils.py  # Memory management utilities
└── workflow/            # Workflow graph and nodes
    ├── __init__.py
    ├── graph.py         # Graph construction
    └── nodes.py         # Node implementations
```

## Installation

1. Clone the repository
2. Install with uv:
```bash
uv sync
```

3. Copy the environment file and configure your API keys:
```bash
cp env.example .env
# Edit .env with your API keys
```

## Usage

Run the attack system:

```bash
# Using uv
uv run python -m mmm_attack

# Or if installed
mmm-attack
```

## Configuration

- **Models**: Configure target, attack, and judge models in `src/mmm_attack/config/settings.py`
- **Prompts**: All prompt templates are in `src/mmm_attack/config/prompts.py`
- **API Keys**: Set in `.env` file (see `env.example`)

## Features

- **Multi-LLM Support**: Claude, Gemini, OpenAI, and local models
- **Memory System**: Persistent strategy memory across sessions
- **Configurable Workflow**: Graph-based attack pipeline
- **Batch Processing**: Process multiple behaviors with automatic saving
- **Safety Evaluation**: Integrated judge model for attack success detection

## Requirements

- Python 3.11+
- CUDA-capable GPU (for local models)
- API keys for external services (optional)

All content and logic has been preserved exactly as it was in the original notebook.
