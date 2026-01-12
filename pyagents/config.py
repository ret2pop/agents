# Models
ORCHESTRATOR_MODEL = "qwen2.5-coder:14b"
VISION_MODEL = "qwen2.5-vl:7b"
EMBEDDING_MODEL = "nomic-embed-text-v2-moe:latest"

CODING_MODEL_CONFIG = {
    "coder": "qwen2.5-coder:14b",
    "tester": "qwen2.5-coder:14b",
    "verifier": "qwen3-vl:8b"
}

DEEP_RESEARCH_MODEL_CONFIG = {
    "global_planner": "qwen3:14b",
    "planner": "qwen3:14b",
    "researcher": "qwen3:14b",
    "writer": "qwen3:14b",
    "editor": "ministral-3:14b",
    "skeptics": [
        "cogito:14b",
        "qwen3:14b",
    ]
}

MATH_MODEL_CONFIG = {
    "theorist": "qwen2.5-coder:14b",
    "formalizer": "qwen2.5-coder:14b",
    "arbiter": "qwen2.5-coder:14b"
}

SEARCH_MODEL = "qwen2.5-coder:14b"

# Limits
MAX_DEPTH = 3
MAX_RETRIES = 10
MAX_SEARCH_RESULTS = 10
MAX_READ_COUNT = 3

# Files
SCRIPT_NAME = "temp_sandbox_script.py"
TEST_NAME = "temp_generated_tests.py"
LEAN_FILE = "proof_attempt.lean"
