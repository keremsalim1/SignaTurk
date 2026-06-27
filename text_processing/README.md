# text_processing

Turns a stream of recognized sign-gloss words into fluent, grammatical
Turkish (and optional speech). A deterministic **rule-based engine** is the
reliable primary; an optional **ML layer** (HF Inference API or local
transformers) proposes a candidate that is validated and arbitrated against
the rule-based output per request.

## Architecture

Dependencies point inward — pure domain at the centre, I/O at the edges.

```
buffer ─▶ pipeline ─▶ grammar ─▶ tts
                        │
        web (FastAPI) ──┘   (text_processing_routes.py = thin shim)
```

### `grammar/` (was a single ~2000-line module)

| module          | responsibility                                            |
| --------------- | --------------------------------------------------------- |
| `linguistics`   | Turkish phonology & morphology — pure domain, no I/O      |
| `registry`      | `ModelSpec` / `MODEL_REGISTRY` / `GrammarConfig`          |
| `prompts`       | versioned zero-shot prompt templates                      |
| `validation`    | sanity checks on ML candidate sentences                   |
| `rules`         | `RuleBasedCorrector` — the primary engine                 |
| `adapters`      | model backends + `_classify_ml_error`                     |
| `arbiter`       | per-request ML-vs-rule quality comparator                 |
| `telemetry`     | in-process decision log (health endpoint)                 |
| `corrector`     | `GrammarCorrector` — the public hybrid facade             |

`grammar/__init__.py` re-exports the full public API, so
`from text_processing.grammar import …` is unchanged.

### `web/` (was the root-level `text_processing_routes.py`)

`schemas` (Pydantic) · `cache` (bounded LRU+TTL pipeline cache) ·
`jobs` (async job store + worker pool) · `router` (the `/api/text/*`
endpoints). The root `text_processing_routes.py` is now a thin
backward-compatible shim (`from text_processing_routes import router`).

## ML observability

ML failures are no longer swallowed into a bare fallback. Each failure is
classified (`quota` / `auth` / `not_served` / `rate_limit` / `timeout` /
`network` / `deps` …) and surfaced as an actionable Turkish message on
`GrammarResult.ml_error` → `CorrectResponse.ml_error`. `POST /api/text/ping`
probes a model's connectivity before a real correction.

## Conventions

- **Code and docstrings are written in English.**
- **User-facing strings are Turkish** — UI copy, model `notes`, and the
  `ml_error` / `ping` messages a user reads.
- The rule-based engine is the safe floor and the fallback for every ML
  error path; it is never removed.

## Dev workflow

No heavy ML/CV deps needed — tests stub them.

```bash
pip install fastapi pydantic            # light runtime deps
python -m unittest discover -s tests    # 242 tests, no network/weights
python -m text_processing.eval --check --min-exact 0.98   # rule-based floor

pip install ruff mypy
ruff check text_processing tests text_processing_routes.py
ruff format --check text_processing tests text_processing_routes.py
mypy text_processing

python -m text_processing.devserver     # standalone test UI at http://127.0.0.1:8000
```

Lint/format/type config lives in `pyproject.toml`; CI runs the same checks
(`.github/workflows/text-processing-ci.yml`).
