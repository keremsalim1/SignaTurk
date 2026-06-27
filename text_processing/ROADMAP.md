# Text Processing Pipeline — Roadmap

Follow-up plan for the items intentionally deferred from the architecture-review
fix-up (the review flagged TTS/ML correctness, observability, and evaluation
gaps; those P0–P2 items plus the **live-flow integration** are now done). This
roadmap covers the remaining, larger work.

## Status

**Done**
- `synthesize_audio` honored per call (gTTS never fires when off); `tts_status` surfaced.
- ML defaults to the cloud Qwen API (no silent local download); API/local defaults split.
- `model_name_override` architecture inferred (causal/seq2seq/inference-api).
- Bounded pipeline cache (LRU + idle TTL + manual unload) + `DELETE /api/text/cache`.
- Structured decision logging + `GET /api/text/health` + optional live HF smoke test.
- Labelled eval set + practical metric harness (`python -m text_processing.eval`).
- **Live integration**: `/api/predict/live` feeds recognized words into the pipeline
  and emits assembled sentences (`type:"sentence"`, additive, rule-based, no latency).
- **R1 — Offline TTS** ✅: pluggable engine (gTTS + optional offline Piper, `auto`
  fallback); `/api/text/health` reports engine state.
- **R2 — Eval growth + CI gate** ✅: dataset grown to 112 gold examples; eval
  `--check` floor + GitHub Actions workflow (`.github/workflows/text-processing-ci.yml`).
  Growth surfaced and fixed a Turkish-locale capitalization bug (`iki` → `İki`,
  not `Iki`).
- **R3 — Async ML** ✅: `POST /api/text/correct/async` + `GET /api/text/jobs/{id}`
  run slow ML off the request thread (bounded, TTL'd job store + thread pool).
- **R5 — Prompt versioning** ✅: `PROMPT_TEMPLATES` registry (zs-1, zs-2),
  selectable via `GrammarConfig.prompt_version` / `$SIGNAI_PROMPT_VERSION` and
  `eval --prompt-version`; version logged with every decision.
- **ML observability** ✅: failures are classified (`quota`/`auth`/`not_served`/
  `rate_limit`/`timeout`/`network`/`deps`) and surfaced as an actionable Turkish
  `ml_error` (no more opaque "ml unavailable"); `POST /api/text/ping` probes a
  model before use; test UI shows a green/amber/neutral status + "Bağlantıyı test et".
- **Clean architecture** ✅: the ~2000-line `grammar.py` god-module is now a
  layered `grammar/` package, and the root `text_processing_routes.py` moved into
  a `text_processing/web/` package (schemas/cache/jobs/router) behind a thin shim.
  Behavior unchanged (242 tests green); see `text_processing/README.md`.
- **Tooling** ✅: `pyproject.toml` adds ruff (lint+format) and mypy; CI runs
  `ruff check`, `ruff format --check`, and `mypy text_processing`.

**Deferred (this roadmap):** live UX polish · offline/local fast model · keep
growing the eval set (→300) · pyupgrade (modernize type hints) · typed
`Result`/error objects instead of `None`.

## Prioritization

| # | Item | Value | Effort | Risk | Priority |
|---|------|-------|--------|------|----------|
| R1 | Offline TTS fallback (Piper) | High | M | Low | ✅ done |
| R2 | Eval growth + ML eval + CI gate | High | M | Low | ✅ done |
| R3 | Async / streaming ML | High | L | Med | ✅ done |
| R5 | Prompt versioning & experiments | Med | M | Low | ✅ done |
| R4 | Live UX polish (sentence panel + TTS + flush) | Med | M | Low | ✅ done¹ |
| R6 | Offline/local fast model (ONNX int8 / fine-tune) | Med | L | Med | **next** |

R1–R5 are done. ¹R4 ships the live "Cümle" panel + **Tamamla** (flush) and
**Seslendir** (speak) buttons in `tsl_nexus_ui_no_upload.html`; JSX was validated
with Babel (no syntax errors) but still wants a quick **visual QA in-browser**.
Smaller R4 follow-ups: hand the sentence to the 3D avatar and make the silence
window configurable via `/api/settings`. Remaining: **R6** (offline local model —
needs ONNX export / fine-tune infra). R3 returns results via polling; token
streaming over SSE/WebSocket is a future extension.

### Cloud model choice (kept lean)

The ML layer pulls **one** strong instruction model from the HF serverless
Inference API — **`qwen2.5-7b-api` (Qwen2.5-7B-Instruct)**, the proven
best-in-domain serverless default (good Turkish, small, no local download, no
custom-endpoint plumbing). Heavyweight frontier models (MiniMax M2/M3, 2026) were
evaluated and **rejected**: coding/agentic-focused, not on HF serverless, and
overkill for short gloss→sentence — they would only burden the system.

To adopt a newer serverless model later (e.g. **Qwen3-8B-Instruct**, ~15% fewer
hallucinations than Qwen2.5) the lean way — verify before flipping the default:
1. Add a registry entry in `grammar.py` (`arch="inference-api"`, `conversational=True`).
2. With `HF_TOKEN` set, score it: `python -m text_processing.eval --use-ml --model <key>`
   plus `RUN_HF_SMOKE=1 … -k smoke` for latency / fallback rate.
3. Promote to `recommended`/default only if it measures better on the TİD eval.

---

## R1 — Offline TTS fallback (Piper)

**Why:** gTTS needs network and returns `None` offline (now visible via
`tts_status:"failed"`, but there is still no speech offline).

**Approach:** make the TTS engine pluggable behind the existing
`TTSSynthesizer`. Add an engine enum (`gtts` | `piper` | `auto`) to `TTSConfig`;
`auto` tries gTTS, falls back to Piper (local Turkish voice, e.g.
`tr_TR-fahrettin-medium`) when gTTS fails or `SIGNAI_TTS_OFFLINE=1`.

- Files: `text_processing/tts.py` (add `_GTTSBackend`, `_PiperBackend`, selection),
  `requirements.txt` (optional `piper-tts`), `gtts_installed()` → generalize to
  `tts_engine_status()` for the health endpoint.
- Interface: keep `synthesize_to_bytes/synthesize_to_file` stable; backends are an
  internal detail. Lazy-import Piper so it stays optional.
- **Acceptance:** with network blocked and a Piper voice present, `synthesize_to_file`
  returns a real mp3/wav; `/api/text/health.tts` reports the active engine; unit tests
  stub both backends (no model download in CI).

## R2 — Eval set growth + ML eval + CI gate

**Why:** the seed set is 47 gold examples; the review wanted 100–300 and real
evidence behind model claims.

**Approach:**
- Grow `text_processing/eval/dataset.jsonl` to ~200–300, sampled from the 179-sign
  `landmarks/` vocabulary as SOV/markered combinations; have a native TİD signer
  review the gold sentences. Add adversarial cases (hallucination bait, dropped
  negation/question) to lock in the anti-hallucination guard.
- Add an ML-mode eval run (`--use-ml`) that records fallback rate + latency, and a
  small `--baseline`/`--check` mode that fails (non-zero exit) if rule-based
  `exact_match` drops below a threshold (e.g. 0.98) — wire into CI.
- Files: `text_processing/eval/dataset.jsonl`, `text_processing/eval/__init__.py`
  (add `--check`), a CI workflow step running `python -m text_processing.eval --check`.
- **Acceptance:** `--check` gates regressions in CI; an ML eval JSON report is
  produced when a token is configured; per-tag coverage report stays green.

## R3 — Async / streaming ML

**Why:** HF Inference API latency (and cold starts) block `/api/text/correct`;
the live path stays snappy only because it is rule-based.

**Approach:** add a non-blocking job API for the ML path:
- `POST /api/text/correct/async` → `{job_id}`; worker runs the pipeline off the
  request thread (asyncio task / thread pool / lightweight queue).
- `GET /api/text/jobs/{job_id}` → `pending|done` + result, **or** stream partial
  output over SSE/WebSocket. Reuse `DECISION_LOG` for status.
- Keep the existing sync endpoint for rule-based / short input.
- Files: `text_processing_routes.py` (+ a small `jobs.py` store, bounded like the
  pipeline cache), optional `_InferenceAPIAdapter` token streaming.
- **Risk:** concurrency/lifecycle; mitigate with a bounded job store + TTL and tests
  that stub the adapter. **Acceptance:** a slow stubbed model does not block other
  requests; job lifecycle covered by tests.

## R4 — Live UX polish

**Why:** the backend now emits `type:"sentence"`; the UI only shows it in the
status line.

**Approach:** dedicated "current sentence" panel in
`frontend/tsl_nexus_ui_no_upload.html`; a **Flush** button that sends
`{"action":"flush"}` (already supported server-side); optional per-sentence TTS
playback (call `/api/text/correct` with `synthesize_audio:true` on the assembled
words) and hand-off to the avatar. Make the silence window configurable via
`/api/settings`. **Acceptance:** sentences render in their own panel; flush button
finalizes on demand; playback works behind a toggle.

## R5 — Prompt versioning & experiments

**Why:** `PROMPT_VERSION` is logged but there is one hard-coded template.

**Approach:** a small prompt registry (`PROMPT_TEMPLATES: {version: builder}`) in
`grammar.py`; select via `GrammarConfig.prompt_version` / env; the eval harness
loops over versions to compare metrics (A/B). Decisions already log the version.
**Acceptance:** switching versions changes prompts without code edits; eval reports
per-version metrics.

## R6 — Offline / local fast model

**Why:** the only ML option is cloud (token + network) or a heavy local
transformer; no light offline path.

**Approach (pick one, gated by R2 numbers):**
- Export `mt0-small` to **ONNX int8** and add an `onnx` adapter arch (`onnxruntime`),
  or
- **Fine-tune** a small seq2seq on the grown eval-style corpus for TİD→Turkish, ship
  quantized.
- Files: new `_OnnxAdapter` in `grammar.py`, registry entries, `requirements.txt`
  (optional `onnxruntime`). **Acceptance:** offline inference under the local timeout
  with eval parity at/above rule-based on the dataset; opt-in, never the default.

---

### Known issues (found via eval growth)

- **Loanword final-stop softening**: dative of `market` is rendered `markede`
  instead of `markete` — the rule engine softens the final `t` without a
  loanword exception (native `kitap`→`kitabı` is correct). Needs a small
  loanword lexicon / exception list in the case-suffix logic before adding such
  words to the eval set.

### Cross-cutting

- **Observability:** export `/api/text/health` counters to the admin dashboard /
  metrics endpoint; alert when `fallback_rate` or `hf_token_present=false` in prod.
- **Security guard:** keep the "add no meaning absent from the input" rule; expand
  adversarial eval coverage (R2) before exposing any user-profile/context to the LLM.

## Deferred from CodeRabbit (PR #2 follow-up)

Triaged during PR #2; the in-scope, eval-gated fixes (pronoun case forms, single-
token pronoun agreement, plus ~13 concurrency/security/correctness items) shipped.
These remain, grouped by why they were deferred:

**Security (needs a real ML-stack test run)**
- Bump CVE'd pins in `requirements.txt`: `torch>=2.6.0` (CVE-2025-32434),
  `sentencepiece>=0.2.1` (CVE-2026-1260), `transformers>=4.48.0`. Verify the
  local seq2seq/causal adapters still load after the bump.

**Live / backend path (outside the text_processing package; needs the heavy stack)**
- `backend.py`: wrap `_build_live_text_pipeline()` in try/except so a live-pipeline
  init error degrades to the word-only flow instead of taking down `/api/predict/live`.
- `buffer.py`: base silence completion on `last_seen_at`, not `last_added_at`
  (debounced repeats keep the session alive).
- `pipeline.py`: `tick()` double-processes when `WordBuffer` already fired
  `on_complete` — add a `_complete_handled` guard so grammar/TTS don't run twice.
- `web/jobs.py`: bound async work with a semaphore so a flood can't leave orphaned
  background jobs after max-size eviction (the TTL-from-completion fix already landed).

**Privacy (policy decision)**
- Decision log emits raw user text (`input_words`/`final_sentence`/`rejected_candidate`)
  at INFO. Decide whether to redact or move to DEBUG; the in-memory health log keeps
  full data regardless.

**Linguistics (needs eval-set growth to validate)**
- **Pronoun objects don't always get a case assigned**: `["sen","o","görmek"]` →
  "Sen o görüyorsun." (should be "…onu…"). The case *forms* are now correct
  (`_PRONOUN_CASE_FORMS`), but the SOV role parser doesn't mark some pronoun objects
  for case — fix in the role-assignment path, gated by new eval examples.
- Verb-only ASCII validator gate over-rejects valid ML output (e.g. `gitmek →
  "Gidiyorum."`); loosen with a stem-overlap check once the eval set covers it.
