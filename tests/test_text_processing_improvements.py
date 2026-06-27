"""Tests for the review-driven improvements:

* synthesize_audio is honored per-call (gTTS never fires when off)
* model_name_override architecture is inferred, not assumed seq2seq
* ML defaults to the cloud API (no silent local download)
* the pipeline cache is bounded (LRU + TTL + unload)
* structured decision logging + health endpoint
* (optional, env-gated) live HF/Qwen smoke test
"""

from __future__ import annotations

import os
import unittest
from pathlib import Path
from unittest.mock import patch

import text_processing_routes as routes
from text_processing import (
    DECISION_LOG,
    DEFAULT_API_MODEL_KEY,
    DEFAULT_LOCAL_MODEL_KEY,
    DEFAULT_MODEL_KEY,
    PROMPT_VERSION,
    GrammarConfig,
    GrammarCorrector,
    PipelineConfig,
    SignTextPipeline,
)
from text_processing.tts import TTSConfig, TTSSynthesizer, _Audio, tts_engine_status


class SynthesizeAudioGatingTests(unittest.TestCase):
    """The headline bug: synthesize_audio=False used to still run gTTS."""

    def _pipeline(self) -> SignTextPipeline:
        return SignTextPipeline(
            PipelineConfig(grammar=GrammarConfig(use_ml=False), synthesize_audio=True)
        )

    def test_per_call_false_skips_tts_entirely(self) -> None:
        pipeline = self._pipeline()
        with patch.object(pipeline.tts, "synthesize_to_file") as m:
            result = pipeline.correct(["merhaba", "ben", "su", "istemek"], synthesize_audio=False)
        m.assert_not_called()
        self.assertIsNone(result.audio_path)
        self.assertEqual(result.tts_status, "disabled")

    def test_per_call_true_runs_tts(self) -> None:
        pipeline = self._pipeline()
        with patch.object(pipeline.tts, "synthesize_to_file", return_value=Path("/tmp/x.mp3")) as m:
            result = pipeline.correct(["merhaba", "ben", "su", "istemek"], synthesize_audio=True)
        m.assert_called_once()
        self.assertEqual(result.tts_status, "ok")
        self.assertEqual(result.audio_path, Path("/tmp/x.mp3"))

    def test_tts_failure_surfaces_status(self) -> None:
        pipeline = self._pipeline()
        with patch.object(pipeline.tts, "synthesize_to_file", return_value=None):
            result = pipeline.correct(["merhaba", "ben", "su", "istemek"], synthesize_audio=True)
        self.assertIsNone(result.audio_path)
        self.assertEqual(result.tts_status, "failed")

    def test_route_false_returns_no_audio(self) -> None:
        req = routes.CorrectRequest(
            words=["ben", "okul", "gitmek"], use_ml=False, synthesize_audio=False
        )
        resp = routes.correct(req)
        self.assertIsNone(resp.audio_url)
        self.assertEqual(resp.tts_status, "disabled")
        self.assertEqual(resp.sentence, "Ben okula gidiyorum.")
        self.assertEqual(resp.source, "rule-based")


class ModelArchInferenceTests(unittest.TestCase):
    def test_causal_name_inferred(self) -> None:
        spec = GrammarConfig(model_name_override="ytu-ce-cosmos/turkish-gpt2").resolve_spec()
        self.assertEqual(spec.arch, "causal")

    def test_seq2seq_name_inferred(self) -> None:
        spec = GrammarConfig(model_name_override="bigscience/mt0-small").resolve_spec()
        self.assertEqual(spec.arch, "seq2seq")

    def test_explicit_arch_override_wins(self) -> None:
        spec = GrammarConfig(
            model_name_override="my/custom-model", model_arch_override="inference-api"
        ).resolve_spec()
        self.assertEqual(spec.arch, "inference-api")
        self.assertTrue(spec.conversational)

    def test_invalid_arch_override_raises(self) -> None:
        with self.assertRaises(ValueError):
            GrammarConfig(model_name_override="x", model_arch_override="bogus").resolve_spec()

    def test_unknown_model_key_still_raises(self) -> None:
        with self.assertRaises(ValueError):
            GrammarConfig(model_key="does-not-exist").resolve_spec()


class DefaultModelSelectionTests(unittest.TestCase):
    def test_ml_default_is_cloud_api(self) -> None:
        self.assertEqual(DEFAULT_MODEL_KEY, DEFAULT_API_MODEL_KEY)
        self.assertEqual(DEFAULT_API_MODEL_KEY, "qwen2.5-7b-api")
        self.assertEqual(DEFAULT_LOCAL_MODEL_KEY, "mt0-small")

    def test_resolve_model_key(self) -> None:
        self.assertEqual(routes._resolve_model_key(True, None), DEFAULT_API_MODEL_KEY)
        self.assertEqual(routes._resolve_model_key(False, None), DEFAULT_MODEL_KEY)
        self.assertEqual(routes._resolve_model_key(True, "mt0-small"), "mt0-small")


class CustomModelRoutingTests(unittest.TestCase):
    """Test-UI feature: pick any HF model by raw id (e.g. Qwen3) + arch."""

    def test_qwen3_entry_present_but_not_recommended(self) -> None:
        from text_processing import MODEL_REGISTRY

        self.assertIn("qwen3-8b-api", MODEL_REGISTRY)
        spec = MODEL_REGISTRY["qwen3-8b-api"]
        self.assertEqual(spec.arch, "inference-api")
        self.assertFalse(spec.recommended)  # qwen2.5 stays the default

    def test_model_name_override_falls_back_gracefully(self) -> None:
        # A raw HF model id is accepted; if it can't load (deps absent or a bad
        # id) the pipeline degrades to rule-based. Stub the adapter build so the
        # outcome is deterministic regardless of whether ML deps are installed.
        routes._PIPELINE_CACHE.clear()
        req = routes.CorrectRequest(
            words=["ben", "okul", "gitmek"],
            model_name="some/seq2seq-model",
            model_arch="seq2seq",
            synthesize_audio=False,
        )
        with patch(
            "text_processing.grammar.corrector._build_adapter",
            side_effect=RuntimeError("ml unavailable in test"),
        ):
            resp = routes.correct(req)
        self.assertEqual(resp.sentence, "Ben okula gidiyorum.")
        self.assertEqual(resp.source, "rule-based")


class PipelineCacheTests(unittest.TestCase):
    def test_lru_eviction_respects_max_size(self) -> None:
        cache = routes.PipelineCache(max_size=2, ttl_seconds=0)
        cache.get_or_create((True, "a"), lambda: "A")
        cache.get_or_create((True, "b"), lambda: "B")
        # Touch "a" so it becomes most-recently-used; factory must NOT run.
        touched = cache.get_or_create((True, "a"), lambda: "SHOULD_NOT_BUILD")
        self.assertEqual(touched, "A")
        cache.get_or_create((True, "c"), lambda: "C")  # evicts LRU = "b"
        self.assertIn((True, "a"), cache._entries)
        self.assertIn((True, "c"), cache._entries)
        self.assertNotIn((True, "b"), cache._entries)

    def test_ttl_expiry(self) -> None:
        cache = routes.PipelineCache(max_size=5, ttl_seconds=100)
        cache.get_or_create((True, "a"), lambda: "A")
        cache._entries[(True, "a")].last_used_at -= 1000  # make it stale
        cache.get_or_create((True, "b"), lambda: "B")  # triggers expiry sweep
        self.assertNotIn((True, "a"), cache._entries)
        self.assertIn((True, "b"), cache._entries)

    def test_unload_and_clear(self) -> None:
        cache = routes.PipelineCache(max_size=5, ttl_seconds=0)
        cache.get_or_create((True, "a"), lambda: "A")
        cache.get_or_create((False, "a"), lambda: "A2")
        self.assertTrue(cache.unload((True, "a")))
        self.assertFalse(cache.unload((True, "a")))
        self.assertEqual(cache.clear(), 1)


class HealthAndLoggingTests(unittest.TestCase):
    def test_health_payload_shape(self) -> None:
        health = routes.text_health()
        self.assertEqual(health["status"], "ok")
        self.assertIsInstance(health["hf_token_present"], bool)
        self.assertEqual(health["default_api_model"], "qwen2.5-7b-api")
        self.assertEqual(health["default_local_model"], "mt0-small")
        self.assertIn("entries", health["cache"])
        self.assertIn("recent_decisions", health)
        self.assertIn("prompts", health)
        self.assertIn("tts", health)
        self.assertIn("jobs", health)

    def test_decision_log_records_structured_entry(self) -> None:
        DECISION_LOG.clear()
        GrammarCorrector(GrammarConfig(use_ml=False)).correct_detailed(["ben", "okul", "gitmek"])
        rec = DECISION_LOG.recent()[-1]
        self.assertEqual(rec.prompt_version, PROMPT_VERSION)
        self.assertEqual(rec.source, "rule-based")
        self.assertEqual(rec.final_sentence, "Ben okula gidiyorum.")
        self.assertGreaterEqual(DECISION_LOG.summary()["window"], 1)

    def test_unload_endpoint_404_when_absent(self) -> None:
        routes._PIPELINE_CACHE.clear()
        from fastapi import HTTPException

        with self.assertRaises(HTTPException) as ctx:
            routes.unload_model("nonexistent-model")
        self.assertEqual(ctx.exception.status_code, 404)


class RuleEngineFixTests(unittest.TestCase):
    """Turkish-locale capitalization at sentence start (i→İ, ı→I)."""

    def test_dotted_i_capitalization(self) -> None:
        from text_processing.grammar import RuleBasedCorrector

        self.assertEqual(RuleBasedCorrector().correct(["iki"]), "İki.")
        self.assertEqual(RuleBasedCorrector().correct(["isim"]), "İsim.")

    def test_dotless_i_capitalization(self) -> None:
        from text_processing.grammar import RuleBasedCorrector

        self.assertEqual(RuleBasedCorrector().correct(["ışık"]), "Işık.")


class PromptVersionTests(unittest.TestCase):
    """Versioned prompt registry: select per request, trace per decision."""

    def test_registry_has_multiple_versions(self) -> None:
        from text_processing import PROMPT_TEMPLATES, PROMPT_VERSION, list_prompt_versions

        self.assertIn(PROMPT_VERSION, PROMPT_TEMPLATES)
        self.assertGreaterEqual(len(PROMPT_TEMPLATES), 2)
        self.assertIn("zs-2", list_prompt_versions())

    def test_default_prompt_unchanged(self) -> None:
        from text_processing.grammar import PROMPT_VERSION, build_zero_shot_prompt

        default = build_zero_shot_prompt(["ben", "okul", "gitmek"])
        explicit = build_zero_shot_prompt(["ben", "okul", "gitmek"], PROMPT_VERSION)
        self.assertEqual(default, explicit)
        self.assertIn("Kelimeler:", default)

    def test_versions_differ(self) -> None:
        from text_processing.grammar import build_zero_shot_messages, build_zero_shot_prompt

        self.assertNotEqual(
            build_zero_shot_prompt(["ben", "gelmek"], "zs-1"),
            build_zero_shot_prompt(["ben", "gelmek"], "zs-2"),
        )
        self.assertNotEqual(
            build_zero_shot_messages(["ben", "gelmek"], "zs-1")[0]["content"],
            build_zero_shot_messages(["ben", "gelmek"], "zs-2")[0]["content"],
        )

    def test_unknown_version_raises(self) -> None:
        from text_processing.grammar import build_zero_shot_prompt

        with self.assertRaises(ValueError):
            build_zero_shot_prompt(["x"], "nope")

    def test_ml_corrector_rejects_unknown_prompt_version(self) -> None:
        corrector = GrammarCorrector(GrammarConfig(use_ml=True, prompt_version="nope"))
        self.assertIsNone(corrector._ml)  # disabled at construction, no network

    def test_decision_log_records_config_version(self) -> None:
        DECISION_LOG.clear()
        GrammarCorrector(GrammarConfig(use_ml=False, prompt_version="zs-2")).correct_detailed(
            ["ben", "okul", "gitmek"]
        )
        self.assertEqual(DECISION_LOG.recent()[-1].prompt_version, "zs-2")


class AsyncJobTests(unittest.TestCase):
    """Non-blocking ML job API: submit → poll, bounded + TTL'd store."""

    def _inline_executor(self):
        # Run the submitted job synchronously so the test is deterministic.
        return patch.object(routes._EXECUTOR, "submit", lambda fn, *a, **k: fn(*a, **k))

    def test_job_lifecycle_done(self) -> None:
        req = routes.CorrectRequest(
            words=["ben", "okul", "gitmek"], use_ml=False, synthesize_audio=False
        )
        with self._inline_executor():
            submit = routes.correct_async(req)
        self.assertEqual(submit.status, "pending")
        job = routes.get_job(submit.job_id)
        self.assertEqual(job.status, "done")
        self.assertIsNotNone(job.result)
        self.assertEqual(job.result.sentence, "Ben okula gidiyorum.")

    def test_job_error_is_captured(self) -> None:
        req = routes.CorrectRequest(words=["   "], use_ml=False)  # empty after strip
        with self._inline_executor():
            submit = routes.correct_async(req)
        job = routes.get_job(submit.job_id)
        self.assertEqual(job.status, "error")
        self.assertIn("words must contain", job.error or "")

    def test_unknown_job_404(self) -> None:
        from fastapi import HTTPException

        with self.assertRaises(HTTPException) as ctx:
            routes.get_job("no-such-job")
        self.assertEqual(ctx.exception.status_code, 404)

    def test_store_bounded_and_ttl(self) -> None:
        from text_processing.web.schemas import CorrectResponse

        resp = CorrectResponse(words=["x"], sentence="X.", source="rule-based", ml_latency_ms=0.0)

        # Bounded: an oldest FINISHED job is evicted to make room; pending work
        # in flight is never dropped.
        store = routes.JobStore(max_size=2, ttl_seconds=0)
        a = store.create()
        store.set_done(a, resp)  # finished → evictable
        b = store.create()
        c = store.create()  # evicts finished a; pending b is kept
        self.assertIsNone(store.get(a))
        self.assertIsNotNone(store.get(b))
        self.assertIsNotNone(store.get(c))

        # Saturated with all-pending jobs → backpressure (RuntimeError → 429),
        # not eviction of work whose result would be lost.
        sat = routes.JobStore(max_size=1, ttl_seconds=0)
        sat.create()
        with self.assertRaises(RuntimeError):
            sat.create()

        # TTL is measured from completion: a long-pending job is NOT evicted,
        # but a finished one past its TTL is.
        expiring = routes.JobStore(max_size=5, ttl_seconds=100)
        pending = expiring.create()
        expiring._jobs[pending].created_at -= 1000  # old, but still pending
        self.assertIsNotNone(expiring.get(pending), "pending jobs must not TTL-expire")

        done = expiring.create()
        expiring.set_done(done, resp)
        expiring._jobs[done].finished_at -= 1000  # finished long ago
        self.assertIsNone(expiring.get(done), "finished jobs past TTL are evicted")

    def test_async_saturation_returns_429(self) -> None:
        from fastapi import HTTPException

        saturated = routes.JobStore(max_size=1, ttl_seconds=0)
        saturated.create()  # one pending job fills the store
        with patch("text_processing.web.router._JOB_STORE", saturated):
            with self.assertRaises(HTTPException) as ctx:
                routes.correct_async(routes.CorrectRequest(words=["ben", "okul", "gitmek"]))
        self.assertEqual(ctx.exception.status_code, 429)


class TTSEngineTests(unittest.TestCase):
    """Pluggable TTS engine: gTTS + optional offline Piper, with auto fallback."""

    def test_auto_prefers_gtts(self) -> None:
        synth = TTSSynthesizer(TTSConfig(engine="auto"))
        with (
            patch.object(
                TTSSynthesizer, "_synth_gtts", return_value=_Audio(b"MP3", "mp3", "audio/mpeg")
            ),
            patch.object(TTSSynthesizer, "_synth_piper") as piper,
        ):
            audio = synth._synthesize("merhaba")
        self.assertEqual(audio.ext, "mp3")
        piper.assert_not_called()

    def test_auto_falls_back_to_piper_when_gtts_unavailable(self) -> None:
        synth = TTSSynthesizer(TTSConfig(engine="auto"))
        with (
            patch.object(TTSSynthesizer, "_synth_gtts", return_value=None),
            patch.object(
                TTSSynthesizer, "_synth_piper", return_value=_Audio(b"WAV", "wav", "audio/wav")
            ),
        ):
            audio = synth._synthesize("merhaba")
        self.assertEqual(audio.ext, "wav")

    def test_engine_piper_skips_gtts(self) -> None:
        synth = TTSSynthesizer(TTSConfig(engine="piper"))
        with (
            patch.object(TTSSynthesizer, "_synth_gtts") as gtts,
            patch.object(
                TTSSynthesizer, "_synth_piper", return_value=_Audio(b"WAV", "wav", "audio/wav")
            ),
        ):
            synth._synthesize("x")
        gtts.assert_not_called()

    def test_all_engines_fail_returns_none(self) -> None:
        synth = TTSSynthesizer(TTSConfig(engine="auto"))
        with (
            patch.object(TTSSynthesizer, "_synth_gtts", return_value=None),
            patch.object(TTSSynthesizer, "_synth_piper", return_value=None),
        ):
            self.assertIsNone(synth._synthesize("merhaba"))
            self.assertIsNone(synth.synthesize_to_bytes("merhaba"))

    def test_file_extension_matches_engine(self) -> None:
        synth = TTSSynthesizer(TTSConfig(engine="auto"))
        with patch.object(
            TTSSynthesizer, "_synth_gtts", return_value=_Audio(b"DATA", "mp3", "audio/mpeg")
        ):
            path = synth.synthesize_to_file("merhaba")
        self.assertIsNotNone(path)
        self.assertTrue(str(path).endswith(".mp3"))
        self.assertTrue(path.exists())
        path.unlink()

    def test_piper_writes_wav(self) -> None:
        synth = TTSSynthesizer(TTSConfig(engine="piper"))
        with patch.object(
            TTSSynthesizer, "_synth_piper", return_value=_Audio(b"DATA", "wav", "audio/wav")
        ):
            path = synth.synthesize_to_file("merhaba")
        self.assertTrue(str(path).endswith(".wav"))
        path.unlink()

    def test_status_shape(self) -> None:
        status = tts_engine_status()
        for key in ("configured", "gtts_installed", "piper_installed"):
            self.assertIn(key, status)


class LiveFlowSeamTests(unittest.TestCase):
    """The seam backend.py /api/predict/live relies on: recognized words fed
    into a rule-based, no-audio pipeline assemble into a Turkish sentence."""

    def _live_pipeline(self) -> SignTextPipeline:
        # Mirror _build_live_text_pipeline() in backend.py.
        return SignTextPipeline(
            PipelineConfig(grammar=GrammarConfig(use_ml=False), synthesize_audio=False)
        )

    def test_flush_assembles_sentence(self) -> None:
        pipeline = self._live_pipeline()
        self.assertTrue(pipeline.feed("ben", 0.9))
        self.assertTrue(pipeline.feed("okul", 0.9))
        self.assertTrue(pipeline.feed("gitmek", 0.9))
        result = pipeline.flush()
        self.assertIsNotNone(result)
        self.assertEqual(result.sentence, "Ben okula gidiyorum.")
        self.assertEqual(result.words, ["ben", "okul", "gitmek"])
        self.assertEqual(result.tts_status, "disabled")  # never touches gTTS

    def test_tick_emits_after_silence(self) -> None:
        pipeline = self._live_pipeline()
        clock = [0.0]
        pipeline.buffer._clock = lambda: clock[0]  # controllable monotonic clock
        for word in ("ben", "su", "istemek"):
            pipeline.feed(word, 0.9)
        self.assertIsNone(pipeline.tick())  # no silence yet
        clock[0] = 100.0  # advance well past the silence gap
        result = pipeline.tick()
        self.assertIsNotNone(result)
        self.assertEqual(result.sentence, "Ben su istiyorum.")

    def test_no_audio_in_live_seam(self) -> None:
        pipeline = self._live_pipeline()
        with patch.object(pipeline.tts, "synthesize_to_file") as m:
            pipeline.feed("ev", 0.9)
            pipeline.flush()
        m.assert_not_called()


class EvalHarnessTests(unittest.TestCase):
    """The labelled eval set must stay gold for the rule-based engine."""

    def test_rule_based_dataset_is_gold(self) -> None:
        from text_processing.eval import aggregate, evaluate, load_dataset

        examples = load_dataset()
        self.assertGreaterEqual(len(examples), 76)
        results = evaluate(examples, GrammarCorrector(GrammarConfig(use_ml=False)))
        report = aggregate(results, use_ml=False)
        self.assertEqual(report["exact_match"], 1.0)
        self.assertEqual(report["intent_ok"], 1.0)
        self.assertEqual(report["negation_ok"], 1.0)
        self.assertEqual(report["question_ok"], 1.0)


@unittest.skipUnless(
    os.environ.get("RUN_HF_SMOKE") == "1",
    "live HF smoke test — set RUN_HF_SMOKE=1 (and HF_TOKEN) to run",
)
class HFSmokeTest(unittest.TestCase):
    """Optional: hit the real Qwen Inference API and report latency/fallback."""

    def test_qwen_live_smoke(self) -> None:
        phrases = [
            ["ben", "su", "istemek"],
            ["sen", "okul", "gitmek"],
            ["o", "kitap", "okumak"],
            ["biz", "yemek", "yemek"],
            ["ben", "ev", "gelmek"],
        ]
        corrector = GrammarCorrector(GrammarConfig(use_ml=True, model_key="qwen2.5-7b-api"))
        ml_used, latencies = 0, []
        for words in phrases:
            result = corrector.correct_detailed(words)
            self.assertTrue(result.sentence, f"empty output for {words}")
            if result.source.startswith("ml:"):
                ml_used += 1
            if result.ml_latency_ms:
                latencies.append(result.ml_latency_ms)
        avg = sum(latencies) / len(latencies) if latencies else 0.0
        print(
            f"\n[HF smoke] ml_used={ml_used}/{len(phrases)} "
            f"fallback={len(phrases) - ml_used} avg_latency_ms={avg:.0f}"
        )


class MLErrorClassificationTests(unittest.TestCase):
    """Stage 1: broad ML failures become an actionable, categorized reason
    instead of being swallowed into an opaque ``None`` / "ml unavailable"."""

    def test_status_codes_map_to_categories(self) -> None:
        from text_processing.grammar import _classify_ml_error

        class _Resp:
            def __init__(self, code: int) -> None:
                self.status_code = code

        class _HTTPError(Exception):
            def __init__(self, msg: str, code: int) -> None:
                super().__init__(msg)
                self.response = _Resp(code)

        cases = {
            402: "quota",
            401: "auth",
            403: "auth",
            404: "not_served",
            429: "rate_limit",
            503: "loading",
            500: "provider_error",
        }
        for code, category in cases.items():
            err = _classify_ml_error(_HTTPError(f"HTTP {code}", code))
            self.assertEqual(err.category, category, f"status {code}")
            self.assertTrue(err.message)  # always a Turkish, non-empty reason

    def test_status_parsed_from_message_without_response(self) -> None:
        from text_processing.grammar import _classify_ml_error

        self.assertEqual(_classify_ml_error(RuntimeError("402 Payment Required")).category, "quota")

    def test_timeout_network_and_deps(self) -> None:
        from text_processing.grammar import _classify_ml_error

        self.assertEqual(_classify_ml_error(TimeoutError("request timed out")).category, "timeout")
        self.assertEqual(_classify_ml_error(OSError("Connection refused")).category, "network")
        self.assertEqual(
            _classify_ml_error(ModuleNotFoundError("No module named 'huggingface_hub'")).category,
            "deps",
        )

    def test_unknown_and_token_scrub(self) -> None:
        from text_processing.grammar import _classify_ml_error

        err = _classify_ml_error(RuntimeError("weird failure with hf_SECRETTOKEN12345 inside"))
        self.assertEqual(err.category, "unknown")
        self.assertNotIn("hf_SECRETTOKEN12345", err.message)
        self.assertIn("hf_***", err.message)

    def test_message_integers_are_not_misread_as_status(self) -> None:
        from text_processing.grammar import _classify_ml_error

        # Unrelated 3-digit numbers in the message must NOT be parsed as an
        # HTTP status (otherwise e.g. "503" in free text → bogus "loading").
        for msg in (
            "max_new_tokens must be <= 512",
            "see line 503 of the config file",
            "temperature 0.500 is invalid",
        ):
            self.assertEqual(_classify_ml_error(RuntimeError(msg)).category, "unknown", msg)

    def test_gateway_timeout_status_classified_as_timeout(self) -> None:
        from text_processing.grammar import _classify_ml_error

        class _Resp:
            def __init__(self, code: int) -> None:
                self.status_code = code

        class _HTTPError(Exception):
            def __init__(self, code: int) -> None:
                super().__init__(f"HTTP {code}")
                self.response = _Resp(code)

        for code in (408, 504):
            self.assertEqual(_classify_ml_error(_HTTPError(code)).category, "timeout", code)


class MLErrorPropagationTests(unittest.TestCase):
    """The classified reason flows adapter → MLGrammarCorrector → GrammarResult."""

    def tearDown(self) -> None:
        import sys

        sys.modules.pop("huggingface_hub", None)

    def _install_failing_chat_stub(self, status_code: int):
        import sys
        import types
        from unittest.mock import MagicMock

        class _Resp:
            def __init__(self, code: int) -> None:
                self.status_code = code

        class _HTTPError(Exception):
            def __init__(self, code: int) -> None:
                super().__init__(f"HTTP {code}")
                self.response = _Resp(code)

        fake_client = MagicMock()
        fake_client.chat_completion.side_effect = _HTTPError(status_code)
        fake_module = types.ModuleType("huggingface_hub")
        fake_module.InferenceClient = MagicMock(return_value=fake_client)
        sys.modules["huggingface_hub"] = fake_module

    def test_corrector_records_last_error(self) -> None:
        from text_processing.grammar import MLGrammarCorrector

        self._install_failing_chat_stub(402)
        ml = MLGrammarCorrector(GrammarConfig(use_ml=True, model_key="qwen2.5-7b-api"))
        self.assertIsNone(ml.correct(["ben", "okul", "gitmek"]))
        self.assertIsNotNone(ml.last_error)
        self.assertEqual(ml.last_error.category, "quota")

    def test_generate_candidate_returns_error_without_storing(self) -> None:
        # Race-free API: the classified error is RETURNED, not read back off
        # shared per-instance state (which concurrent requests could clobber).
        from text_processing.grammar import MLGrammarCorrector

        self._install_failing_chat_stub(402)
        ml = MLGrammarCorrector(GrammarConfig(use_ml=True, model_key="qwen2.5-7b-api"))
        text, err = ml.generate_candidate(["ben", "okul", "gitmek"])
        self.assertIsNone(text)
        self.assertIsNotNone(err)
        self.assertEqual(err.category, "quota")

    def test_empty_adapter_output_yields_empty_reason(self) -> None:
        # Adapter returns None for blank content WITHOUT setting last_error;
        # the corrector must still surface an actionable reason, not null.
        import sys
        import types
        from unittest.mock import MagicMock

        from text_processing.grammar import MLGrammarCorrector

        fake_client = MagicMock()
        fake_client.chat_completion.return_value = {
            "choices": [{"message": {"content": "   "}}]  # blank → adapter returns None
        }
        fake_module = types.ModuleType("huggingface_hub")
        fake_module.InferenceClient = MagicMock(return_value=fake_client)
        sys.modules["huggingface_hub"] = fake_module

        ml = MLGrammarCorrector(GrammarConfig(use_ml=True, model_key="qwen2.5-7b-api"))
        self.assertIsNone(ml.correct(["ben", "okul", "gitmek"]))
        self.assertIsNotNone(ml.last_error)
        self.assertEqual(ml.last_error.category, "empty")

    def test_grammar_result_surfaces_ml_error(self) -> None:
        self._install_failing_chat_stub(404)
        corrector = GrammarCorrector(GrammarConfig(use_ml=True, model_key="qwen2.5-7b-api"))
        # >2 tokens, no critical marker → the ML path actually runs.
        result = corrector.correct_detailed(["ben", "okul", "gitmek", "dün"])
        self.assertEqual(result.source, "rule-based")  # safe fallback preserved
        self.assertTrue(result.sentence)  # rule-based still produced a sentence
        self.assertEqual(result.ml_error_category, "not_served")
        self.assertTrue(result.ml_error)

    def test_decision_log_summary_counts_ml_errors(self) -> None:
        self._install_failing_chat_stub(401)
        DECISION_LOG.clear()
        GrammarCorrector(GrammarConfig(use_ml=True, model_key="qwen2.5-7b-api")).correct_detailed(
            ["sen", "okul", "gitmek", "dün"]
        )
        summary = DECISION_LOG.summary()
        self.assertGreaterEqual(summary["ml_errors"], 1)
        self.assertIn("auth", summary["last_ml_error_categories"])


class PingEndpointTests(unittest.TestCase):
    """POST /api/text/ping: a one-shot connectivity probe with a clear reason."""

    def test_local_model_reports_local_without_loading(self) -> None:
        resp = routes.ping_model(routes.PingRequest(model_key="mt0-small"))
        self.assertTrue(resp.ok)
        self.assertEqual(resp.category, "local")
        self.assertEqual(resp.arch, "seq2seq")

    def test_unknown_model_is_400(self) -> None:
        from fastapi import HTTPException

        with self.assertRaises(HTTPException) as ctx:
            routes.ping_model(routes.PingRequest(model_key="no-such-model"))
        self.assertEqual(ctx.exception.status_code, 400)

    def test_cloud_ok(self) -> None:

        class _OK:
            def __init__(self, cfg) -> None:
                self.last_error = None

            def correct(self, words):
                return "Merhaba, nasılsın?"

        with patch("text_processing.MLGrammarCorrector", _OK):
            resp = routes.ping_model(routes.PingRequest(model_key="qwen2.5-7b-api"))
        self.assertTrue(resp.ok)
        self.assertEqual(resp.category, "ok")

    def test_cloud_failure_surfaces_category(self) -> None:
        from text_processing.grammar import MLError

        class _Fail:
            def __init__(self, cfg) -> None:
                self.last_error = MLError(
                    "quota", "HF ücretsiz kotası doldu — başka bir model seçin."
                )

            def correct(self, words):
                return None

        with patch("text_processing.MLGrammarCorrector", _Fail):
            resp = routes.ping_model(routes.PingRequest(model_key="qwen2.5-7b-api"))
        self.assertFalse(resp.ok)
        self.assertEqual(resp.category, "quota")
        self.assertIn("kota", resp.detail)


if __name__ == "__main__":
    unittest.main()
