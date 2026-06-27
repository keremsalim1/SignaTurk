"""Unit tests for the text_processing pipeline. No network, no model weights."""

from __future__ import annotations

import sys
import types
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from text_processing import (
    MODEL_REGISTRY,
    BufferConfig,
    GrammarConfig,
    GrammarCorrector,
    PipelineConfig,
    RuleBasedCorrector,
    SignTextPipeline,
    TTSConfig,
    WordBuffer,
    apply_ablative,
    apply_accusative,
    apply_consonant_assimilation,
    apply_consonant_softening,
    apply_dative,
    apply_dative_suffix,
    apply_locative,
    apply_vowel_harmony,
    apply_vowel_narrowing,
    build_causal_prompt,
    build_seq2seq_prompt,
    build_zero_shot_messages,
    build_zero_shot_prompt,
    conjugate_present_continuous,
    list_available_models,
    normalize_words,
    validate_ml_output,
)


class FakeClock:
    def __init__(self, t: float = 1000.0) -> None:
        self.t = t

    def __call__(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += dt


class WordBufferTests(unittest.TestCase):
    def test_debounces_duplicate_word(self) -> None:
        clock = FakeClock()
        buf = WordBuffer(BufferConfig(debounce_seconds=1.0), clock=clock)
        self.assertTrue(buf.add("merhaba"))
        clock.advance(0.3)
        self.assertFalse(buf.add("merhaba"))
        clock.advance(0.8)
        self.assertTrue(buf.add("merhaba"))
        self.assertEqual(buf.peek(), ["merhaba", "merhaba"])

    def test_distinct_consecutive_words_pass(self) -> None:
        clock = FakeClock()
        buf = WordBuffer(BufferConfig(debounce_seconds=1.0), clock=clock)
        buf.add("ben")
        clock.advance(0.1)
        buf.add("su")
        self.assertEqual(buf.peek(), ["ben", "su"])

    def test_completes_after_silence(self) -> None:
        clock = FakeClock()
        results = []
        buf = WordBuffer(
            BufferConfig(silence_seconds=2.5),
            on_complete=results.append,
            clock=clock,
        )
        buf.add("merhaba")
        clock.advance(1.0)
        buf.add("ben")
        clock.advance(1.0)
        buf.add("su")
        clock.advance(1.0)
        buf.add("istemek")
        self.assertIsNone(buf.tick())
        clock.advance(3.0)
        completed = buf.tick()
        self.assertEqual(completed, ["merhaba", "ben", "su", "istemek"])
        self.assertEqual(results, [["merhaba", "ben", "su", "istemek"]])
        self.assertEqual(len(buf), 0)

    def test_max_length_force_flush(self) -> None:
        clock = FakeClock()
        results = []
        buf = WordBuffer(
            BufferConfig(debounce_seconds=0.0, max_sequence_length=3),
            on_complete=results.append,
            clock=clock,
        )
        buf.add("a")
        buf.add("b")
        buf.add("c")
        self.assertEqual(results, [["a", "b", "c"]])
        self.assertEqual(len(buf), 0)


class ConjugationTests(unittest.TestCase):
    def test_regular_vowel_harmony(self) -> None:
        self.assertEqual(conjugate_present_continuous("istemek", "ben"), "istiyorum")
        self.assertEqual(conjugate_present_continuous("yapmak", "ben"), "yapıyorum")
        self.assertEqual(conjugate_present_continuous("gelmek", "sen"), "geliyorsun")
        self.assertEqual(conjugate_present_continuous("içmek", "o"), "içiyor")

    def test_irregular_verbs(self) -> None:
        self.assertEqual(conjugate_present_continuous("gitmek", "ben"), "gidiyorum")
        self.assertEqual(conjugate_present_continuous("yemek", "ben"), "yiyorum")
        self.assertEqual(conjugate_present_continuous("demek", "ben"), "diyorum")
        self.assertEqual(conjugate_present_continuous("etmek", "ben"), "ediyorum")
        self.assertEqual(conjugate_present_continuous("tatmak", "biz"), "tadıyoruz")

    def test_stem_ending_in_vowel(self) -> None:
        self.assertEqual(conjugate_present_continuous("okumak", "ben"), "okuyorum")
        self.assertEqual(conjugate_present_continuous("uyumak", "o"), "uyuyor")

    def test_vowel_narrowing_in_conjugation(self) -> None:
        # a/e final stems get narrowed per §1.8.2 — including the round
        # harmony cases (oyna → oynu, söyle → söylü).
        self.assertEqual(conjugate_present_continuous("anlamak", "ben"), "anlıyorum")
        self.assertEqual(conjugate_present_continuous("oynamak", "o"), "oynuyor")
        self.assertEqual(conjugate_present_continuous("beklemek", "biz"), "bekliyoruz")
        self.assertEqual(conjugate_present_continuous("söylemek", "sen"), "söylüyorsun")
        self.assertEqual(conjugate_present_continuous("başlamak", "o"), "başlıyor")

    def test_person_endings_full_paradigm(self) -> None:
        paradigm = {
            "ben": "geliyorum",
            "sen": "geliyorsun",
            "o": "geliyor",
            "biz": "geliyoruz",
            "siz": "geliyorsunuz",
            "onlar": "geliyorlar",
        }
        for pronoun, expected in paradigm.items():
            self.assertEqual(conjugate_present_continuous("gelmek", pronoun), expected)

    def test_default_pronoun_is_first_person_singular(self) -> None:
        # Sign-language input is overwhelmingly first-person; an implicit
        # "ben" reads more naturally than the impersonal third-person.
        self.assertEqual(conjugate_present_continuous("gelmek"), "geliyorum")
        self.assertEqual(conjugate_present_continuous("okumak"), "okuyorum")

    def test_monosyllabic_stem_no_softening(self) -> None:
        # yap, tut, koş — monosyllabic, no softening of final p/t/k.
        self.assertEqual(conjugate_present_continuous("yapmak", "ben"), "yapıyorum")
        self.assertEqual(conjugate_present_continuous("tutmak", "ben"), "tutuyorum")
        self.assertEqual(conjugate_present_continuous("koşmak", "o"), "koşuyor")

    def test_buffer_vowel_4way_harmony(self) -> None:
        # Buffer vowel before -yor follows 4-way harmony on the stem.
        self.assertEqual(conjugate_present_continuous("görmek", "ben"), "görüyorum")
        self.assertEqual(conjugate_present_continuous("içmek", "ben"), "içiyorum")
        self.assertEqual(conjugate_present_continuous("gülmek", "o"), "gülüyor")


class RuleBasedCorrectorTests(unittest.TestCase):
    def test_transcript_example(self) -> None:
        # istemek is no longer in _TRANSITIVE_VERBS (indefinite-object
        # naturalness fix) → su stays bare.
        out = RuleBasedCorrector().correct(["merhaba", "ben", "su", "istemek"])
        self.assertEqual(out, "Merhaba, ben su istiyorum.")

    def test_noun_first_token_becomes_third_person_subject(self) -> None:
        # SOV parser: the first non-pronoun token is treated as a 3rd-
        # person noun subject. ``["su", "istemek"]`` therefore reads as
        # "Water wants" rather than "[I] want water" — strictly correct
        # per the spec, though semantically odd without a pronoun.
        self.assertEqual(RuleBasedCorrector().correct(["su", "istemek"]), "Su istiyor.")

    def test_verb_only_falls_back_to_first_person(self) -> None:
        # No pre-verb tokens at all → conjugation defaults to "ben",
        # which reads naturally for sign streams.
        self.assertEqual(RuleBasedCorrector().correct(["gitmek"]), "Gidiyorum.")
        self.assertEqual(RuleBasedCorrector().correct(["okumak"]), "Okuyorum.")

    def test_second_person(self) -> None:
        self.assertEqual(RuleBasedCorrector().correct(["sen", "gelmek"]), "Sen geliyorsun.")

    def test_empty_input(self) -> None:
        self.assertEqual(RuleBasedCorrector().correct([]), "")
        self.assertEqual(RuleBasedCorrector().correct(["", "  "]), "")

    def test_non_verb_untouched(self) -> None:
        self.assertEqual(
            RuleBasedCorrector().correct(["merhaba", "ben", "eray"]),
            "Merhaba, ben eray.",
        )

    # ── Dative (motion verbs) ──
    def test_dative_okul_gitmek(self) -> None:
        self.assertEqual(
            RuleBasedCorrector().correct(["sen", "okul", "gitmek"]),
            "Sen okula gidiyorsun.",
        )

    def test_dative_ev_gelmek(self) -> None:
        self.assertEqual(
            RuleBasedCorrector().correct(["o", "ev", "gelmek"]),
            "O eve geliyor.",
        )

    def test_dative_vowel_final_noun(self) -> None:
        self.assertEqual(
            RuleBasedCorrector().correct(["ben", "araba", "gitmek"]),
            "Ben arabaya gidiyorum.",
        )

    def test_dative_dönmek(self) -> None:
        self.assertEqual(
            RuleBasedCorrector().correct(["biz", "iş", "dönmek"]),
            "Biz işe dönüyoruz.",
        )

    # ── Locative (stative verbs) ──
    def test_locative_ev_beklemek(self) -> None:
        self.assertEqual(
            RuleBasedCorrector().correct(["ben", "ev", "beklemek"]),
            "Ben evde bekliyorum.",
        )

    def test_locative_okul_kalmak(self) -> None:
        self.assertEqual(
            RuleBasedCorrector().correct(["o", "okul", "kalmak"]),
            "O okulda kalıyor.",
        )

    def test_locative_hardening_after_voiceless(self) -> None:
        # iş ends in ş (voiceless) → -de hardens to -te
        self.assertEqual(
            RuleBasedCorrector().correct(["ben", "iş", "durmak"]),
            "Ben işte duruyorum.",
        )

    # ── Ablative (ablation verbs) ──
    def test_ablative_ev_çıkmak(self) -> None:
        self.assertEqual(
            RuleBasedCorrector().correct(["ben", "ev", "çıkmak"]),
            "Ben evden çıkıyorum.",
        )

    def test_ablative_hardening_after_voiceless(self) -> None:
        self.assertEqual(
            RuleBasedCorrector().correct(["o", "iş", "ayrılmak"]),
            "O işten ayrılıyor.",
        )

    # ── Accusative (transitive verbs that survived the naturalness pass) ──
    def test_accusative_su_içmek(self) -> None:
        # içmek is still in _TRANSITIVE_VERBS; vowel-final su → suyu.
        self.assertEqual(
            RuleBasedCorrector().correct(["ben", "su", "içmek"]),
            "Ben suyu içiyorum.",
        )

    def test_accusative_kitap_yazmak(self) -> None:
        # yazmak is still transitive; kitap → kitabı (softening p→b).
        self.assertEqual(
            RuleBasedCorrector().correct(["ben", "kitap", "yazmak"]),
            "Ben kitabı yazıyorum.",
        )

    # ── Indefinite-object naturalness (verbs removed from transitive set) ──
    def test_indefinite_object_okumak(self) -> None:
        # okumak removed → bare object reading, more natural in sign input.
        self.assertEqual(
            RuleBasedCorrector().correct(["ben", "kitap", "okumak"]),
            "Ben kitap okuyorum.",
        )

    def test_indefinite_object_istemek(self) -> None:
        self.assertEqual(
            RuleBasedCorrector().correct(["ben", "su", "istemek"]),
            "Ben su istiyorum.",
        )

    def test_indefinite_object_görmek(self) -> None:
        self.assertEqual(
            RuleBasedCorrector().correct(["sen", "ağaç", "görmek"]),
            "Sen ağaç görüyorsun.",
        )

    def test_indefinite_object_almak(self) -> None:
        self.assertEqual(
            RuleBasedCorrector().correct(["o", "araba", "almak"]),
            "O araba alıyor.",
        )

    # ── Homonym exception ──
    def test_homonym_noun_verb_pair_yemek_yemek(self) -> None:
        # First "yemek" stays bare (noun reading) even though the second
        # is transitive — accusative is suppressed for the homonym pair.
        self.assertEqual(
            RuleBasedCorrector().correct(["biz", "yemek", "yemek"]),
            "Biz yemek yiyoruz.",
        )

    def test_homonym_default_first_yemek_as_subject(self) -> None:
        # Under SOV parsing, the first ``yemek`` is the subject (3rd
        # person) — there is no pronoun to override it. The natural
        # "I'm eating food" reading requires an explicit subject pronoun
        # (see ``test_homonym_noun_verb_pair_yemek_yemek``).
        self.assertEqual(
            RuleBasedCorrector().correct(["yemek", "yemek"]),
            "Yemek yiyor.",
        )

    # ── Generalized SOV regression / coverage ──
    def test_sov_screenshot_regression_sen_ben_sevmek(self) -> None:
        # The exact bug from the demo screenshot: two pronouns, the
        # second one must take accusative ("beni"), and the verb must
        # agree with the FIRST pronoun ("sen" → -sun).
        self.assertEqual(
            RuleBasedCorrector().correct(["sen", "ben", "sevmek"]),
            "Sen beni seviyorsun.",
        )

    def test_sov_screenshot_regression_inverted(self) -> None:
        # Same shape with the pronouns swapped — agreement and the
        # accusative target both flip.
        self.assertEqual(
            RuleBasedCorrector().correct(["ben", "sen", "anlamak"]),
            "Ben seni anlıyorum.",
        )

    def test_irregular_pronoun_case_forms(self) -> None:
        # Personal pronouns decline irregularly: dative ben->bana (not the
        # generic "bene"), o->ona/onu, etc. The case-applier consults the
        # explicit table before the regular noun suffixers.
        self.assertEqual(RuleBasedCorrector._apply_case("ben", "dative"), "bana")
        self.assertEqual(RuleBasedCorrector._apply_case("sen", "dative"), "sana")
        self.assertEqual(RuleBasedCorrector._apply_case("o", "dative"), "ona")
        self.assertEqual(RuleBasedCorrector._apply_case("o", "accusative"), "onu")
        self.assertEqual(RuleBasedCorrector._apply_case("o", "locative"), "onda")
        self.assertEqual(RuleBasedCorrector._apply_case("onlar", "ablative"), "onlardan")
        # No case requested → unchanged (subject position).
        self.assertEqual(RuleBasedCorrector._apply_case("ben", None), "ben")

    def test_pronoun_dative_object_in_sentence(self) -> None:
        # Motion verb governs dative; the pronoun object now declines correctly
        # ("bana"/"ona") instead of the malformed "bene"/"oye".
        self.assertEqual(
            RuleBasedCorrector().correct(["sen", "ben", "gitmek"]),
            "Sen bana gidiyorsun.",
        )
        self.assertEqual(
            RuleBasedCorrector().correct(["ben", "o", "gitmek"]),
            "Ben ona gidiyorum.",
        )

    def test_single_token_pronoun_predicate_agreement(self) -> None:
        # A lone pronoun + marker is the subject, so the copula/question must
        # agree with its person (was hard-coded to 3rd person).
        r = RuleBasedCorrector()
        self.assertEqual(r.correct(["ben", "değil"]), "Ben değilim.")
        self.assertEqual(r.correct(["ben", "yok"]), "Ben yokum.")
        self.assertEqual(r.correct(["ben", "mi"]), "Ben miyim?")
        self.assertEqual(r.correct(["biz", "yok"]), "Biz yokuz.")
        self.assertEqual(r.correct(["sen", "değil"]), "Sen değilsin.")
        # 3rd person and non-pronoun predicates stay 3s.
        self.assertEqual(r.correct(["o", "mi"]), "O mu?")
        self.assertEqual(r.correct(["o", "değil"]), "O değil.")

    def test_sov_proper_noun_subject(self) -> None:
        # The first token isn't in _PRONOUNS so it's treated as a 3rd-
        # person noun subject; the middle token gets accusative.
        self.assertEqual(
            RuleBasedCorrector().correct(["ahmet", "kitap", "yazmak"]),
            "Ahmet kitabı yazıyor.",
        )

    def test_sov_greeting_only_input(self) -> None:
        self.assertEqual(RuleBasedCorrector().correct(["merhaba"]), "Merhaba.")

    # ── Dynamic verb-scan: inverted/devrik inputs ──
    def test_sov_verb_first_canonicalizes(self) -> None:
        # Verb arrives first in the stream — parser must still find it
        # and emit canonical SOV order.
        self.assertEqual(
            RuleBasedCorrector().correct(["gitmek", "sen", "okul"]),
            "Sen okula gidiyorsun.",
        )

    def test_sov_verb_middle_canonicalizes(self) -> None:
        # Verb in the middle of the stream.
        self.assertEqual(
            RuleBasedCorrector().correct(["sen", "gitmek", "okul"]),
            "Sen okula gidiyorsun.",
        )

    def test_sov_inverted_multi_pronoun(self) -> None:
        # Verb first, then two pronouns — accusative on the second
        # pronoun, agreement off the first.
        self.assertEqual(
            RuleBasedCorrector().correct(["sevmek", "sen", "ben"]),
            "Sen beni seviyorsun.",
        )

    def test_sov_picks_rightmost_infinitive(self) -> None:
        # Two infinitives — the rightmost is the main verb; the earlier
        # one stays in mastar (complement-verb pattern).
        self.assertEqual(
            RuleBasedCorrector().correct(["ben", "gelmek", "istemek"]),
            "Ben gelmek istiyorum.",
        )

    # ── Capitalization + punctuation ──
    def test_capitalization_and_terminator(self) -> None:
        out = RuleBasedCorrector().correct(["ben", "okul", "gitmek"])
        self.assertTrue(out[0].isupper())
        self.assertTrue(out.endswith("."))

    def test_greeting_gets_comma(self) -> None:
        out = RuleBasedCorrector().correct(["merhaba", "ben", "su", "içmek"])
        self.assertTrue(out.startswith("Merhaba,"))

    # ── Predicate-shape: question particle (mi/mı/mu/mü) ──

    def test_question_particle_second_person(self) -> None:
        self.assertEqual(
            RuleBasedCorrector().correct(["sen", "gelmek", "mi"]),
            "Sen geliyor musun?",
        )

    def test_question_particle_first_person(self) -> None:
        self.assertEqual(
            RuleBasedCorrector().correct(["ben", "gelmek", "mi"]),
            "Ben geliyor muyum?",
        )

    def test_question_particle_third_person(self) -> None:
        self.assertEqual(
            RuleBasedCorrector().correct(["o", "gelmek", "mi"]),
            "O geliyor mu?",
        )

    def test_question_particle_with_dative_object(self) -> None:
        # The arbiter's middle-token case marking still runs on the
        # object (``okula``); only the question particle is exempt.
        self.assertEqual(
            RuleBasedCorrector().correct(["sen", "okul", "gitmek", "mi"]),
            "Sen okula gidiyor musun?",
        )

    def test_question_particle_does_not_get_dative_suffix(self) -> None:
        # Regression: the original bug produced ``Sen miye geliyorsun.``
        # because ``mi`` was case-marked as a dative object.
        out = RuleBasedCorrector().correct(["sen", "gelmek", "mi"])
        self.assertNotIn("miye", out.lower())
        self.assertNotIn("miyi", out.lower())

    def test_question_particle_on_nominal_predicate(self) -> None:
        self.assertEqual(
            RuleBasedCorrector().correct(["sen", "öğrenci", "mi"]),
            "Sen öğrenci misin?",
        )

    # ── Predicate-shape: verbal negation via değil (§2.6.1) ──

    def test_verbal_negation_first_person(self) -> None:
        self.assertEqual(
            RuleBasedCorrector().correct(["ben", "gelmek", "değil"]),
            "Ben gelmiyorum.",
        )

    def test_verbal_negation_second_person(self) -> None:
        self.assertEqual(
            RuleBasedCorrector().correct(["sen", "gitmek", "değil"]),
            "Sen gitmiyorsun.",
        )

    def test_verbal_negation_third_person_with_dative_object(self) -> None:
        self.assertEqual(
            RuleBasedCorrector().correct(["o", "okul", "gitmek", "değil"]),
            "O okula gitmiyor.",
        )

    def test_verbal_negation_back_vowel_verb(self) -> None:
        self.assertEqual(
            RuleBasedCorrector().correct(["ben", "okumak", "değil"]),
            "Ben okumuyorum.",
        )

    def test_verbal_negation_drops_degil_token(self) -> None:
        # When the verb is conjugated negatively, the standalone
        # ``değil`` copula must not appear in the surface form.
        out = RuleBasedCorrector().correct(["ben", "gelmek", "değil"])
        self.assertNotIn("değil", out.lower())
        self.assertNotIn("değile", out.lower())

    def test_verbal_negation_no_case_marker_leak(self) -> None:
        # Hard regression: the marker must NEVER reach _apply_case.
        # Asserts the *exact* expected output and forbids every form
        # the leak has historically produced.
        out = RuleBasedCorrector().correct(["ben", "gelmek", "değil"])
        self.assertEqual(out, "Ben gelmiyorum.")
        lower = out.lower()
        for leaked in ("değle", "değile", "değili", "deği̇le", "deği̇li"):
            self.assertNotIn(leaked, lower, f"{leaked!r} leaked into output")

    def test_verbal_negation_handles_uppercase_turkish_input(self) -> None:
        # The Turkish dotted/dotless ``i`` pair (İ ↔ i, I ↔ ı) tripped
        # Python's locale-naive ``str.lower()`` and caused ``DEĞİL``
        # to bypass the negation lookup, slip into the SOV pipeline,
        # and emerge dative-marked as ``deği̇le``. Casefolding must
        # canonicalize all four code points.
        out = RuleBasedCorrector().correct(["BEN", "GELMEK", "DEĞİL"])
        self.assertEqual(out, "Ben gelmiyorum.")

    def test_dual_marker_hayir_plus_degil_is_idempotent(self) -> None:
        # Both markers force verb negation; co-occurrence must collapse
        # to a single cohesive negative predicate without crashing or
        # double-suffixing.
        out = RuleBasedCorrector().correct(["hayır", "ben", "gitmek", "değil"])
        self.assertEqual(out, "Hayır, ben gitmiyorum.")
        self.assertEqual(out.count("değil"), 0)
        self.assertEqual(out.lower().count("hayır"), 1)

    def test_duplicate_degil_tokens_are_idempotent(self) -> None:
        out = RuleBasedCorrector().correct(["ben", "gelmek", "değil", "değil"])
        self.assertEqual(out, "Ben gelmiyorum.")

    # ── Nominal negation: değil stays as copula when there's no verb ──

    def test_nominal_negation_keeps_degil_copula(self) -> None:
        self.assertEqual(
            RuleBasedCorrector().correct(["ben", "öğrenci", "değil"]),
            "Ben öğrenci değilim.",
        )

    # ── Predicate-shape: existential negation (yok) ──

    def test_existential_negation_third_person(self) -> None:
        self.assertEqual(RuleBasedCorrector().correct(["ev", "yok"]), "Ev yok.")

    def test_existential_negation_first_person(self) -> None:
        self.assertEqual(
            RuleBasedCorrector().correct(["ben", "evde", "yok"]),
            "Ben evde yokum.",
        )

    # ── Predicate-shape: standalone negative (hayır) ──

    def test_standalone_hayir_alone(self) -> None:
        self.assertEqual(RuleBasedCorrector().correct(["hayır"]), "Hayır.")

    def test_standalone_hayir_forces_verb_negation(self) -> None:
        # ``hayır`` is semantically negative: any accompanying verb in
        # the same clause must be conjugated in the negative form even
        # without an explicit ``değil`` token.
        self.assertEqual(
            RuleBasedCorrector().correct(["hayır", "ben", "gelmek"]),
            "Hayır, ben gelmiyorum.",
        )

    def test_standalone_hayir_with_dative_object(self) -> None:
        self.assertEqual(
            RuleBasedCorrector().correct(["hayır", "sen", "okul", "gitmek"]),
            "Hayır, sen okula gitmiyorsun.",
        )

    # ── Single-token safeguards ──

    def test_single_number_word_returns_capitalized(self) -> None:
        self.assertEqual(RuleBasedCorrector().correct(["üç"]), "Üç.")

    def test_single_digit_returns_capitalized(self) -> None:
        self.assertEqual(RuleBasedCorrector().correct(["5"]), "5.")

    def test_single_noun_returns_capitalized(self) -> None:
        self.assertEqual(RuleBasedCorrector().correct(["ev"]), "Ev.")

    def test_whitespace_only_tokens_return_empty(self) -> None:
        self.assertEqual(RuleBasedCorrector().correct(["  ", "\t", ""]), "")

    def test_output_has_no_duplicate_spaces(self) -> None:
        # Stress the punctuator with several middle tokens; ensure
        # collapsed whitespace and a single trailing terminator.
        out = RuleBasedCorrector().correct(["ben", "okul", "yarın", "gitmek"])
        self.assertNotIn("  ", out)
        self.assertEqual(out.count("."), 1)
        self.assertTrue(out.endswith("."))


class VowelHarmonyTests(unittest.TestCase):
    def test_2way_back_returns_a(self) -> None:
        for w in ["okul", "kitap", "araba", "su", "yol"]:
            self.assertEqual(apply_vowel_harmony(w, "2way"), "a", w)

    def test_2way_front_returns_e(self) -> None:
        for w in ["ev", "deniz", "göz", "köy", "gül"]:
            self.assertEqual(apply_vowel_harmony(w, "2way"), "e", w)

    def test_4way_full_paradigm(self) -> None:
        # back unrounded → ı, back rounded → u,
        # front unrounded → i, front rounded → ü.
        self.assertEqual(apply_vowel_harmony("kitap", "4way"), "ı")
        self.assertEqual(apply_vowel_harmony("okul", "4way"), "u")
        self.assertEqual(apply_vowel_harmony("ev", "4way"), "i")
        self.assertEqual(apply_vowel_harmony("göz", "4way"), "ü")

    def test_invalid_kind_raises(self) -> None:
        with self.assertRaises(ValueError):
            apply_vowel_harmony("ev", "8way")


class ConsonantSofteningTests(unittest.TestCase):
    def test_multi_syllable_softening(self) -> None:
        self.assertEqual(apply_consonant_softening("kitap"), "kitab")
        self.assertEqual(apply_consonant_softening("ağaç"), "ağac")
        self.assertEqual(apply_consonant_softening("kanat"), "kanad")
        self.assertEqual(apply_consonant_softening("çocuk"), "çocuğ")

    def test_monosyllable_no_softening(self) -> None:
        for w in ["park", "at", "ip", "ek", "saç", "yap", "tut"]:
            self.assertEqual(apply_consonant_softening(w), w, w)

    def test_monosyllable_exceptions(self) -> None:
        # Lexical exceptions from §1.7.3.
        self.assertEqual(apply_consonant_softening("renk"), "reng")
        self.assertEqual(apply_consonant_softening("uç"), "uc")
        self.assertEqual(apply_consonant_softening("dip"), "dib")

    def test_non_softening_final(self) -> None:
        # Consonants outside p/ç/t/k are unchanged.
        for w in ["ev", "yol", "kaz", "kız"]:
            self.assertEqual(apply_consonant_softening(w), w, w)

    def test_empty_returns_empty(self) -> None:
        self.assertEqual(apply_consonant_softening(""), "")


class ConsonantAssimilationTests(unittest.TestCase):
    def test_d_to_t_after_voiceless(self) -> None:
        self.assertEqual(apply_consonant_assimilation("kitap", "da"), "ta")
        self.assertEqual(apply_consonant_assimilation("ağaç", "dan"), "tan")
        self.assertEqual(apply_consonant_assimilation("çiçek", "de"), "te")

    def test_c_to_ç_after_voiceless(self) -> None:
        self.assertEqual(apply_consonant_assimilation("çiçek", "ci"), "çi")
        self.assertEqual(apply_consonant_assimilation("çorap", "cı"), "çı")

    def test_g_to_k_after_voiceless(self) -> None:
        self.assertEqual(apply_consonant_assimilation("kitap", "gi"), "ki")

    def test_no_change_after_voiced(self) -> None:
        self.assertEqual(apply_consonant_assimilation("ev", "de"), "de")
        self.assertEqual(apply_consonant_assimilation("kaz", "dan"), "dan")

    def test_no_change_for_non_cdg_suffix(self) -> None:
        self.assertEqual(apply_consonant_assimilation("kitap", "lar"), "lar")


class VowelNarrowingTests(unittest.TestCase):
    def test_basic_narrowing(self) -> None:
        # Examples from TDK guide §1.8.2 verbatim.
        self.assertEqual(apply_vowel_narrowing("başla"), "başlı")
        self.assertEqual(apply_vowel_narrowing("oyna"), "oynu")
        self.assertEqual(apply_vowel_narrowing("bekle"), "bekli")
        self.assertEqual(apply_vowel_narrowing("izle"), "izli")

    def test_round_harmony_takes_previous_vowel(self) -> None:
        # Roundness comes from the *previous* vowel, not the a/e itself.
        self.assertEqual(apply_vowel_narrowing("söyle"), "söylü")  # ö → ü
        self.assertEqual(apply_vowel_narrowing("oyna"), "oynu")  # o → u

    def test_monosyllable_uses_self_as_reference(self) -> None:
        self.assertEqual(apply_vowel_narrowing("ye"), "yi")
        self.assertEqual(apply_vowel_narrowing("de"), "di")

    def test_no_narrowing_if_not_a_or_e(self) -> None:
        for w in ["oku", "uyu", "iste", "git"]:
            # iste ends in e — should narrow; git ends in t — should not.
            if w[-1] in "ae":
                continue
            self.assertEqual(apply_vowel_narrowing(w), w, w)


class CaseSuffixTests(unittest.TestCase):
    # ── Dative ──
    def test_dative_back_consonant_final(self) -> None:
        self.assertEqual(apply_dative("okul"), "okula")

    def test_dative_front_consonant_final(self) -> None:
        self.assertEqual(apply_dative("ev"), "eve")

    def test_dative_vowel_final_takes_y_buffer(self) -> None:
        self.assertEqual(apply_dative("araba"), "arabaya")
        self.assertEqual(apply_dative("kapı"), "kapıya")
        self.assertEqual(apply_dative("su"), "suya")

    def test_dative_multi_syllable_softening(self) -> None:
        self.assertEqual(apply_dative("ayak"), "ayağa")
        self.assertEqual(apply_dative("kitap"), "kitaba")
        self.assertEqual(apply_dative("ağaç"), "ağaca")

    def test_dative_monosyllable_no_softening(self) -> None:
        self.assertEqual(apply_dative("park"), "parka")

    def test_dative_back_compat_alias(self) -> None:
        # The old name still works for callers that imported it.
        self.assertEqual(apply_dative_suffix("ev"), "eve")

    # ── Locative ──
    def test_locative_voiced_stem(self) -> None:
        self.assertEqual(apply_locative("ev"), "evde")
        self.assertEqual(apply_locative("okul"), "okulda")

    def test_locative_voiceless_stem_hardens_d_to_t(self) -> None:
        self.assertEqual(apply_locative("kitap"), "kitapta")
        self.assertEqual(apply_locative("ağaç"), "ağaçta")
        self.assertEqual(apply_locative("iş"), "işte")

    def test_locative_vowel_final(self) -> None:
        self.assertEqual(apply_locative("araba"), "arabada")
        self.assertEqual(apply_locative("kapı"), "kapıda")

    # ── Ablative ──
    def test_ablative_voiced_stem(self) -> None:
        self.assertEqual(apply_ablative("ev"), "evden")
        self.assertEqual(apply_ablative("okul"), "okuldan")

    def test_ablative_voiceless_stem_hardens(self) -> None:
        self.assertEqual(apply_ablative("kitap"), "kitaptan")
        self.assertEqual(apply_ablative("ağaç"), "ağaçtan")
        self.assertEqual(apply_ablative("iş"), "işten")

    # ── Accusative ──
    def test_accusative_consonant_final_no_softening(self) -> None:
        self.assertEqual(apply_accusative("ev"), "evi")
        self.assertEqual(apply_accusative("yol"), "yolu")

    def test_accusative_consonant_final_softening(self) -> None:
        self.assertEqual(apply_accusative("kitap"), "kitabı")
        self.assertEqual(apply_accusative("ağaç"), "ağacı")
        self.assertEqual(apply_accusative("çocuk"), "çocuğu")

    def test_accusative_mono_softening_exception(self) -> None:
        # §1.7.3: renk softens despite being monosyllabic.
        self.assertEqual(apply_accusative("renk"), "rengi")

    def test_accusative_vowel_final_y_buffer(self) -> None:
        self.assertEqual(apply_accusative("araba"), "arabayı")
        self.assertEqual(apply_accusative("kapı"), "kapıyı")
        self.assertEqual(apply_accusative("su"), "suyu")

    # ── Empty input ──
    def test_empty_returns_input(self) -> None:
        self.assertEqual(apply_dative(""), "")
        self.assertEqual(apply_locative(""), "")
        self.assertEqual(apply_ablative(""), "")
        self.assertEqual(apply_accusative(""), "")


class ModelRegistryTests(unittest.TestCase):
    def test_default_model_present(self) -> None:
        from text_processing import DEFAULT_MODEL_KEY

        self.assertIn(DEFAULT_MODEL_KEY, MODEL_REGISTRY)

    def test_registry_listing(self) -> None:
        specs = list_available_models()
        self.assertGreaterEqual(len(specs), 4)
        for spec in specs:
            self.assertIn(spec.arch, {"seq2seq", "causal", "inference-api"})
            self.assertTrue(spec.hf_name)

    def test_qwen_is_recommended_api_model(self) -> None:
        self.assertIn("qwen2.5-7b-api", MODEL_REGISTRY)
        spec = MODEL_REGISTRY["qwen2.5-7b-api"]
        self.assertEqual(spec.arch, "inference-api")
        self.assertEqual(spec.hf_name, "Qwen/Qwen2.5-7B-Instruct")
        self.assertTrue(spec.recommended)
        self.assertTrue(spec.instruction_tuned)
        self.assertTrue(spec.conversational)

    def test_registry_is_lean_qwen_is_only_api_model(self) -> None:
        # Qwen2.5-7B is the only *recommended/default* cloud model. Extra API
        # entries (e.g. an experimental Qwen3 for A/B) are allowed but must be
        # non-recommended so the UI never auto-selects them.
        recommended_api = [
            k for k, s in MODEL_REGISTRY.items() if s.arch == "inference-api" and s.recommended
        ]
        self.assertEqual(recommended_api, ["qwen2.5-7b-api"])
        # The earlier mt0-*-api entries should be gone.
        for legacy_key in ("mt0-small-api", "mt0-base-api", "mt0-large-api", "mt0-xl-api"):
            self.assertNotIn(legacy_key, MODEL_REGISTRY, legacy_key)

    def test_only_one_recommended_model(self) -> None:
        # The dropdown auto-selects the first recommended model — keep
        # exactly one to avoid ambiguous UX.
        recommended = [s for s in list_available_models() if s.recommended]
        self.assertEqual(len(recommended), 1)
        self.assertEqual(recommended[0].key, "qwen2.5-7b-api")

    def test_resolve_unknown_key_raises(self) -> None:
        cfg = GrammarConfig(use_ml=True, model_key="does-not-exist")
        with self.assertRaises(ValueError):
            cfg.resolve_spec()

    def test_resolve_override(self) -> None:
        cfg = GrammarConfig(use_ml=True, model_name_override="custom/model")
        spec = cfg.resolve_spec()
        self.assertEqual(spec.hf_name, "custom/model")


class PromptTests(unittest.TestCase):
    def test_zero_shot_prompt_structure(self) -> None:
        prompt = build_zero_shot_prompt(["ben", "okul", "gitmek"])
        self.assertIn("Verilen işaret dili kelimelerini", prompt)
        self.assertIn("Kelimeler: ben okul gitmek", prompt)
        self.assertTrue(prompt.rstrip().endswith("Cümle:"))

    def test_zero_shot_strips_trailing_punctuation(self) -> None:
        # Input pre-processing means "gitmek." normalizes to "gitmek".
        prompt = build_zero_shot_prompt(["ben", "okul", "gitmek."])
        self.assertIn("Kelimeler: ben okul gitmek", prompt)
        self.assertNotIn("gitmek.", prompt)

    def test_no_few_shot_examples_in_prompt(self) -> None:
        # All four legacy few-shot artefacts must be absent — the spec
        # explicitly removed them to stop the parrot-the-example bug.
        prompt = build_zero_shot_prompt(["sen", "okul", "gitmek"])
        for banned in (
            "Girdi:",  # old seq2seq scaffold label
            "Çıktı:",  # old seq2seq scaffold label
            "Merhaba,",  # leaked few-shot output
            "su istiyorum",  # leaked few-shot output
            "gidiyorsun",  # leaked few-shot output
            "yiyoruz",  # leaked few-shot output
        ):
            self.assertNotIn(banned, prompt, f"prompt leaked: {banned!r}")

    def test_seq2seq_and_causal_aliases_match_zero_shot(self) -> None:
        # Historical helpers are now thin aliases on top of the unified
        # zero-shot builder so every adapter sees identical input.
        words = ["ben", "kitap", "okumak"]
        self.assertEqual(build_seq2seq_prompt(words), build_zero_shot_prompt(words))
        self.assertEqual(build_causal_prompt(words), build_zero_shot_prompt(words))


class GrammarConfigDefaultsTests(unittest.TestCase):
    def test_deterministic_decoding_defaults(self) -> None:
        # Defaults must enforce strict greedy decoding so the ML branch
        # cannot hallucinate from the zero-shot prompt.
        cfg = GrammarConfig()
        self.assertEqual(cfg.num_beams, 1)
        self.assertLessEqual(cfg.temperature, 0.1 + 1e-9)
        self.assertEqual(cfg.max_new_tokens, 50)


class NormalizeWordsTests(unittest.TestCase):
    def test_strips_trailing_period(self) -> None:
        self.assertEqual(normalize_words(["ben", "okul", "gitmek."]), ["ben", "okul", "gitmek"])

    def test_strips_mixed_punctuation_and_case(self) -> None:
        self.assertEqual(
            normalize_words(["Merhaba,", " Ben ", "su!"]),
            ["merhaba", "ben", "su"],
        )

    def test_drops_punctuation_only_tokens(self) -> None:
        self.assertEqual(normalize_words([".", ",", "  ", "ev"]), ["ev"])

    def test_handles_unicode_quotes(self) -> None:
        self.assertEqual(normalize_words(["“merhaba”", "“ben”"]), ["merhaba", "ben"])

    def test_rule_based_handles_period_input(self) -> None:
        # The exact failure from the demo screenshot — should now work.
        out = RuleBasedCorrector().correct(["ben", "okul", "gitmek."])
        self.assertEqual(out, "Ben okula gidiyorum.")


class ValidatorTests(unittest.TestCase):
    def test_empty_rejected(self) -> None:
        self.assertFalse(validate_ml_output("", ["x"]).ok)
        self.assertFalse(validate_ml_output("   ", ["x"]).ok)

    def test_internal_tokens_rejected(self) -> None:
        v = validate_ml_output("<extra_id_0> foo", ["foo"])
        self.assertFalse(v.ok)
        self.assertIn("model-internal", v.reason)

    def test_too_long_rejected(self) -> None:
        v = validate_ml_output("a" * 500, ["x"])
        self.assertFalse(v.ok)
        self.assertIn("too long", v.reason)

    def test_too_short_rejected(self) -> None:
        v = validate_ml_output("ab", ["merhaba"])
        self.assertFalse(v.ok)
        self.assertIn("too short", v.reason)

    def test_no_turkish_and_no_overlap_rejected(self) -> None:
        v = validate_ml_output("xxx yyy zzz", ["merhaba", "ben"])
        self.assertFalse(v.ok)

    def test_input_overlap_accepted(self) -> None:
        v = validate_ml_output("Merhaba ben buradayim", ["merhaba", "ben"])
        self.assertTrue(v.ok)

    def test_turkish_chars_accepted(self) -> None:
        v = validate_ml_output("İyi günler dilerim çok güzel", ["foo"])
        self.assertTrue(v.ok)

    def test_scaffolding_token_rejected(self) -> None:
        # The exact failure mode from the demo screenshot.
        v = validate_ml_output("Girdi: merhaba, ben su istemek", ["ben", "okul", "gitmek"])
        self.assertFalse(v.ok)
        self.assertIn("scaffolding", v.reason)

    def test_scaffolding_other_labels_rejected(self) -> None:
        for leak in ("Çıktı: foo bar baz", "Kelimeler: ben gel", "Cümle: bir şey"):
            v = validate_ml_output(leak, ["ben", "gel"])
            self.assertFalse(v.ok, f"should reject: {leak!r}")

    # NOTE: the older few-shot-input-leak tests were dropped because the
    # prompt is now zero-shot — there are no example phrases for the
    # model to parrot. The scaffolding regex above (Girdi:, Çıktı:,
    # Kelimeler:, Cümle:) still catches the practically important leak.


class HybridGrammarTests(unittest.TestCase):
    def test_ml_disabled_uses_rule_based(self) -> None:
        c = GrammarCorrector(GrammarConfig(use_ml=False))
        result = c.correct_detailed(["merhaba", "ben", "su", "istemek"])
        self.assertEqual(result.sentence, "Merhaba, ben su istiyorum.")
        self.assertEqual(result.source, "rule-based")

    def test_ml_good_candidate_is_returned(self) -> None:
        c = GrammarCorrector(GrammarConfig(use_ml=True))
        self.assertIsNotNone(c._ml)
        candidate = "Merhaba, ben su istiyorum."
        with patch.object(c._ml, "generate_candidate", return_value=(candidate, None)):
            result = c.correct_detailed(["merhaba", "ben", "su", "istemek"])
        self.assertEqual(result.sentence, candidate)
        self.assertTrue(result.source.startswith("ml:"))

    def test_ml_bad_candidate_falls_back(self) -> None:
        c = GrammarCorrector(GrammarConfig(use_ml=True))
        with patch.object(c._ml, "generate_candidate", return_value=("<extra_id_0>", None)):
            result = c.correct_detailed(["merhaba", "ben", "su", "istemek"])
        self.assertEqual(result.sentence, "Merhaba, ben su istiyorum.")
        self.assertEqual(result.source, "rule-based")
        self.assertEqual(result.rejected_candidate, "<extra_id_0>")
        self.assertIsNotNone(result.rejection_reason)

    def test_ml_returns_none_falls_back(self) -> None:
        c = GrammarCorrector(GrammarConfig(use_ml=True))
        with patch.object(c._ml, "generate_candidate", return_value=(None, None)):
            result = c.correct_detailed(["merhaba", "ben", "su", "istemek"])
        self.assertEqual(result.source, "rule-based")
        self.assertIsNone(result.rejected_candidate)

    def test_demo_screenshot_regression_falls_back_to_dative(self) -> None:
        """End-to-end repro of the demo bug: user types 'ben okul gitmek.',
        local model parrots the few-shot template ('Girdi: merhaba, ben su istemek'),
        validator must reject and rule-based must produce the dative form."""
        c = GrammarCorrector(GrammarConfig(use_ml=True))
        leaked = "Girdi: merhaba, ben su istemek"
        with patch.object(c._ml, "generate_candidate", return_value=(leaked, None)):
            result = c.correct_detailed(["ben", "okul", "gitmek."])
        self.assertEqual(result.sentence, "Ben okula gidiyorum.")
        self.assertEqual(result.source, "rule-based")
        self.assertEqual(result.rejected_candidate, leaked)
        self.assertIn("scaffolding", (result.rejection_reason or ""))


class InferenceAPIAdapterTests(unittest.TestCase):
    """Adapter must not hit the network — stub huggingface_hub."""

    def _install_stub(self, returned_text: str = "Sen okula gidiyorsun."):
        fake_client = MagicMock()
        fake_client.text_generation.return_value = returned_text
        fake_module = types.ModuleType("huggingface_hub")
        fake_module.InferenceClient = MagicMock(return_value=fake_client)
        sys.modules["huggingface_hub"] = fake_module
        return fake_module, fake_client

    def tearDown(self) -> None:
        sys.modules.pop("huggingface_hub", None)

    @staticmethod
    def _text_generation_spec():
        """Synthetic non-conversational inference-api spec used to
        exercise the legacy text_generation transport path. The lean
        production registry no longer ships such a model (Qwen is
        conversational), so the tests build one directly."""
        from text_processing.grammar import ModelSpec

        return ModelSpec(
            key="test-textgen-api",
            hf_name="test/non-conversational",
            arch="inference-api",
            approx_size_mb=0,
            instruction_tuned=True,
            conversational=False,
        )

    def test_inference_api_adapter_returns_text(self) -> None:
        self._install_stub("Sen okula gidiyorsun.")
        from text_processing.grammar import _InferenceAPIAdapter

        cfg = GrammarConfig(use_ml=True)
        adapter = _InferenceAPIAdapter(self._text_generation_spec(), cfg)
        out = adapter.generate(["sen", "okul", "gitmek"], cfg)
        self.assertEqual(out, "Sen okula gidiyorsun.")

    def test_inference_api_adapter_propagates_exception(self) -> None:
        # The adapter RAISES transport errors (the corrector classifies them);
        # it no longer swallows them to None. None means empty output only.
        _, fake_client = self._install_stub()
        fake_client.text_generation.side_effect = RuntimeError("boom")
        from text_processing.grammar import _InferenceAPIAdapter

        cfg = GrammarConfig(use_ml=True)
        adapter = _InferenceAPIAdapter(self._text_generation_spec(), cfg)
        with self.assertRaises(RuntimeError):
            adapter.generate(["sen", "okul", "gitmek"], cfg)

    def test_inference_api_adapter_reads_hf_token_env(self) -> None:
        fake_module, _ = self._install_stub()
        from text_processing.grammar import _InferenceAPIAdapter

        cfg = GrammarConfig(use_ml=True)
        spec = self._text_generation_spec()
        with patch.dict("os.environ", {"HF_TOKEN": "hf_dummy_token"}, clear=False):
            _InferenceAPIAdapter(spec, cfg)
        kwargs = fake_module.InferenceClient.call_args.kwargs
        self.assertEqual(kwargs.get("token"), "hf_dummy_token")
        self.assertEqual(kwargs.get("model"), "test/non-conversational")

    # ── Conversational dispatch (Qwen path) ──

    def _install_chat_stub(self, content: str = "Ben okula gidiyorum."):
        """Stub a huggingface_hub whose client exposes chat_completion."""
        fake_client = MagicMock()
        fake_client.chat_completion.return_value = {
            "choices": [{"message": {"role": "assistant", "content": content}}]
        }
        fake_module = types.ModuleType("huggingface_hub")
        fake_module.InferenceClient = MagicMock(return_value=fake_client)
        sys.modules["huggingface_hub"] = fake_module
        return fake_module, fake_client

    def test_qwen_dispatches_to_chat_completion(self) -> None:
        _, fake_client = self._install_chat_stub("Ben okula gidiyorum.")
        from text_processing.grammar import _InferenceAPIAdapter

        cfg = GrammarConfig(use_ml=True, model_key="qwen2.5-7b-api")
        adapter = _InferenceAPIAdapter(cfg.resolve_spec(), cfg)
        out = adapter.generate(["ben", "okul", "gitmek"], cfg)

        self.assertEqual(out, "Ben okula gidiyorum.")
        fake_client.chat_completion.assert_called_once()
        # Conversational dispatch MUST NOT touch text_generation —
        # that's the call the HF gateway rejects for Qwen.
        fake_client.text_generation.assert_not_called()

    def test_chat_completion_messages_structure(self) -> None:
        _, fake_client = self._install_chat_stub()
        from text_processing.grammar import _InferenceAPIAdapter

        cfg = GrammarConfig(use_ml=True, model_key="qwen2.5-7b-api")
        adapter = _InferenceAPIAdapter(cfg.resolve_spec(), cfg)
        adapter.generate(["sen", "okul", "gitmek"], cfg)

        kwargs = fake_client.chat_completion.call_args.kwargs
        messages = kwargs.get("messages")
        self.assertIsInstance(messages, list)
        self.assertEqual(messages[0]["role"], "system")
        # Persona token that survives prompt rewrites.
        self.assertIn("çevirmen", messages[0]["content"])
        self.assertEqual(messages[1]["role"], "user")
        self.assertIn("sen okul gitmek", messages[1]["content"])

    def test_chat_completion_strict_decoding(self) -> None:
        _, fake_client = self._install_chat_stub()
        from text_processing.grammar import _InferenceAPIAdapter

        cfg = GrammarConfig(use_ml=True, model_key="qwen2.5-7b-api")
        adapter = _InferenceAPIAdapter(cfg.resolve_spec(), cfg)
        adapter.generate(["ben", "okul", "gitmek"], cfg)

        kwargs = fake_client.chat_completion.call_args.kwargs
        # temperature is clamped to >=0.01 to satisfy stricter providers
        # but should still be near-deterministic.
        self.assertLessEqual(kwargs.get("temperature"), 0.1 + 1e-9)
        self.assertEqual(kwargs.get("max_tokens"), cfg.max_new_tokens)

    def test_chat_completion_propagates_exception(self) -> None:
        # Transport errors propagate (classified upstream), not swallowed.
        _, fake_client = self._install_chat_stub()
        fake_client.chat_completion.side_effect = RuntimeError(
            "Model Qwen/Qwen2.5-7B-Instruct is not supported for task ..."
        )
        from text_processing.grammar import _InferenceAPIAdapter

        cfg = GrammarConfig(use_ml=True, model_key="qwen2.5-7b-api")
        adapter = _InferenceAPIAdapter(cfg.resolve_spec(), cfg)
        with self.assertRaises(RuntimeError):
            adapter.generate(["a", "b", "c"], cfg)

    def test_chat_completion_extracts_object_response(self) -> None:
        # Newer huggingface_hub returns dataclass-like objects, not
        # dicts. Verify the extractor handles attribute access too.
        from types import SimpleNamespace

        result_obj = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="Cevap."))]
        )
        fake_client = MagicMock()
        fake_client.chat_completion.return_value = result_obj
        fake_module = types.ModuleType("huggingface_hub")
        fake_module.InferenceClient = MagicMock(return_value=fake_client)
        sys.modules["huggingface_hub"] = fake_module

        from text_processing.grammar import _InferenceAPIAdapter

        cfg = GrammarConfig(use_ml=True, model_key="qwen2.5-7b-api")
        adapter = _InferenceAPIAdapter(cfg.resolve_spec(), cfg)
        self.assertEqual(adapter.generate(["a", "b"], cfg), "Cevap.")

    def test_non_conversational_dispatches_to_text_generation(self) -> None:
        # Backward-compat: a hypothetical non-conversational API model
        # (registered now or in the future) must keep using
        # text_generation — chat_completion would be wrong for it.
        _, fake_client = self._install_stub("Ben okula gidiyorum.")
        from text_processing.grammar import _InferenceAPIAdapter

        cfg = GrammarConfig(use_ml=True)
        adapter = _InferenceAPIAdapter(self._text_generation_spec(), cfg)
        out = adapter.generate(["ben", "okul", "gitmek"], cfg)

        self.assertEqual(out, "Ben okula gidiyorum.")
        fake_client.text_generation.assert_called_once()
        fake_client.chat_completion.assert_not_called()


class ZeroShotMessagesTests(unittest.TestCase):
    def test_messages_have_system_and_user_roles(self) -> None:
        msgs = build_zero_shot_messages(["ben", "okul", "gitmek"])
        self.assertEqual([m["role"] for m in msgs], ["system", "user"])

    def test_user_message_carries_normalized_words(self) -> None:
        msgs = build_zero_shot_messages(["BEN", "okul", "gitmek."])
        self.assertIn("Kelimeler: ben okul gitmek", msgs[1]["content"])
        self.assertNotIn("gitmek.", msgs[1]["content"])

    def test_system_instruction_forbids_extra_prose(self) -> None:
        # The system instruction needs to suppress chattier behavior or
        # the validator's scaffolding regex will reject the answer.
        sys_msg = build_zero_shot_messages(["a"])[0]["content"]
        self.assertIn("Yalnızca", sys_msg)
        self.assertIn("etiket", sys_msg)

    def test_system_instruction_has_translator_persona_and_rules(self) -> None:
        # TİD-translator persona + semantics-aware rules. The earlier
        # "ASLA mastar" hard rule was replaced — verbs are now allowed
        # to stay in mastar form when the natural translation is a
        # noun phrase (isim tamlaması) or a fiilimsi group.
        sys_msg = build_zero_shot_messages(["a"])[0]["content"]
        self.assertIn("Türk İşaret Dili (TİD)", sys_msg)
        self.assertIn("çevirmen", sys_msg)
        # Semantics: noun-phrase / verbal-noun awareness.
        self.assertIn("tamlama", sys_msg)
        self.assertIn("fiilimsi", sys_msg)
        # Main-clause verb conjugation rule (replaces the absolute
        # no-mastar rule).
        self.assertIn("asıl yargı", sys_msg)
        self.assertIn("çekimle", sys_msg)
        # Case-suffix labels for object marking.
        for case_label in ("yönelme", "bulunma", "belirtme"):
            self.assertIn(case_label, sys_msg, f"missing case label: {case_label}")

    def test_system_instruction_forbids_inventing_verbs(self) -> None:
        # Regression for the "biz mezun olmak" → "Biz mezun olmak
        # istiyoruz." hallucination: Qwen invented ``istemek`` even
        # though nothing in the input meant "to want". The system
        # message now carries an explicit rule against fabricating
        # unprovided verbs (istemek, gerekmek, başlamak, ...) or
        # adding extra intent that wasn't in the signed tokens.
        sys_msg = build_zero_shot_messages(["biz", "mezun", "olmak"])[0]["content"]
        self.assertIn("KESİNLİKLE uydurma", sys_msg)
        self.assertIn("istemek", sys_msg)
        self.assertIn("gerekmek", sys_msg)
        self.assertIn("başlamak", sys_msg)
        # Positive framing — translate only what was given.
        self.assertIn("verilen kelimelerdeki", sys_msg)


class ArbiterTests(unittest.TestCase):
    """Smart-arbiter routing: fast-track simple inputs, arbitrate complex ones.

    Tests use the existing ``mt0-small`` model_key (which has a usable
    MLGrammarCorrector wrapper) and ``patch.object(c._ml, "generate_candidate", ...)``
    to control the ML output — no network calls.
    """

    def _corrector(self) -> GrammarCorrector:
        return GrammarCorrector(GrammarConfig(use_ml=True))

    # ── Fast-track (no ML call) ──

    def test_single_verb_fast_tracks_to_rule_based(self) -> None:
        c = self._corrector()
        with patch.object(c._ml, "generate_candidate") as ml_mock:
            result = c.correct_detailed(["gitmek"])
        ml_mock.assert_not_called()
        self.assertEqual(result.source, "rule-based")
        self.assertIn("fast-track", result.reason)
        self.assertEqual(result.sentence, "Gidiyorum.")

    def test_two_token_pronoun_verb_fast_tracks(self) -> None:
        c = self._corrector()
        with patch.object(c._ml, "generate_candidate") as ml_mock:
            result = c.correct_detailed(["sen", "gelmek"])
        ml_mock.assert_not_called()
        self.assertEqual(result.source, "rule-based")
        self.assertIn("fast-track", result.reason)
        self.assertEqual(result.sentence, "Sen geliyorsun.")

    def test_two_token_noun_verb_fast_tracks(self) -> None:
        c = self._corrector()
        with patch.object(c._ml, "generate_candidate") as ml_mock:
            result = c.correct_detailed(["su", "istemek"])
        ml_mock.assert_not_called()
        self.assertEqual(result.source, "rule-based")
        self.assertIn("fast-track", result.reason)

    def test_three_token_input_does_not_fast_track(self) -> None:
        c = self._corrector()
        with patch.object(
            c._ml, "generate_candidate", return_value=("Sen okula gidiyorsun.", None)
        ) as ml_mock:
            c.correct_detailed(["sen", "okul", "gitmek"])
        ml_mock.assert_called_once()

    # ── Universal fast-track: critical markers ──

    def test_question_particle_fast_tracks(self) -> None:
        c = self._corrector()
        with patch.object(c._ml, "generate_candidate") as ml_mock:
            result = c.correct_detailed(["sen", "gelmek", "mi"])
        ml_mock.assert_not_called()
        self.assertEqual(result.source, "rule-based")
        self.assertIn("interrogative", result.reason)

    def test_negation_fast_tracks(self) -> None:
        c = self._corrector()
        with patch.object(c._ml, "generate_candidate") as ml_mock:
            result = c.correct_detailed(["ben", "gelmek", "değil"])
        ml_mock.assert_not_called()
        self.assertEqual(result.source, "rule-based")
        self.assertIn("negation", result.reason)

    def test_numeric_word_fast_tracks(self) -> None:
        c = self._corrector()
        with patch.object(c._ml, "generate_candidate") as ml_mock:
            result = c.correct_detailed(["ben", "üç", "kitap", "okumak"])
        ml_mock.assert_not_called()
        self.assertEqual(result.source, "rule-based")
        self.assertIn("numeric", result.reason)

    def test_digit_token_fast_tracks(self) -> None:
        c = self._corrector()
        with patch.object(c._ml, "generate_candidate") as ml_mock:
            result = c.correct_detailed(["ben", "5", "kitap", "okumak"])
        ml_mock.assert_not_called()
        self.assertEqual(result.source, "rule-based")
        self.assertIn("numeric", result.reason)

    def test_has_critical_marker_helper(self) -> None:
        from text_processing.grammar import _has_critical_marker

        self.assertTrue(_has_critical_marker(["sen", "gelmek", "mi"]))
        self.assertTrue(_has_critical_marker(["MI"]))  # case-insensitive
        self.assertTrue(_has_critical_marker(["değil"]))
        self.assertTrue(_has_critical_marker(["hayır"]))
        self.assertTrue(_has_critical_marker(["yok"]))
        self.assertTrue(_has_critical_marker(["üç"]))
        self.assertTrue(_has_critical_marker(["42"]))
        self.assertFalse(_has_critical_marker(["sen", "okul", "gitmek"]))
        self.assertFalse(_has_critical_marker([]))
        self.assertFalse(_has_critical_marker(["", "  "]))

    # ── Anti-hallucination guard ──

    def test_anti_hallucination_rejects_unsupported_istiyor(self) -> None:
        # ``Sen gelmek mi`` would fast-track, so use an input without
        # a critical marker but still missing the ``iste`` lemma.
        c = self._corrector()
        with patch.object(
            c._ml, "generate_candidate", return_value=("Sen okula gitmek istiyorsun.", None)
        ):
            result = c.correct_detailed(["sen", "okul", "gitmek"])
        self.assertEqual(result.source, "rule-based")
        self.assertEqual(
            result.rejection_reason,
            "arbiter rejected: hallucinated intent verb",
        )

    def test_anti_hallucination_rejects_unsupported_lazim(self) -> None:
        c = self._corrector()
        with patch.object(
            c._ml, "generate_candidate", return_value=("Sen okula gitmen lazım.", None)
        ):
            result = c.correct_detailed(["sen", "okul", "gitmek"])
        self.assertEqual(result.source, "rule-based")
        self.assertIn("hallucinated intent verb", result.rejection_reason or "")

    def test_anti_hallucination_rejects_unsupported_gerekiyor(self) -> None:
        c = self._corrector()
        with patch.object(
            c._ml, "generate_candidate", return_value=("Sen okula gitmen gerekiyor.", None)
        ):
            result = c.correct_detailed(["sen", "okul", "gitmek"])
        self.assertEqual(result.source, "rule-based")
        self.assertIn("hallucinated intent verb", result.rejection_reason or "")

    def test_anti_hallucination_allows_supported_intent_verb(self) -> None:
        # Input ``istemek`` supplies the ``iste`` lemma, so ``istiyorum``
        # in the ML output is legitimate and the guard stands down.
        c = self._corrector()
        with patch.object(
            c._ml, "generate_candidate", return_value=("Ben su içmek istiyorum.", None)
        ):
            result = c.correct_detailed(["ben", "su", "içmek", "istemek"])
        self.assertEqual(result.sentence, "Ben su içmek istiyorum.")
        self.assertTrue(result.source.startswith("ml:"))
        self.assertIn("passed arbiter", result.reason)

    # ── Arbiter outcomes for complex inputs ──

    def test_arbiter_accepts_clean_ml_output(self) -> None:
        c = self._corrector()
        candidate = "Sen okula gidiyorsun."
        with patch.object(c._ml, "generate_candidate", return_value=(candidate, None)):
            result = c.correct_detailed(["sen", "okul", "gitmek"])
        self.assertEqual(result.sentence, candidate)
        self.assertTrue(result.source.startswith("ml:"))
        self.assertIn("passed arbiter", result.reason)
        self.assertIn("preservation=1.00", result.reason)

    def test_arbiter_rejects_unconjugated_infinitive(self) -> None:
        # Reproduces the demo screenshot bug: Qwen returned the mastar
        # verb. The arbiter must catch it and fall back to rule-based.
        c = self._corrector()
        with patch.object(c._ml, "generate_candidate", return_value=("Sen beni sevmek.", None)):
            result = c.correct_detailed(["sen", "ben", "sevmek"])
        self.assertEqual(result.source, "rule-based")
        self.assertIn("infinitive", result.rejection_reason or "")
        self.assertEqual(result.rejected_candidate, "Sen beni sevmek.")
        self.assertEqual(result.sentence, "Sen beni seviyorsun.")

    def test_arbiter_rejects_severe_hallucination(self) -> None:
        # ML output preserves *zero* input roots — a real
        # hallucination. With the strict (≤3 token) policy the
        # threshold tightens to 1.00, so the rejection message
        # records the strict floor.
        c = self._corrector()
        with patch.object(
            c._ml, "generate_candidate", return_value=("Bugün hava çok güzel.", None)
        ):
            result = c.correct_detailed(["sen", "okul", "gitmek"])
        self.assertEqual(result.source, "rule-based")
        self.assertIn("dropped key roots", result.rejection_reason or "")
        self.assertIn("0.00 < 1.00", result.rejection_reason or "")
        self.assertEqual(result.sentence, "Sen okula gidiyorsun.")

    def test_arbiter_rejects_dropped_root_on_short_inputs(self) -> None:
        # Strict policy: for ≤3-token inputs every lemma must survive.
        # Dropping ``zaman`` (preservation 0.67) is allowed for longer
        # inputs but rejected here.
        c = self._corrector()
        with patch.object(c._ml, "generate_candidate", return_value=("Kafe kapanıyor.", None)):
            result = c.correct_detailed(["kafe", "kapanmak", "zaman"])
        self.assertEqual(result.source, "rule-based")
        self.assertIn("dropped key roots", result.rejection_reason or "")
        self.assertIn("0.67 < 1.00", result.rejection_reason or "")

    def test_arbiter_allows_natural_token_dropping_on_long_inputs(self) -> None:
        # For inputs longer than the strict-threshold window (>3 tokens)
        # the relaxed 0.50 floor lets the model drop a redundant token.
        # 4 input roots, 3 preserved → 0.75 clears the relaxed floor.
        c = self._corrector()
        with patch.object(
            c._ml, "generate_candidate", return_value=("Sen okula gidiyorsun.", None)
        ):
            result = c.correct_detailed(["sen", "okul", "gitmek", "yarın"])
        self.assertEqual(result.sentence, "Sen okula gidiyorsun.")
        self.assertTrue(result.source.startswith("ml:"))
        self.assertIn("passed arbiter", result.reason)
        self.assertIn("0.75", result.reason)

    def test_arbiter_accepts_noun_phrase_output(self) -> None:
        # Same input — but semantically it's an isim tamlaması, not a
        # sentence. With the relaxed prompt the LLM produces
        # "Kafenin kapanma zamanı." (genitive ``kafe-nin`` + verbal
        # noun ``kapan-ma`` + possessive ``zaman-ı``). The arbiter
        # must accept this even though it doesn't end in a conjugated
        # verb:
        #   * last word "zamanı" → not -mek/-mak → no-mastar passes
        #   * all three roots preserved (kafe, kapan, zaman) → 1.0
        #   * length is comparable to the rule-based baseline.
        c = self._corrector()
        with patch.object(
            c._ml, "generate_candidate", return_value=("Kafenin kapanma zamanı.", None)
        ):
            result = c.correct_detailed(["kafe", "kapanmak", "zaman"])
        self.assertEqual(result.sentence, "Kafenin kapanma zamanı.")
        self.assertTrue(result.source.startswith("ml:"))
        self.assertIn("passed arbiter", result.reason)
        self.assertIn("preservation=1.00", result.reason)

    def test_arbiter_accepts_compound_infinitive_output(self) -> None:
        # The other example from the new prompt: a complement-verb
        # compound where the inner verb stays in mastar form by
        # design. "Ben su içmek istiyorum." has "içmek" as a
        # legitimate fiilimsi-style complement; the main verb
        # ("istiyorum") IS conjugated, so the no-mastar guard (which
        # only checks the LAST word) lets this through.
        c = self._corrector()
        with patch.object(
            c._ml, "generate_candidate", return_value=("Ben su içmek istiyorum.", None)
        ):
            result = c.correct_detailed(["ben", "su", "içmek", "istemek"])
        self.assertEqual(result.sentence, "Ben su içmek istiyorum.")
        self.assertTrue(result.source.startswith("ml:"))
        self.assertIn("passed arbiter", result.reason)

    def test_arbiter_rejects_too_short_ml_output(self) -> None:
        # 7-char candidate against a 21-char rule-based baseline. The
        # ``ş`` lets it clear the validator's Turkish-chars check, so
        # the arbiter's length floor is what trips.
        c = self._corrector()
        with patch.object(c._ml, "generate_candidate", return_value=("Şu var.", None)):
            result = c.correct_detailed(["sen", "okul", "gitmek"])
        self.assertEqual(result.source, "rule-based")
        self.assertIn("too short", result.rejection_reason or "")

    def test_arbiter_rejects_too_long_ml_output(self) -> None:
        # Lands in the arbiter zone (> 4× rule-based but < 120-char
        # validator cap), preserves every root, and has no mastar
        # ending — so only the arbiter's length ceiling can flip it.
        c = self._corrector()
        ml_out = (
            "Sen okula çok hızlı bir şekilde sabahın erken saatlerinde "
            "mutlu mutlu mutlu gidiyorsun bugün."
        )
        with patch.object(c._ml, "generate_candidate", return_value=(ml_out, None)):
            result = c.correct_detailed(["sen", "okul", "gitmek"])
        self.assertEqual(result.source, "rule-based")
        self.assertIn("too long", result.rejection_reason or "")

    def test_ml_unavailable_falls_back_with_reason(self) -> None:
        c = self._corrector()
        with patch.object(c._ml, "generate_candidate", return_value=(None, None)):
            result = c.correct_detailed(["sen", "okul", "gitmek"])
        self.assertEqual(result.source, "rule-based")
        self.assertEqual(result.reason, "ml unavailable")
        self.assertIsNone(result.rejected_candidate)
        self.assertEqual(result.sentence, "Sen okula gidiyorsun.")

    def test_ml_disabled_sets_reason(self) -> None:
        c = GrammarCorrector(GrammarConfig(use_ml=False))
        result = c.correct_detailed(["sen", "okul", "gitmek"])
        self.assertEqual(result.source, "rule-based")
        self.assertEqual(result.reason, "ml disabled")

    def test_empty_input_reason(self) -> None:
        c = GrammarCorrector(GrammarConfig(use_ml=False))
        result = c.correct_detailed([])
        self.assertEqual(result.sentence, "")
        self.assertEqual(result.source, "rule-based")
        self.assertEqual(result.reason, "empty input bypass")

    def test_whitespace_only_input_reason(self) -> None:
        c = GrammarCorrector(GrammarConfig(use_ml=False))
        result = c.correct_detailed(["", "   ", "\t"])
        self.assertEqual(result.sentence, "")
        self.assertEqual(result.source, "rule-based")
        self.assertEqual(result.reason, "empty input bypass")

    def test_validator_rejection_recorded_in_reason(self) -> None:
        # Validator catches scaffolding-leak before the arbiter runs.
        c = self._corrector()
        with patch.object(c._ml, "generate_candidate", return_value=("Cümle: foo bar baz", None)):
            result = c.correct_detailed(["ben", "okul", "gitmek"])
        self.assertEqual(result.source, "rule-based")
        self.assertIn("validator rejected", result.reason)
        self.assertIn("scaffolding", result.reason)


class ArbiterHelperTests(unittest.TestCase):
    """Direct tests of the private heuristic helpers."""

    def test_is_simple_input_at_or_below_two(self) -> None:
        from text_processing.grammar import _is_simple_input

        self.assertTrue(_is_simple_input(["gitmek"]))
        self.assertTrue(_is_simple_input(["su", "istemek"]))
        self.assertTrue(_is_simple_input(["sen", "gelmek"]))
        self.assertFalse(_is_simple_input(["sen", "okul", "gitmek"]))
        self.assertFalse(_is_simple_input(["merhaba", "ben", "su", "istemek"]))

    def test_lemma_forms_verb_includes_irregular_stem(self) -> None:
        from text_processing.grammar import _lemma_forms

        forms = _lemma_forms("gitmek")
        self.assertIn("git", forms)
        self.assertIn("gid", forms)

    def test_lemma_forms_verb_includes_narrowing_prefix(self) -> None:
        from text_processing.grammar import _lemma_forms

        # söyle → narrowed to söylü; the truncated form matches both.
        forms = _lemma_forms("söylemek")
        self.assertIn("söyle", forms)
        self.assertIn("söyl", forms)
        forms = _lemma_forms("anlamak")
        self.assertIn("anl", forms)

    def test_lemma_forms_noun_includes_softened_root(self) -> None:
        from text_processing.grammar import _lemma_forms

        forms = _lemma_forms("kitap")
        self.assertIn("kitap", forms)
        self.assertIn("kitab", forms)
        forms = _lemma_forms("ağaç")
        self.assertIn("ağaç", forms)
        self.assertIn("ağac", forms)

    def test_root_preservation_full_match(self) -> None:
        from text_processing.grammar import _root_preservation_score

        score = _root_preservation_score(
            ["sen", "okul", "gitmek"],
            "Sen okula gidiyorsun.",
        )
        self.assertEqual(score, 1.0)

    def test_root_preservation_partial_match(self) -> None:
        from text_processing.grammar import _root_preservation_score

        score = _root_preservation_score(
            ["sen", "okul", "gitmek"],
            "Sen gidiyorsun.",  # "okul" dropped
        )
        self.assertAlmostEqual(score, 2 / 3, places=2)

    def test_root_preservation_handles_softening(self) -> None:
        from text_processing.grammar import _root_preservation_score

        # ML output uses softened forms — the helper must still match.
        score = _root_preservation_score(
            ["ben", "kitap", "yazmak"],
            "Ben kitabı yazıyorum.",
        )
        self.assertEqual(score, 1.0)

    def test_ends_with_unconjugated_verb(self) -> None:
        from text_processing.grammar import _ends_with_unconjugated_verb

        self.assertTrue(_ends_with_unconjugated_verb("Sen beni sevmek."))
        self.assertTrue(_ends_with_unconjugated_verb("Ben kitap okumak"))
        # Mastar in the middle of a compound (legitimate) doesn't count.
        self.assertFalse(_ends_with_unconjugated_verb("Gelmek istiyorum."))
        self.assertFalse(_ends_with_unconjugated_verb("Sen beni seviyorsun."))
        self.assertFalse(_ends_with_unconjugated_verb(""))


class PipelineTests(unittest.TestCase):
    def _make_pipeline(self) -> SignTextPipeline:
        cfg = PipelineConfig(
            buffer=BufferConfig(debounce_seconds=0.5, silence_seconds=2.0),
            grammar=GrammarConfig(use_ml=False),
            tts=TTSConfig(),
            synthesize_audio=False,
        )
        return SignTextPipeline(cfg)

    def test_full_flow_via_correct(self) -> None:
        pipeline = self._make_pipeline()
        result = pipeline.correct(["merhaba", "ben", "su", "istemek"])
        self.assertEqual(result.sentence, "Merhaba, ben su istiyorum.")
        self.assertIsNone(result.audio_path)
        self.assertEqual(result.grammar_source, "rule-based")
        # Arbiter reason propagated all the way to PipelineResult.
        self.assertEqual(result.reason, "ml disabled")

    def test_tts_synthesis_is_called(self) -> None:
        cfg = PipelineConfig(synthesize_audio=True, grammar=GrammarConfig(use_ml=False))
        pipeline = SignTextPipeline(cfg)
        with patch.object(
            pipeline.tts, "synthesize_to_file", return_value=Path("/tmp/fake.mp3")
        ) as m:
            result = pipeline.correct(["merhaba", "ben", "su", "istemek"])
        m.assert_called_once_with("Merhaba, ben su istiyorum.")
        self.assertEqual(result.audio_path, Path("/tmp/fake.mp3"))

    def test_pipeline_records_rejection_metadata(self) -> None:
        cfg = PipelineConfig(synthesize_audio=False, grammar=GrammarConfig(use_ml=True))
        pipeline = SignTextPipeline(cfg)
        with patch.object(pipeline.grammar._ml, "generate_candidate", return_value=("<pad>", None)):
            result = pipeline.correct(["merhaba", "ben", "su", "istemek"])
        self.assertEqual(result.sentence, "Merhaba, ben su istiyorum.")
        self.assertEqual(result.grammar_source, "rule-based")
        self.assertEqual(result.rejected_candidate, "<pad>")


if __name__ == "__main__":
    unittest.main()
