"""Local tests: synthetic data and a small causal oracle; no torch, NPU or downloads."""
import copy
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

# Keep direct unittest execution independent of torch and the training package.
EVAL_DIR = Path(__file__).resolve().parents[3] / "model_eval"
sys.path.insert(0, str(EVAL_DIR))

from eval_mmlu import asset_hashes, run
from mmlu_core import (LABELS, REVISION, SUBJECTS, TOTAL, aggregate, digest, encode_pair,
                       load_data, make_context, protect_output, score_choices, validate_group, write_json)
from run_mmlu_serial import execute, preflight


class ToyTokenizer:
    """Boundary-sensitive tokenizer: separate encoding of ' A' deliberately differs."""
    def __init__(self, multi=False):
        self.multi = multi

    def encode(self, text, add_special_tokens=False):
        assert not add_special_tokens
        if text == "ctx":
            return [1, 2, 3]
        if text.startswith("ctx ") and text[-1] in LABELS:
            token = 5 + LABELS.index(text[-1])
            return [1, 2, 3] + ([4, token] if self.multi else [token])
        if text.startswith(" "):
            return [99]  # Incorrect separate tokenization must never be used.
        raise AssertionError(text)


def oracle(ids, positions, targets):
    # At each position the known causal distribution depends on current AND previous tokens.
    # This catches using prompt logits, missing teacher forcing or an off-by-one causal shift.
    values = []
    for pos, target in zip(positions, targets):
        logits = [((sum(ids[:pos + 1]) + 3 * k) % 11) / 4 for k in range(10)]
        normalizer = math.log(sum(math.exp(x) for x in logits))
        values.append(logits[target] - normalizer)
    return values


def synthetic_data():
    subjects = {}
    for s in SUBJECTS:
        subjects[s] = {}
        for split, count in (("dev", 5), ("test", 1)):
            subjects[s][split] = [{"id": f"{s}/{split}/{i}", "question": "  Q?  ",
                                   "choices": ["x", "y", "z", "w"], "answer": i % 4}
                                  for i in range(count)]
    s = SUBJECTS[-1]
    subjects[s]["test"] = [{**subjects[s]["test"][0], "id": f"{s}/test/{i}"}
                            for i in range(TOTAL - 56)]
    return {"schema_version": 1, "dataset": "cais/mmlu", "revision": REVISION,
            "subjects": subjects, "content_sha256": digest(subjects)}


class ScoringTests(unittest.TestCase):
    def test_joint_boundary_and_space(self):
        token = ToyTokenizer()
        self.assertEqual(encode_pair(token, "ctx ", "A"), ([1, 2, 3], [5]))
        self.assertEqual(encode_pair(token, "ctx", " A"), ([1, 2, 3], [5]))

    def test_reject_cross_boundary_merge(self):
        token = types.SimpleNamespace(encode=lambda s, **kw: [1, 2] if s == "ctx" else [1, 7, 8])
        with self.assertRaisesRegex(ValueError, "boundary_mismatch"):
            encode_pair(token, "ctx", " A")

    def test_single_token_one_forward_matches_reference(self):
        calls = []
        def forward(ids, positions, targets):
            calls.append((ids, positions, targets))
            return oracle(ids, positions, targets)
        result = score_choices(ToyTokenizer(), "ctx", forward)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0], ([1, 2, 3], [2, 2, 2, 2], [5, 6, 7, 8]))
        for label, token in zip(LABELS, range(5, 9)):
            self.assertAlmostEqual(result["scores"][label], oracle([1, 2, 3], [2], [token])[0])

    def test_multitoken_teacher_forcing_shift_and_sum(self):
        calls = []
        def forward(ids, positions, targets):
            calls.append((ids, positions, targets))
            return oracle(ids, positions, targets)
        result = score_choices(ToyTokenizer(multi=True), "ctx", forward)
        self.assertEqual(len(calls), 4)
        self.assertEqual(calls[0], ([1, 2, 3, 4], [2, 3], [4, 5]))
        for label, target in zip(LABELS, range(5, 9)):
            expected = oracle([1, 2, 3], [2], [4])[0] + oracle([1, 2, 3, 4], [3], [target])[0]
            self.assertAlmostEqual(result["scores"][label], expected)

    def test_mixed_answer_lengths_not_length_normalized(self):
        suffixes = {"A": [5], "B": [4, 6], "C": [4, 4, 7], "D": [8]}
        tokenizer = types.SimpleNamespace(encode=lambda text, **kw: [1, 2, 3] if text == "ctx"
                                         else [1, 2, 3] + suffixes[text[-1]])
        # Equal probability per token must favor shorter answers; normalization would tie all four.
        result = score_choices(tokenizer, "ctx", lambda ids, pos, targets: [-2.] * len(targets))
        self.assertEqual(result["scores"], {"A": -2., "B": -4., "C": -6., "D": -2.})
        self.assertEqual(result["prediction"], "A")

    def test_overflow_before_forward_includes_answer(self):
        with self.assertRaisesRegex(ValueError, "overflow"):
            score_choices(ToyTokenizer(multi=True), "ctx", lambda *a: self.fail("forward called"), 4)
        score_choices(ToyTokenizer(multi=True), "ctx", oracle, 5)

    def test_nan_scores_rejected(self):
        with self.assertRaisesRegex(ValueError, "non_finite"):
            score_choices(ToyTokenizer(), "ctx", lambda *a: [math.nan] * 4)

    def test_ties_use_stable_label_order(self):
        self.assertEqual(score_choices(ToyTokenizer(), "ctx", lambda *a: [-1.] * 4)["prediction"], "A")

    def test_prompt_matches_literal_standard_template(self):
        row = {"question": "  Q?\n", "choices": ["x", "y", "z", "w"], "answer": 1}
        context = make_context("abstract_algebra", [row] * 5, row)
        question = "Q?\nA. x\nB. y\nC. z\nD. w\nAnswer:"
        self.assertEqual(context, "The following are multiple choice questions (with answers) about abstract algebra.\n\n"
                         + (question + " B\n\n") * 5 + question)


class AccountingTests(unittest.TestCase):
    def test_weighted_not_macro_and_partial_not_formal(self):
        records = [{"id": "a0", "subject": "a", "status": "ok", "answer": "A", "prediction": "A"}]
        records += [{"id": f"b{i}", "subject": "b", "status": "ok", "answer": "A", "prediction": "B"}
                    for i in range(3)]
        summary = aggregate(records, {"a": 1, "b": 3}, "smoke")
        self.assertEqual(summary["acc_on_evaluated"], .25)
        self.assertIsNone(summary["overall_acc"])
        records[1]["status"] = "error"
        summary = aggregate(records[:-1], {"a": 1, "b": 3}, "full")
        self.assertEqual((summary["correct"], summary["evaluated"], summary["errors"], summary["unattempted"]), (1, 2, 1, 1))
        self.assertFalse(summary["complete"])
        self.assertFalse(summary["formal_score"])

    def test_complete_full_score_and_duplicate_rejected(self):
        data = synthetic_data()
        expected = {s: len(data["subjects"][s]["test"]) for s in SUBJECTS}
        records = [{"id": r["id"], "subject": s, "status": "ok", "answer": "A", "prediction": "A"}
                   for s in SUBJECTS for r in data["subjects"][s]["test"]]
        summary = aggregate(records, expected, "full")
        self.assertTrue(summary["formal_score"])
        self.assertEqual(summary["overall_acc"], 1.)
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            aggregate(records + records[:1], expected, "full")


class PreflightAndSerialTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.assets = self.root / "assets"
        self.assets.mkdir()
        (self.assets / "config.json").write_text("{}")
        (self.assets / "tokenizer.json").write_text("{}")
        self.ckpt = self.root / "iter_0005000"
        self.ckpt.mkdir()
        (self.ckpt / ".metadata").write_text("synthetic")
        (self.ckpt / "__0_0.distcp").write_text("synthetic")
        self.data = self.root / "data.json"
        write_json(self.data, synthetic_data())
        self.group = {"enabled": True, "group_id": "valid", "description": "synthetic fixture",
                      "checkpoint": str(self.ckpt), "config": str(self.assets), "tokenizer": str(self.assets),
                      "expected_step": 5000}
        self.manifest = {"schema_version": 1, "evaluation": {"data": str(self.data), "device": "cpu",
                                                               "seed": 0, "max_length": 4096},
                         "groups": [self.group]}

    def test_placeholder_duplicate_step_and_unavailable(self):
        for change in ({"checkpoint": ""}, {"config": "/path/to/config"}, {"expected_step": 4999},
                       {"tokenizer": str(self.root / "absent")}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                validate_group({**self.group, **change})
        self.manifest["groups"].append(copy.deepcopy(self.group))
        self.assertTrue(all("Duplicate" in " ".join(r["errors"]) for r in preflight(self.manifest)["groups"]))

    def test_data_revision_integrity_counts(self):
        for change in ("revision", "hash", "counts"):
            data = synthetic_data()
            if change == "revision":
                data["revision"] = "main"
            elif change == "hash":
                data["subjects"][SUBJECTS[0]]["test"][0]["question"] = "tampered"
            else:
                data["subjects"][SUBJECTS[0]]["test"].pop()
            write_json(self.data, data)
            with self.subTest(change=change), self.assertRaises(ValueError):
                load_data(self.data)

    def test_different_tokenizers_fail_preflight(self):
        other = self.root / "other_assets"
        other.mkdir()
        (other / "tokenizer.json").write_text('{"different": true}')
        self.manifest["groups"].append({**self.group, "group_id": "other", "tokenizer": str(other)})
        rows = preflight(self.manifest)["groups"]
        self.assertEqual(rows[0]["status"], "ready")
        self.assertEqual(rows[1]["status"], "preflight_failed")

    def test_t5_path_cannot_masquerade_as_t4(self):
        ckpt = self.root / "longTest_05-new_dyn_lr4e-4_hg0" / "iter_0005000"
        ckpt.mkdir(parents=True)
        (ckpt / ".metadata").write_text("fixture")
        (ckpt / "__0_0.distcp").write_text("fixture")
        with self.assertRaisesRegex(ValueError, "T5"):
            validate_group({**self.group, "group_id": "t4_strict_hg0", "checkpoint": str(ckpt)})

    def test_no_enabled_is_not_success_and_preflight_never_launches(self):
        self.group["enabled"] = False
        self.assertEqual(execute(self.manifest, self.root, preflight_only=True,
                                 launch=lambda *a, **k: self.fail("launched")), 1)
        self.group["enabled"] = True
        self.assertEqual(execute(self.manifest, self.root, preflight_only=True,
                                 launch=lambda *a, **k: self.fail("launched")), 0)

    def test_serial_missing_and_failed_continue_actual_child_processes(self):
        self.manifest["groups"] = [{**self.group, "group_id": "missing", "checkpoint": ""},
                                   {**self.group, "group_id": "fail"},
                                   {**self.group, "group_id": "pass"}]
        calls = []
        def child(command, **kwargs):
            gid = command[command.index("--group-id") + 1]
            out = command[command.index("--out") + 1]
            calls.append(gid)
            # Real, tiny subprocess exit and output handling; no model/torch imports.
            code = ("import pathlib,json,sys; p=pathlib.Path(sys.argv[1]); p.mkdir(); "
                    "(p/'summary.json').write_text(json.dumps({'group_id':sys.argv[2],"
                    "'status':'smoke_complete','complete':True,'formal_score':False,'mode':'smoke'})); "
                    "(p/'run.json').write_text(sys.argv[3],encoding='utf-8'); "
                    "sys.exit(7 if sys.argv[2]=='fail' else 0)")
            metadata = json.dumps({"config_assets": asset_hashes(self.assets), "tokenizer_assets": asset_hashes(self.assets),
                                   "data_content_sha256": load_data(self.data)["content_sha256"]})
            return subprocess.run([sys.executable, "-c", code, out, gid, metadata], **kwargs)
        self.assertEqual(execute(self.manifest, self.root, smoke=True, launch=child), 1)
        self.assertEqual(calls, ["fail", "pass"])
        summary = json.loads((self.root / "serial_summary.json").read_text(encoding="utf-8"))
        self.assertEqual([r["status"] for r in summary["groups"]], ["preflight_failed", "failed", "smoke_complete"])
        self.assertFalse(summary["all_enabled_succeeded"])

    def test_missing_child_summary_cannot_be_success(self):
        self.assertEqual(execute(self.manifest, self.root, launch=lambda *a, **k: types.SimpleNamespace(returncode=0)), 1)

    def test_output_cannot_write_into_checkpoint(self):
        with self.assertRaisesRegex(ValueError, "read-only"):
            protect_output(self.ckpt / "results", [str(self.ckpt)])

    def test_single_evaluator_smoke_and_overflow_results(self):
        tokenizer = types.SimpleNamespace(encode=lambda text, **kw: list(text.encode("utf-8")))
        for max_length, expected_code in ((4096, 0), (1, 1)):
            out = self.root / f"single_{max_length}"
            args = types.SimpleNamespace(out=str(out), ckpt=str(self.ckpt), config=str(self.assets),
                                         tokenizer=str(self.assets), data=str(self.data), seed=0, device="cpu",
                                         expected_step=5000, description="test", group_id="single",
                                         max_length=max_length, smoke=True)
            # The fixture must have two smoke rows per subject while keeping 14042 full rows.
            data = load_data(self.data)
            for s in ("abstract_algebra", "anatomy"):
                if len(data["subjects"][s]["test"]) == 1:
                    data["subjects"][s]["test"].append({**data["subjects"][s]["test"][0], "id": f"{s}/test/1"})
                    data["subjects"][SUBJECTS[-1]]["test"].pop()
            data["content_sha256"] = digest(data["subjects"])
            write_json(self.data, data)
            with patch("eval_mmlu.load_model", return_value=(object(), tokenizer, {})), \
                    patch("eval_mmlu.torch_forward", return_value=lambda ids, pos, target: [-1.] * len(target)), \
                    patch("eval_mmlu.importlib.metadata.version", return_value="local-fixture"):
                self.assertEqual(run(args), expected_code)
            summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["total"], 4)
            self.assertEqual(summary["errors"], 0 if expected_code == 0 else 4)
            self.assertFalse(summary["formal_score"])
            self.assertIsNone(summary["overall_acc"])
            self.assertEqual(len((out / "questions.jsonl").read_text(encoding="utf-8").splitlines()), 4)
            with self.assertRaises(FileExistsError):
                run(args)

    def test_eight_group_mapping(self):
        template = json.loads((EVAL_DIR / "mmlu_groups.template.json").read_text(encoding="utf-8"))
        groups = {g["group_id"]: g for g in template["groups"]}
        self.assertEqual(set(groups), {"adamw_lr4e4", "t2_old_dynamic_lr4e4", "t3_old_dynamic_lr2e3",
                                       "fix_rms_old_lr4e4", "t4_strict_hg0", "t6_strict_hg1",
                                       "t7_strict_hg2", "t9_strict_hg4"})
        self.assertTrue(all(g["expected_step"] == 5000 for g in groups.values()))


if __name__ == "__main__":
    unittest.main()
