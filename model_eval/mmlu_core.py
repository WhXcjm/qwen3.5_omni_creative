"""Dependency-free MMLU protocol, token scoring plans, data checks and accounting.

Template reference: EleutherAI/lm-evaluation-harness v0.4.11,
lm_eval/tasks/mmlu/default/_default_template_yaml and lm_eval/api/model.py::_encode_pair.
"""
import hashlib
import json
import math
import re
from pathlib import Path

REVISION = "c30699e8356da336a370243923dbaf21066bb9fe"
LABELS = "ABCD"
TOTAL = 14042
SUBJECTS = tuple(sorted("""
abstract_algebra anatomy astronomy business_ethics clinical_knowledge college_biology
college_chemistry college_computer_science college_mathematics college_medicine
college_physics computer_security conceptual_physics econometrics electrical_engineering
elementary_mathematics formal_logic global_facts high_school_biology high_school_chemistry
high_school_computer_science high_school_european_history high_school_geography
high_school_government_and_politics high_school_macroeconomics high_school_mathematics
high_school_microeconomics high_school_physics high_school_psychology high_school_statistics
high_school_us_history high_school_world_history human_aging human_sexuality international_law
jurisprudence logical_fallacies machine_learning management marketing medical_genetics
miscellaneous moral_disputes moral_scenarios nutrition philosophy prehistory
professional_accounting professional_law professional_medicine professional_psychology
public_relations security_studies sociology us_foreign_policy virology world_religions
""".split()))


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, value):
    """Atomic publication; callers must own a fresh run directory."""
    path = Path(path)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                    encoding="utf-8")
    temp.replace(path)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def validate_data(data):
    if data.get("dataset") != "cais/mmlu" or data.get("revision") != REVISION:
        raise ValueError("Only the pinned cais/mmlu revision is accepted")
    if data.get("schema_version") != 1 or set(data.get("subjects", {})) != set(SUBJECTS):
        raise ValueError("Expected schema_version=1 and exactly 57 standard MMLU subjects")
    total = 0
    for subject in SUBJECTS:
        splits = data["subjects"][subject]
        if len(splits.get("dev", [])) != 5 or not splits.get("test"):
            raise ValueError(f"{subject}: expected five dev rows and nonempty test")
        for split in ("dev", "test"):
            for i, row in enumerate(splits[split]):
                if row.get("id") != f"{subject}/{split}/{i}":
                    raise ValueError(f"{subject}/{split}: IDs/order changed at row {i}")
                if (not isinstance(row.get("question"), str) or not row["question"].strip()
                        or not isinstance(row.get("choices"), list) or len(row["choices"]) != 4
                        or not all(isinstance(c, str) for c in row["choices"])
                        or type(row.get("answer")) is not int or row["answer"] not in range(4)):
                    raise ValueError(f"Invalid row: {row.get('id')}")
        total += len(splits["test"])
    if total != TOTAL:
        raise ValueError(f"Expected {TOTAL} test rows, got {total}")
    if data.get("content_sha256") != digest(data["subjects"]):
        raise ValueError("Data content hash mismatch")


def load_data(path):
    data = read_json(path)
    validate_data(data)
    return data


def question_text(row):
    return row["question"].strip() + "\n" + "\n".join(
        f"{label}. {choice}" for label, choice in zip(LABELS, row["choices"])
    ) + "\nAnswer:"


def make_context(subject, dev, row):
    if len(dev) != 5:
        raise ValueError("Exactly five dev examples required; no fewshot reduction")
    description = ("The following are multiple choice questions (with answers) about "
                   + subject.replace("_", " ") + ".\n\n")
    examples = "".join(question_text(d) + " " + LABELS[d["answer"]] + "\n\n" for d in dev)
    return description + examples + question_text(row)


def encode_pair(tokenizer, context, continuation):
    """Joint encoding with a verified token boundary; never score prompt tokens.

    As in harness _encode_pair, move context's trailing whitespace to continuation.
    Unlike its length-only slicing, reject a tokenizer that merges across the
    remaining context boundary: scoring that merged token would score prompt text.
    Multi-token continuations with a valid boundary are supported without truncation.
    """
    n = len(context) - len(context.rstrip())
    if n:
        context, continuation = context[:-n], context[-n:] + continuation
    prefix = list(tokenizer.encode(context, add_special_tokens=False))
    whole = list(tokenizer.encode(context + continuation, add_special_tokens=False))
    if not prefix or whole[:len(prefix)] != prefix:
        raise ValueError("context_continuation_token_boundary_mismatch")
    suffix = whole[len(prefix):]
    if not suffix:
        raise ValueError("empty_continuation_tokens")
    return prefix, suffix


def score_choices(tokenizer, context, forward_logprobs, max_length=4096):
    """forward_logprobs(ids, positions, targets) scores target after each position.

    Single token: all four candidates use the same final context logits.
    Multi token: teacher force context+answer[:-1], score positions C-1 .. C+K-2.
    No normalization; no cache; max length includes the complete answer suffix.
    """
    pairs = [encode_pair(tokenizer, context, " " + label) for label in LABELS]
    if any(len(c) + len(a) > max_length for c, a in pairs):
        raise ValueError(f"context_overflow: full candidate exceeds {max_length} tokens")
    prefix = pairs[0][0]
    fast = all(c == prefix and len(a) == 1 for c, a in pairs)
    if fast:
        scores = forward_logprobs(prefix, [len(prefix) - 1] * 4,
                                  [a[0] for _, a in pairs])
    else:
        scores = []
        for c, a in pairs:
            seq = (c + a)[:-1]
            token_scores = forward_logprobs(seq, list(range(len(c) - 1, len(seq))), a)
            if len(token_scores) != len(a):
                raise ValueError("forward returned wrong continuation length")
            scores.append(sum(token_scores))
    if len(scores) != 4 or not all(math.isfinite(x) for x in scores):
        raise ValueError("non_finite_or_invalid_scores")
    return {"scores": dict(zip(LABELS, scores)), "prediction": LABELS[max(range(4), key=scores.__getitem__)],
            "scoring_path": "shared_single_token" if fast else "multi_token",
            "context_tokens": len(prefix), "answer_token_ids": dict(zip(LABELS, [a for _, a in pairs]))}


def aggregate(records, expected, mode):
    subjects = {}
    seen = set()
    for row in records:
        subject = row["subject"]
        if row["id"] in seen or subject not in expected:
            raise ValueError("Duplicate result ID or unexpected subject")
        seen.add(row["id"])
        count = subjects.setdefault(subject, {"correct": 0, "evaluated": 0, "errors": 0})
        if row["status"] == "ok":
            count["evaluated"] += 1
            count["correct"] += int(row["prediction"] == row["answer"])
        else:
            count["errors"] += 1
    for subject, total in expected.items():
        count = subjects.setdefault(subject, {"correct": 0, "evaluated": 0, "errors": 0})
        count["total"] = total
        count["unattempted"] = total - count["evaluated"] - count["errors"]
        if count["unattempted"] < 0:
            raise ValueError("Too many result rows")
        count["acc_on_evaluated"] = count["correct"] / count["evaluated"] if count["evaluated"] else None
    totals = {k: sum(c[k] for c in subjects.values())
              for k in ("correct", "evaluated", "errors", "total", "unattempted")}
    complete = totals["evaluated"] == totals["total"] and totals["total"] > 0
    formal = complete and mode == "full" and totals["total"] == TOTAL and len(expected) == 57
    return {**totals, "subjects": subjects, "complete": complete, "formal_score": formal,
            "overall_acc": totals["correct"] / totals["total"] if formal else None,
            "acc_on_evaluated": totals["correct"] / totals["evaluated"] if totals["evaluated"] else None}


def protocol(seed=0, max_length=4096, mode="full"):
    return {"protocol_version": 1, "dataset": "cais/mmlu", "revision": REVISION,
            "harness_reference": "v0.4.11/default/mmlu (template metadata 1.0, group 2)",
            "fewshot": 5, "sampler": "dev original order first_n", "seed": seed,
            "target_delimiter": " ", "fewshot_delimiter": "\n\n", "labels": list(LABELS),
            "chat_template": False, "add_special_tokens": False, "length_normalization": False,
            "score": "sum conditional answer-suffix log-probabilities; FP32 log_softmax",
            "boundary_policy": "joint encoding with exact context prefix; reject cross-boundary merges",
            "max_length": max_length, "overflow": "error; no truncation or fewshot reduction",
            "batch_size": 1, "use_cache": False, "dtype": "bfloat16", "attn": "eager",
            "mode": mode, "smoke_selection": "first 2 test rows of abstract_algebra and anatomy"}


def checked_path(value, kind="dir"):
    if not isinstance(value, str) or not value.strip() or re.search(
            r"[<>$]|\b(TODO|TBD|CHANGE_ME|REPLACE_ME)\b|/path/to/", value, re.I):
        raise ValueError(f"Missing/placeholder {kind} path: {value!r}")
    path = Path(value).expanduser()
    if not path.is_absolute():
        raise ValueError(f"Use an absolute path: {path}")
    if (kind == "dir" and not path.is_dir()) or (kind == "file" and not path.is_file()):
        raise ValueError(f"Unavailable {kind}: {path}")
    return path.resolve()


def protect_output(output, inputs):
    """Do not write results under any read-only model/checkpoint asset directory."""
    for value in inputs:
        if isinstance(value, str) and value.strip():
            source = Path(value).expanduser().resolve()
            if output == source or source in output.parents:
                raise ValueError(f"Output must be outside read-only input directory: {source}")


def validate_group(group):
    if not re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,79}", group.get("group_id", "")):
        raise ValueError("Invalid group_id (lowercase letters, digits, underscore, hyphen)")
    if not isinstance(group.get("description"), str) or not group["description"].strip():
        raise ValueError("Missing experiment description")
    if type(group.get("expected_step")) is not int or group["expected_step"] < 0:
        raise ValueError("expected_step must be a nonnegative integer")
    paths = {key: checked_path(group.get(key)) for key in ("checkpoint", "config", "tokenizer")}
    ckpt = paths["checkpoint"]
    if not (ckpt / ".metadata").is_file() or not any(ckpt.glob("*.distcp")):
        raise ValueError("Checkpoint must contain .metadata and DCP shards")
    match = re.fullmatch(r"iter_(\d+)", ckpt.name)
    if not match or int(match[1]) != group["expected_step"]:
        raise ValueError("Checkpoint directory must be iter_<expected_step>; verify actual step separately")
    if not (paths["config"] / "config.json").is_file():
        raise ValueError("config directory lacks config.json")
    if not any((paths["tokenizer"] / name).is_file()
               for name in ("tokenizer.json", "tokenizer.model", "vocab.json")):
        raise ValueError("Tokenizer directory lacks local tokenizer assets")
    if group.get("group_id") == "t4_strict_hg0" and "longtest_05-new_dyn_lr4e-4_hg0" in str(ckpt).lower():
        raise ValueError("Known T5 a/b->AdamW path cannot be used as T4")
    return {key: str(path) for key, path in paths.items()}
