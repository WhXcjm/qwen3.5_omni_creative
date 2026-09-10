#!/usr/bin/env python
"""Single DCP checkpoint MMLU evaluation; uses the validated completion loader unchanged."""
import argparse
import gc
import importlib.metadata
import json
import os
import sys
import time
import traceback
from pathlib import Path

from mmlu_core import (LABELS, SUBJECTS, aggregate, checked_path, digest, file_hash,
                       load_data, make_context, protect_output, protocol, score_choices, validate_group, write_json)


def torch_forward(model, device):
    import torch

    def forward(ids, positions, targets):
        # All requested positions are at the end of the causal input. Avoid L*vocab logits.
        start = min(positions)
        with torch.inference_mode():
            output = model(input_ids=torch.tensor([ids], dtype=torch.long, device=device),
                           use_cache=False, output_router_logits=False, logits_to_keep=len(ids) - start)
            logprobs = torch.log_softmax(output.logits[0].float(), dim=-1)
            indices = torch.tensor([x - start for x in positions], device=device)
            tokens = torch.tensor(targets, device=device)
            return logprobs[indices, tokens].cpu().tolist()
    return forward


def asset_hashes(root):
    # Include config/custom tokenizer code and vocabulary, never model weights.
    return {p.relative_to(root).as_posix(): file_hash(p) for p in sorted(Path(root).rglob("*"))
            if p.is_file() and p.suffix in (".json", ".model", ".txt", ".py", ".jinja", ".tiktoken")}


def checkpoint_identity(root):
    # Read-only metadata hash + shard inventory for audit, NOT proof of equal weight bytes.
    root = Path(root)
    return {"resolved_path": str(root.resolve()), "metadata_sha256": file_hash(root / ".metadata"),
            "shards": [{"name": p.name, "bytes": p.stat().st_size, "mtime_ns": p.stat().st_mtime_ns}
                       for p in sorted(root.glob("*.distcp"))]}


def load_model(paths, device, seed):
    # Enforce offline HF calls before importing any HF/model modules. Paths were verified local.
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    repo = str(Path(__file__).resolve().parents[1])
    sys.path.insert(0, repo)
    os.chdir(repo)
    from eval_completion import (_setup_single_process_dist, build_model, load_text_weights,
                                 load_into_model, _move_to_device)
    import torch
    from transformers import AutoTokenizer

    torch.manual_seed(seed)
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "16")))
    _setup_single_process_dist(device)
    tokenizer = AutoTokenizer.from_pretrained(paths["tokenizer"], trust_remote_code=True,
                                              local_files_only=True)
    model, _ = build_model(paths["config"], "eager", "bfloat16", device)
    weights = load_text_weights(paths["checkpoint"])
    report = load_into_model(model, weights, "bfloat16", device)
    del weights
    gc.collect()
    missing_text = [k for k in report["missing"] if k.startswith(("model.language_model.", "lm_head."))]
    if missing_text:
        raise RuntimeError(f"Missing text weights: {missing_text[:20]}")
    _move_to_device(model, device)
    model.eval()
    return model, tokenizer, report


def run(args):
    out = checked_path(args.out, "output")
    protect_output(out, [args.ckpt, args.config, args.tokenizer])
    out.mkdir(parents=True, exist_ok=False)  # Never resume or treat partial output as complete.
    started = time.time()
    records, expected = [], {}
    mode = "smoke" if args.smoke else "full"
    shard = getattr(args, "shard", None)
    num_shards = getattr(args, "num_shards", 1)
    meta = {"group_id": args.group_id, "expected_step": args.expected_step,
            "protocol": protocol(args.seed, args.max_length, mode), "device": args.device,
            "formal_score": False, "status": "running"}
    write_json(out / "run.json", meta)
    try:
        paths = validate_group({"group_id": args.group_id, "description": args.description,
                                "expected_step": args.expected_step, "checkpoint": args.ckpt,
                                "config": args.config, "tokenizer": args.tokenizer})
        data_path = checked_path(args.data, "file")
        data = load_data(data_path)
        selected = ("abstract_algebra", "anatomy") if args.smoke else SUBJECTS
        rows = {s: data["subjects"][s]["test"][:2] if args.smoke else data["subjects"][s]["test"]
                for s in selected}
        if shard is not None and num_shards > 1:
            # Deterministic round-robin over the subject-major ordering: each shard scores
            # every num_shards-th question and writes only its own slice.
            kept = {}
            for i, (s, r) in enumerate((s, r) for s in selected for r in rows[s]):
                if i % num_shards == shard:
                    kept.setdefault(s, []).append(r)
            rows = kept
        expected = {s: len(r) for s, r in rows.items()}
        identity = checkpoint_identity(paths["checkpoint"])
        meta.update({"paths": paths, "description": args.description, "checkpoint_identity": identity,
                     "data_path": str(data_path), "data_content_sha256": data["content_sha256"],
                     "data_file_sha256": file_hash(data_path), "data_provenance": data.get("provenance"),
                     "config_assets": asset_hashes(paths["config"]),
                     "tokenizer_assets": asset_hashes(paths["tokenizer"]),
                     "code_sha256": {p.name: file_hash(p) for p in Path(__file__).parent.glob("*.py")},
                     "fewshot_ids": {s: [d["id"] for d in data["subjects"][s]["dev"]] for s in selected},
                     "selected_test_ids": {s: [r["id"] for r in rows[s]] for s in rows}})
        meta["protocol_sha256"] = digest({"protocol": meta["protocol"], "data": data["content_sha256"],
                                           "config": meta["config_assets"], "tokenizer": meta["tokenizer_assets"]})
        write_json(out / "run.json", meta)
        model, tokenizer, report = load_model(paths, args.device, args.seed)
        meta["load_report"] = report
        meta["versions"] = {p: importlib.metadata.version(p) for p in ("torch", "transformers", "accelerate")}
        if args.device == "npu":
            meta["versions"]["torch-npu"] = importlib.metadata.version("torch-npu")
        meta["tokenizer_class"] = type(tokenizer).__name__
        write_json(out / "run.json", meta)
        forward = torch_forward(model, args.device)
        with (out / "questions.jsonl").open("x", encoding="utf-8") as stream:
            for subject in rows:
                for row in rows[subject]:
                    context = make_context(subject, data["subjects"][subject]["dev"], row)
                    result = {"id": row["id"], "subject": subject, "answer": LABELS[row["answer"]],
                              "status": "ok", "context_sha256": digest(context),
                              "scores": None, "prediction": None}
                    try:
                        result.update(score_choices(tokenizer, context, forward, args.max_length))
                    except ValueError as error:
                        # Protocol/input errors are per-question; device/model failures abort this group.
                        result.update(status="error", error=str(error))
                    except Exception as error:
                        result.update(status="error", error=f"{type(error).__name__}: {error}")
                        records.append(result)
                        stream.write(json.dumps(result, ensure_ascii=False) + "\n")
                        stream.flush()
                        raise
                    records.append(result)
                    stream.write(json.dumps(result, ensure_ascii=False, allow_nan=False) + "\n")
                    stream.flush()
                print(f"[{args.group_id}] {subject}: {len(records)}/{sum(expected.values())}", flush=True)
        if (checkpoint_identity(paths["checkpoint"]) != identity
                or asset_hashes(paths["config"]) != meta["config_assets"]
                or asset_hashes(paths["tokenizer"]) != meta["tokenizer_assets"]
                or file_hash(data_path) != meta["data_file_sha256"]):
            raise RuntimeError("Checkpoint/assets/data changed during evaluation; score invalid")
        summary = aggregate(records, expected, mode)
        if shard is not None and not records:
            summary["complete"] = True  # Empty shard slice: vacuous completion, no formal score.
        status = "smoke_complete" if args.smoke and summary["complete"] else (
            "complete" if summary["complete"] else "incomplete")
    except Exception as error:
        traceback.print_exc()
        summary = aggregate(records, expected, mode)
        summary.update(complete=False, formal_score=False, overall_acc=None,
                       fatal_error=f"{type(error).__name__}: {error}")
        status = "failed"
    summary.update(group_id=args.group_id, mode=mode, status=status, requested_total=4 if args.smoke else 14042,
                   wall_seconds=round(time.time() - started, 3),
                   protocol_sha256=meta.get("protocol_sha256"))
    meta.update(status=status, formal_score=summary["formal_score"])
    write_json(out / "run.json", meta)
    write_json(out / "summary.json", summary)
    print(f"[{args.group_id}] {status}: {summary['evaluated']}/{summary['total']}; "
          f"errors={summary['errors']}; overall_acc={summary['overall_acc']}", flush=True)
    return 0 if status in ("complete", "smoke_complete") else 1


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for flag in ("ckpt", "config", "tokenizer", "data", "out"):
        p.add_argument("--" + flag, required=True)
    p.add_argument("--group-id", default="single_checkpoint")
    p.add_argument("--description", default="Single checkpoint MMLU evaluation")
    p.add_argument("--expected-step", type=int, default=5000)
    p.add_argument("--device", choices=["npu", "cpu"], default="npu")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-length", type=int, default=4096)
    p.add_argument("--smoke", action="store_true", help="Fixed 4 questions, never a formal score")
    p.add_argument("--shard", type=int, default=None, help="0-based shard index (round-robin over questions)")
    p.add_argument("--num-shards", type=int, default=1,
                   help="Total shards; a shard scores every num-shards-th question")
    args = p.parse_args()
    if args.max_length < 1:
        p.error("--max-length must be positive")
    if (args.shard is None) != (args.num_shards == 1):
        p.error("--shard and --num-shards must be provided together (or use defaults)")
    if args.shard is not None and not (0 <= args.shard < args.num_shards):
        p.error("--shard must satisfy 0 <= shard < num-shards")
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
