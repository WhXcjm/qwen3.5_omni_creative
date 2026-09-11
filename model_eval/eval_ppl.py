#!/usr/bin/env python
"""Held-out perplexity (PPL) for a single DCP checkpoint; forward-only, packed windows.

Streams the held-out JSON-array corpus, tokenizes with the model's own tokenizer,
packs it into non-overlapping 4096-token windows (the training packing convention),
and computes the mean next-token NLL. PPL = exp(mean NLL).

Model/weights loading is reused verbatim from eval_completion via eval_mmlu.load_model
(read-only DCP, unsharded single-process reconstruction). No backward, no cache.
"""
import argparse
import json
import math
import os
import sys
import time
import traceback
from pathlib import Path

from mmlu_core import checked_path, file_hash, protect_output, validate_group, write_json
from eval_mmlu import asset_hashes, checkpoint_identity, load_model

SEQ_LEN = 4096          # training cutoff_len
GLOBAL_BATCH = 256      # training GBS; 1 iter == GLOBAL_BATCH * SEQ_LEN tokens
SMOKE_WINDOWS = 8       # windows (~32K tokens) for the smoke test


def iter_texts(path):
    """Yield each object's "text" from a pretty-printed JSON array, streaming (no full load)."""
    decoder = json.JSONDecoder()
    with open(path, encoding="utf-8") as stream:
        if stream.read(1).lstrip() != "[":
            raise ValueError("Held-out file must be a JSON array")
        buf = ""
        for line in stream:
            if not line.strip():
                continue
            buf += line
            candidate = buf.lstrip()
            if candidate.startswith(","):
                candidate = candidate[1:].lstrip()
            try:
                obj, _ = decoder.raw_decode(candidate)
            except json.JSONDecodeError:
                continue
            yield obj.get("text", "")
            buf = ""


def run(args):
    out = checked_path(args.out, "output")
    protect_output(out, [args.ckpt, args.config, args.tokenizer])
    out.mkdir(parents=True, exist_ok=False)
    started = time.time()
    mode = "smoke" if args.smoke else "full"
    meta = {"group_id": args.group_id, "expected_step": args.expected_step, "mode": mode,
            "device": args.device, "seed": args.seed, "shard": args.shard,
            "num_shards": args.num_shards, "fast": args.fast, "status": "running"}
    write_json(out / "run.json", meta)
    try:
        paths = validate_group({"group_id": args.group_id, "description": args.description,
                                "expected_step": args.expected_step, "checkpoint": args.ckpt,
                                "config": args.config, "tokenizer": args.tokenizer})
        data_path = checked_path(args.data, "file")
        total_windows = args.max_iters * GLOBAL_BATCH
        target = SMOKE_WINDOWS if args.smoke else total_windows
        per_shard = max(1, (target + args.num_shards - 1) // args.num_shards)
        meta.update({
            "paths": paths, "description": args.description,
            "checkpoint_identity": checkpoint_identity(paths["checkpoint"]),
            "data_path": str(data_path), "data_file_sha256": file_hash(data_path),
            "config_assets": asset_hashes(paths["config"]),
            "tokenizer_assets": asset_hashes(paths["tokenizer"]),
            "code_sha256": {p.name: file_hash(p) for p in Path(__file__).parent.glob("*.py")},
            "packing": True, "seq_len": SEQ_LEN, "max_iters": args.max_iters,
            "total_windows": total_windows, "target_windows_per_shard": per_shard,
        })
        write_json(out / "run.json", meta)

        # load_model imports eval_completion first (sets OMP/MKL thread caps, then torch),
        # so torch must be imported here — after the loader — never at module top.
        model, tokenizer, report = load_model(paths, args.device, args.seed,
                                              use_triton_gdn=args.fast,
                                              use_grouped_expert_matmul=args.fast)
        import torch
        import torch.nn.functional as F
        meta["load_report"] = report
        write_json(out / "run.json", meta)

        n_tokens, sum_nll, n_windows = 0, 0.0, 0
        buf = []
        last_log = time.time()
        forward_started = time.time()  # after model load; progress rate excludes load time
        log_interval = 30.0  # seconds between progress lines (shard log)
        for doc_idx, text in enumerate(iter_texts(data_path)):
            if args.shard is not None and doc_idx % args.num_shards != args.shard:
                continue
            buf.extend(tokenizer.encode(text, add_special_tokens=False))
            while len(buf) > SEQ_LEN:
                win = buf[:SEQ_LEN + 1]
                ids = torch.tensor([win[:SEQ_LEN]], dtype=torch.long, device=args.device)
                labels = torch.tensor(win[1:], dtype=torch.long, device=args.device)
                with torch.inference_mode():
                    logits = model(input_ids=ids, use_cache=False,
                                   output_router_logits=False).logits
                sum_nll += F.cross_entropy(logits[0].float(), labels, reduction="sum").item()
                del logits
                n_tokens += SEQ_LEN
                n_windows += 1
                del buf[:SEQ_LEN]
                now = time.time()
                if now - last_log >= log_interval:
                    rate = n_windows * SEQ_LEN / max(now - forward_started, 1e-6)
                    print(f"[{args.group_id}] shard={args.shard} windows={n_windows}/{per_shard} "
                          f"tokens={n_tokens} {rate:.0f} tok/s elapsed={now - forward_started:.0f}s",
                          flush=True)
                    last_log = now
                if n_windows >= per_shard:
                    break
            if n_windows >= per_shard:
                break
        if n_tokens == 0:
            raise RuntimeError("No tokens processed (empty shard slice)")
        mean_nll = sum_nll / n_tokens
        summary = {
            "group_id": args.group_id, "mode": mode, "status": "complete",
            "n_tokens": n_tokens, "n_windows": n_windows, "sum_nll": sum_nll,
            "mean_nll": mean_nll, "ppl": math.exp(mean_nll),
            "packing": True, "seq_len": SEQ_LEN, "max_iters": args.max_iters,
            "shard": args.shard, "num_shards": args.num_shards, "fast": args.fast,
            "wall_seconds": round(time.time() - started, 3),
        }
    except Exception as error:
        traceback.print_exc()
        summary = {"group_id": args.group_id, "mode": mode, "status": "failed",
                   "fatal_error": f"{type(error).__name__}: {error}"}
    write_json(out / "summary.json", summary)
    print(f"[{args.group_id}] shard={args.shard} {summary['status']}: "
          f"tokens={summary.get('n_tokens', 0)} mean_nll={summary.get('mean_nll')} "
          f"ppl={summary.get('ppl')}", flush=True)
    return 0 if summary.get("status") == "complete" else 1


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for flag in ("ckpt", "config", "tokenizer", "data", "out"):
        p.add_argument("--" + flag, required=True)
    p.add_argument("--group-id", default="single_checkpoint")
    p.add_argument("--description", default="Held-out perplexity evaluation")
    p.add_argument("--expected-step", type=int, default=5000)
    p.add_argument("--device", choices=["npu", "cpu"], default="npu")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-iters", type=int, default=100)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--fast", action="store_true",
                   help="Enable Triton GDN + grouped-expert matmul (training path); "
                        "valid for full-length prefill PPL, not decode/completion.")
    p.add_argument("--shard", type=int, default=None)
    p.add_argument("--num-shards", type=int, default=1)
    args = p.parse_args()
    if args.max_iters < 1:
        p.error("--max-iters must be positive")
    if (args.shard is None) != (args.num_shards == 1):
        p.error("--shard and --num-shards must be provided together")
    if args.shard is not None and not (0 <= args.shard < args.num_shards):
        p.error("--shard must satisfy 0 <= shard < num-shards")
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
