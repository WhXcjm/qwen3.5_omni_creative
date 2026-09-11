#!/usr/bin/env python
"""Preflight and evaluate enabled PPL groups, optionally sharding each across NPUs.

Mirrors run_mmlu_serial.py's orchestration but for held-out perplexity: each group is
split across N child eval_ppl.py processes (document round-robin), each pinned to a
distinct NPU; their (sum_nll, n_tokens) are merged into one formal PPL.
"""
import argparse
import math
from collections import Counter
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import uuid

from mmlu_core import checked_path, digest, file_hash, protect_output, read_json, validate_group, write_json
from eval_mmlu import asset_hashes


def preflight(manifest):
    groups = manifest.get("groups")
    if manifest.get("schema_version") != 1 or not isinstance(groups, list) or not groups:
        raise ValueError("Expected schema_version=1 and nonempty groups")
    shared = manifest.get("evaluation", {})
    data_path = checked_path(shared.get("data"), "file")
    data_sha = file_hash(data_path)
    max_iters, seq_len, device, seed = (shared.get("max_iters"), shared.get("seq_len", 4096),
                                        shared.get("device"), shared.get("seed"))
    if type(max_iters) is not int or max_iters < 1:
        raise ValueError("evaluation.max_iters must be a positive integer")
    if type(seq_len) is not int or seq_len < 1:
        raise ValueError("evaluation.seq_len must be a positive integer")
    if device not in ("npu", "cpu") or type(seed) is not int:
        raise ValueError("evaluation requires device npu/cpu and integer seed")
    ids = Counter(str(g.get("group_id")) for g in groups)
    report = []
    common_assets = None
    for index, group in enumerate(groups):
        row = {"index": index, "group_id": group.get("group_id"), "enabled": group.get("enabled"),
               "description": group.get("description"), "errors": []}
        if type(group.get("enabled")) is not bool:
            row["errors"].append("enabled must be an explicit boolean")
        if ids[str(group.get("group_id"))] != 1:
            row["errors"].append("Duplicate group_id (including disabled entries)")
        if group.get("enabled") is not False:
            try:
                row["paths"] = validate_group(group)
                assets = digest({"config": asset_hashes(row["paths"]["config"]),
                                 "tokenizer": asset_hashes(row["paths"]["tokenizer"])})
                if common_assets is None:
                    common_assets = assets
                elif assets != common_assets:
                    raise ValueError("Config/tokenizer contents differ from the first available group")
                row["asset_sha256"] = assets
            except Exception as error:
                row["errors"].append(str(error))
        row["status"] = "preflight_failed" if row["errors"] else (
            "disabled" if group.get("enabled") is False else "ready")
        report.append(row)
    return {"groups": report, "data_sha256": data_sha}


def _command(row, group, shared, smoke, out, fast):
    return [sys.executable, str(Path(__file__).with_name("eval_ppl.py")),
            "--group-id", group["group_id"], "--description", group["description"],
            "--expected-step", str(group["expected_step"]), "--ckpt", row["paths"]["checkpoint"],
            "--config", row["paths"]["config"], "--tokenizer", row["paths"]["tokenizer"],
            "--data", shared["data"], "--out", str(out), "--device", shared["device"],
            "--seed", str(shared["seed"]), "--max-iters", str(shared["max_iters"])] \
        + (["--smoke"] if smoke else []) + (["--fast"] if fast else [])


def _merge(shard_outs):
    nll, tok, nw = 0.0, 0, 0
    statuses = []
    for so in shard_outs:
        try:
            s = read_json(so / "summary.json")
        except Exception:
            statuses.append("missing")
            continue
        statuses.append(s.get("status"))
        if s.get("status") == "complete":
            nll += float(s.get("sum_nll", 0.0))
            tok += int(s.get("n_tokens", 0))
            nw += int(s.get("n_windows", 0))
    return {"n_tokens": tok, "n_windows": nw, "sum_nll": nll,
            "mean_nll": (nll / tok) if tok else None,
            "ppl": math.exp(nll / tok) if tok else None,
            "shard_statuses": statuses}


def execute(manifest, run_dir, smoke=False, preflight_only=False, num_shards=1, fast=False):
    report = preflight(manifest)
    report.update(status="running", mode="preflight" if preflight_only else ("smoke" if smoke else "full"),
                  fast=fast, all_enabled_succeeded=False)
    run_dir = Path(run_dir)
    write_json(run_dir / "manifest.resolved.json", manifest)
    write_json(run_dir / "serial_summary.json", report)
    shared = manifest.get("evaluation", {})
    ready = [r for r in report["groups"] if r["status"] == "ready" and not preflight_only]

    def env_for(i):
        env = os.environ.copy()
        env.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", PYTHONUNBUFFERED="1")
        if num_shards > 1:
            env["ASCEND_RT_VISIBLE_DEVICES"] = str(i)
            env["MASTER_PORT"] = str(29511 + i)
        return env

    for row in ready:
        group = manifest["groups"][row["index"]]
        row.update(status="running", output=str(run_dir / group["group_id"]))
        shard_outs, procs = [], []
        for k in range(num_shards):
            so = run_dir / f"{group['group_id']}__shard{k}"
            command = _command(row, group, shared, smoke, so, fast) + ["--shard", str(k), "--num-shards", str(num_shards)]
            log = (run_dir / f"{group['group_id']}__shard{k}.log").open("x", encoding="utf-8")
            print(f"[{group['group_id']}] shard{k}/{num_shards} starting on npu:{k}", flush=True)
            procs.append((k, so, log, subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=env_for(k))))
            shard_outs.append(so)
        returncodes = {}
        for k, so, log, proc in procs:
            proc.wait()
            log.close()
            returncodes[k] = proc.returncode
            print(f"[{group['group_id']}] shard{k} exit={proc.returncode}", flush=True)
        merged = _merge(shard_outs)
        ok = (all(rc == 0 for rc in returncodes.values())
              and merged["n_tokens"] > 0 and merged["ppl"] is not None)
        status = ("smoke_complete" if smoke else "complete") if ok else "failed"
        merged.update(group_id=group["group_id"], mode="smoke" if smoke else "full",
                      status=status, shard_count=num_shards, shard_returncodes=returncodes,
                      max_iters=shared.get("max_iters"), seq_len=shared.get("seq_len", 4096))
        final_out = run_dir / group["group_id"]
        final_out.mkdir(parents=True, exist_ok=True)
        write_json(final_out / "summary.json", merged)
        base = {}
        for so in shard_outs:
            if (so / "run.json").is_file():
                base = read_json(so / "run.json")
                break
        base.update(status=status, shard_count=num_shards, ppl=merged.get("ppl"), n_tokens=merged["n_tokens"])
        write_json(final_out / "run.json", base)
        if ok:
            row.update(status=status, result=merged)
        else:
            bad = "; ".join(f"shard{k}={rc}" for k, rc in returncodes.items() if rc)
            row.update(status="failed", error=bad or "no tokens", result=merged)
        print(f"[{group['group_id']}] {row['status']}: ppl={merged.get('ppl')} "
              f"tokens={merged['n_tokens']}", flush=True)
        write_json(run_dir / "serial_summary.json", report)

    enabled = [r for r in report["groups"] if r["enabled"] is not False]
    success_status = "ready" if preflight_only else ("smoke_complete" if smoke else "complete")
    ok = (bool(enabled) and all(r["status"] == success_status for r in enabled)
          and not any(r["errors"] for r in report["groups"]))
    report.update(status=("preflight_passed" if preflight_only else "finished") if ok else "incomplete",
                  all_enabled_succeeded=ok if not preflight_only else False,
                  enabled_count=len(enabled), skipped_disabled=sum(r["enabled"] is False for r in report["groups"]))
    write_json(run_dir / "serial_summary.json", report)
    return 0 if ok else 1


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", required=True)
    p.add_argument("--output-root", help="Override manifest output_root")
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--preflight", action="store_true")
    mode.add_argument("--smoke", action="store_true")
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--fast", action="store_true",
                   help="Enable Triton GDN + grouped-expert matmul (training path) for full-length PPL.")
    args = p.parse_args()
    if args.num_shards < 1:
        p.error("--num-shards must be positive")
    manifest = read_json(args.manifest)
    root = checked_path(args.output_root or manifest.get("output_root"), "output")
    protect_output(root, [g.get(k) for g in manifest.get("groups", []) if isinstance(g, dict)
                          for k in ("checkpoint", "config", "tokenizer")])
    root.mkdir(parents=True, exist_ok=True)
    name = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:10]
    run_dir = root / name
    run_dir.mkdir()
    try:
        code = execute(manifest, run_dir, args.smoke, args.preflight, args.num_shards, args.fast)
    except Exception as error:
        write_json(run_dir / "serial_summary.json", {"status": "failed", "all_enabled_succeeded": False,
                                                     "error": f"{type(error).__name__}: {error}"})
        code = 1
    print(f"Summary: {run_dir / 'serial_summary.json'}", flush=True)
    return code


if __name__ == "__main__":
    sys.exit(main())
