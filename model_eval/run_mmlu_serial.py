#!/usr/bin/env python
"""Preflight and evaluate enabled groups; optionally shard one group across many NPUs.

Groups are processed one at a time. With --num-shards N>1, a group's questions are
split round-robin across N child processes, each pinned to a distinct NPU via
ASCEND_RT_VISIBLE_DEVICES; their per-question results are merged into one formal
summary so a single model's MMLU score lands ~N times faster. N=1 keeps the
original strictly-serial single-process-per-group behavior.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import uuid

from eval_mmlu import asset_hashes
from mmlu_core import (SUBJECTS, aggregate, checked_path, digest, load_data, protect_output,
                       read_json, validate_group, write_json)


def preflight(manifest):
    groups = manifest.get("groups")
    if manifest.get("schema_version") != 1 or not isinstance(groups, list) or not groups:
        raise ValueError("Expected schema_version=1 and nonempty groups list")
    shared = manifest.get("evaluation", {})
    global_errors = []
    data_hash = None
    expected = None
    try:
        data = load_data(checked_path(shared.get("data"), "file"))
        data_hash = data["content_sha256"]
        expected = {s: len(data["subjects"][s]["test"]) for s in SUBJECTS}
        if (type(shared.get("seed")) is not int or type(shared.get("max_length")) is not int
                or shared["max_length"] <= 0 or shared.get("device") not in ("npu", "cpu")):
            raise ValueError("evaluation requires integer seed, positive max_length, and device npu/cpu")
    except (ValueError, OSError, KeyError, TypeError) as error:
        global_errors.append(str(error))
    if not all(isinstance(g, dict) for g in groups):
        raise ValueError("Each group must be an object")
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
            row["errors"].extend(global_errors)
            try:
                row["paths"] = validate_group(group)
                # Architecture/tokenizer and protocol must be the same across this comparison.
                assets = digest({"config": asset_hashes(row["paths"]["config"]),
                                 "tokenizer": asset_hashes(row["paths"]["tokenizer"])})
                if common_assets is None:
                    common_assets = assets
                elif assets != common_assets:
                    raise ValueError("Config/tokenizer contents differ from the first available group")
                row["asset_sha256"] = assets
            except (ValueError, OSError, KeyError, TypeError) as error:
                row["errors"].append(str(error))
        row["status"] = "preflight_failed" if row["errors"] else (
            "disabled" if group.get("enabled") is False else "ready")
        report.append(row)
    return {"groups": report, "data_content_sha256": data_hash, "global_errors": global_errors,
            "expected": expected}


def _build_command(row, manifest, run_dir, shared, smoke, out=None):
    group = manifest["groups"][row["index"]]
    if out is None:
        out = run_dir / group["group_id"]
    # Child exclusively owns its new output directory; parent owns the log outside it.
    command = [sys.executable, str(Path(__file__).with_name("eval_mmlu.py")),
               "--group-id", group["group_id"], "--description", group["description"],
               "--expected-step", str(group["expected_step"]), "--ckpt", row["paths"]["checkpoint"],
               "--config", row["paths"]["config"], "--tokenizer", row["paths"]["tokenizer"],
               "--data", shared["data"], "--out", str(out), "--device", shared["device"],
               "--seed", str(shared["seed"]), "--max-length", str(shared["max_length"])]
    if smoke:
        command.append("--smoke")
    return group, out, command


def _finalize_row(row, out, returncode, smoke, data_sha256):
    """Read the child's summary/metadata and validate against preflight; update `row`."""
    row["returncode"] = returncode
    try:
        summary = read_json(out / "summary.json")
        row["result"] = summary  # Preserve partial counts/fatal_error even when the child failed.
        metadata = read_json(out / "run.json")
        assets = digest({"config": metadata.get("config_assets"), "tokenizer": metadata.get("tokenizer_assets")})
        if (assets != row["asset_sha256"] or metadata.get("data_content_sha256") != data_sha256):
            raise ValueError("Child data/config/tokenizer differs from preflight; comparison invalid")
        wanted = "smoke_complete" if smoke else "complete"
        if (returncode != 0 or summary.get("status") != wanted or not summary.get("complete")
                or summary.get("group_id") != row["group_id"]
                or summary.get("mode") != ("smoke" if smoke else "full")
                or (not smoke and not summary.get("formal_score"))
                or (smoke and summary.get("formal_score"))):
            raise ValueError(f"Child failed/incomplete/inconsistent: exit={returncode}, summary={summary.get('status')}")
        row.update(status=wanted, result=summary)
    except Exception as error:
        row.update(status="failed", error=f"{type(error).__name__}: {error}")


def _merge_shards(row, out, shard_outs, returncodes, smoke, expected):
    """Merge per-shard question records into one formal summary; update `row`."""
    mode = "smoke" if smoke else "full"
    expected = {"abstract_algebra": 2, "anatomy": 2} if smoke else expected
    records = []
    errors = []
    for k, so in enumerate(shard_outs):
        if returncodes[k] != 0:
            errors.append(f"shard{k}: exit {returncodes[k]}")
        qfile = so / "questions.jsonl"
        if qfile.is_file():
            for line in qfile.open(encoding="utf-8"):
                if line.strip():
                    records.append(json.loads(line))
        else:
            errors.append(f"shard{k}: no questions.jsonl")
    out.mkdir(parents=True, exist_ok=True)
    # Deterministic canonical order (subject-major, then test index) for the merged audit file.
    records.sort(key=lambda r: (SUBJECTS.index(r["subject"]), int(r["id"].rsplit("/", 1)[1])))
    with (out / "questions.jsonl").open("w", encoding="utf-8") as stream:
        for rec in records:
            stream.write(json.dumps(rec, ensure_ascii=False, allow_nan=False) + "\n")
    try:
        final = aggregate(records, expected, mode)
    except Exception as error:
        final = {"complete": False, "formal_score": False, "overall_acc": None,
                 "correct": 0, "evaluated": 0, "errors": 0, "total": 0, "unattempted": 0,
                 "subjects": {}, "acc_on_evaluated": None}
        errors.append(f"{type(error).__name__}: {error}")
    # Reuse the first shard's run.json metadata (protocol, checkpoint identity, asset hashes).
    base = {}
    for so in shard_outs:
        if (so / "run.json").is_file():
            base = read_json(so / "run.json")
            break
    wanted = "smoke_complete" if smoke else "complete"
    ok = not errors and final.get("complete") and (smoke or final.get("formal_score"))
    status = wanted if ok else "failed"
    final.update(group_id=row["group_id"], mode=mode, status=status,
                 requested_total=(4 if smoke else 14042),
                 protocol_sha256=base.get("protocol_sha256"),
                 shard_count=len(shard_outs), shard_returncodes=returncodes)
    if errors:
        final["shard_errors"] = errors
    write_json(out / "summary.json", final)
    base.update(status=status, formal_score=final.get("formal_score"), shard_count=len(shard_outs))
    write_json(out / "run.json", base)
    if ok:
        row.update(status=wanted, result=final)
    else:
        row.update(status="failed", error="; ".join(errors) or "incomplete", result=final)


def execute(manifest, run_dir, smoke=False, preflight_only=False, num_shards=1, launch=subprocess.run):
    """launch injection is for local CPU-free process/failure tests only; no CLI override.

    num_shards>1 splits each group's questions round-robin across that many NPUs and
    merges the per-shard results into one formal summary (one model at a time)."""
    report = preflight(manifest)
    report.update(status="running", mode="preflight" if preflight_only else ("smoke" if smoke else "full"),
                  all_enabled_succeeded=False)
    run_dir = Path(run_dir)
    write_json(run_dir / "manifest.resolved.json", manifest)
    write_json(run_dir / "serial_summary.json", report)
    shared = manifest.get("evaluation", {})
    ready = [row for row in report["groups"] if row["status"] == "ready" and not preflight_only]

    def env_for(i):
        env = os.environ.copy()
        env.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", PYTHONUNBUFFERED="1")
        if num_shards > 1:
            env["ASCEND_RT_VISIBLE_DEVICES"] = str(i)
            env["MASTER_PORT"] = str(29511 + i)
        return env

    for row in ready:
        group = manifest["groups"][row["index"]]
        if num_shards > 1:
            row.update(status="running", output=str(run_dir / group["group_id"]))
            procs, shard_outs = [], []
            for k in range(num_shards):
                so = run_dir / f"{group['group_id']}__shard{k}"
                _, _, command = _build_command(row, manifest, run_dir, shared, smoke, out=so)
                command += ["--shard", str(k), "--num-shards", str(num_shards)]
                log = (run_dir / f"{group['group_id']}__shard{k}.log").open("x", encoding="utf-8")
                print(f"[{group['group_id']}] shard{k}/{num_shards} starting on npu:{k}", flush=True)
                procs.append((k, so, log, subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                                           env=env_for(k))))
                shard_outs.append(so)
            write_json(run_dir / "serial_summary.json", report)
            returncodes = {}
            for k, so, log, proc in procs:
                proc.wait()
                log.close()
                returncodes[k] = proc.returncode
                print(f"[{group['group_id']}] shard{k} exit={proc.returncode}", flush=True)
            _merge_shards(row, run_dir / group["group_id"], shard_outs, returncodes, smoke, report["expected"])
            print(f"[{group['group_id']}] {row['status']}", flush=True)
            write_json(run_dir / "serial_summary.json", report)
        else:
            group, out, command = _build_command(row, manifest, run_dir, shared, smoke)
            row.update(status="running", output=str(out), command=command)
            write_json(run_dir / "serial_summary.json", report)
            print(f"[{group['group_id']}] starting; log={run_dir / (group['group_id'] + '.log')}", flush=True)
            try:
                env = env_for(0)
                with (run_dir / (group["group_id"] + ".log")).open("x", encoding="utf-8") as log:
                    child = launch(command, stdout=log, stderr=subprocess.STDOUT, env=env, check=False)
                _finalize_row(row, out, child.returncode, smoke, report["data_content_sha256"])
            except Exception as error:
                row.update(status="failed", error=f"{type(error).__name__}: {error}")
            print(f"[{group['group_id']}] {row['status']}", flush=True)
            write_json(run_dir / "serial_summary.json", report)

    enabled = [r for r in report["groups"] if r["enabled"] is not False]
    success_status = "ready" if preflight_only else ("smoke_complete" if smoke else "complete")
    ok = bool(enabled) and all(r["status"] == success_status for r in enabled)
    # Manifest defects are errors even on disabled duplicate entries.
    ok = ok and not any(r["errors"] for r in report["groups"])
    report.update(status=("preflight_passed" if preflight_only else "finished") if ok else "incomplete",
                  all_enabled_succeeded=ok if not preflight_only else False,
                  preflight_passed=ok if preflight_only else None,
                  enabled_count=len(enabled), skipped_disabled=sum(r["enabled"] is False for r in report["groups"]))
    write_json(run_dir / "serial_summary.json", report)
    return 0 if ok else 1


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", required=True)
    p.add_argument("--output-root", help="Override manifest output_root (absolute directory)")
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--preflight", action="store_true", help="No model/torch import or child process")
    mode.add_argument("--smoke", action="store_true", help="Same fixed four questions in each enabled group")
    p.add_argument("--num-shards", type=int, default=1,
                   help="Split each group's questions round-robin across this many NPUs "
                        "(default 1 = single process per group)")
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
        code = execute(manifest, run_dir, args.smoke, args.preflight, args.num_shards)
    except Exception as error:
        write_json(run_dir / "serial_summary.json", {"status": "failed", "all_enabled_succeeded": False,
                                                     "error": f"{type(error).__name__}: {error}"})
        code = 1
    print(f"Summary: {run_dir / 'serial_summary.json'}", flush=True)
    return code


if __name__ == "__main__":
    sys.exit(main())
