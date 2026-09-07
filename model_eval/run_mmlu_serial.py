#!/usr/bin/env python
"""Preflight and evaluate enabled groups in fresh, strictly serial child processes."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import os
from pathlib import Path
import subprocess
import sys
import uuid

from eval_mmlu import asset_hashes
from mmlu_core import (checked_path, digest, load_data, protect_output, read_json,
                       validate_group, write_json)


def preflight(manifest):
    groups = manifest.get("groups")
    if manifest.get("schema_version") != 1 or not isinstance(groups, list) or not groups:
        raise ValueError("Expected schema_version=1 and nonempty groups list")
    shared = manifest.get("evaluation", {})
    global_errors = []
    data_hash = None
    try:
        data_hash = load_data(checked_path(shared.get("data"), "file"))["content_sha256"]
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
    return {"groups": report, "data_content_sha256": data_hash, "global_errors": global_errors}


def execute(manifest, run_dir, smoke=False, preflight_only=False, launch=subprocess.run):
    """launch injection is for local CPU-free process/failure tests only; no CLI override."""
    report = preflight(manifest)
    report.update(status="running", mode="preflight" if preflight_only else ("smoke" if smoke else "full"),
                  all_enabled_succeeded=False)
    run_dir = Path(run_dir)
    write_json(run_dir / "manifest.resolved.json", manifest)
    write_json(run_dir / "serial_summary.json", report)
    shared = manifest.get("evaluation", {})
    for row in report["groups"]:
        if row["status"] != "ready" or preflight_only:
            continue
        group = manifest["groups"][row["index"]]
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
        row.update(status="running", output=str(out), command=command)
        write_json(run_dir / "serial_summary.json", report)
        print(f"[{group['group_id']}] starting; log={run_dir / (group['group_id'] + '.log')}", flush=True)
        try:
            env = os.environ.copy()
            env.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", PYTHONUNBUFFERED="1")
            with (run_dir / (group["group_id"] + ".log")).open("x", encoding="utf-8") as log:
                child = launch(command, stdout=log, stderr=subprocess.STDOUT, env=env, check=False)
            row["returncode"] = child.returncode
            summary = read_json(out / "summary.json")
            row["result"] = summary  # Preserve partial counts/fatal_error even when the child failed.
            metadata = read_json(out / "run.json")
            assets = digest({"config": metadata.get("config_assets"), "tokenizer": metadata.get("tokenizer_assets")})
            if (assets != row["asset_sha256"] or metadata.get("data_content_sha256") != report["data_content_sha256"]):
                raise ValueError("Child data/config/tokenizer differs from preflight; comparison invalid")
            wanted = "smoke_complete" if smoke else "complete"
            if (child.returncode != 0 or summary.get("status") != wanted or not summary.get("complete")
                    or summary.get("group_id") != group["group_id"]
                    or summary.get("mode") != ("smoke" if smoke else "full")
                    or (not smoke and not summary.get("formal_score"))
                    or (smoke and summary.get("formal_score"))):
                raise ValueError(f"Child failed/incomplete/inconsistent: exit={child.returncode}, summary={summary.get('status')}")
            row.update(status=wanted, result=summary)
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
    args = p.parse_args()
    manifest = read_json(args.manifest)
    root = checked_path(args.output_root or manifest.get("output_root"), "output")
    protect_output(root, [g.get(k) for g in manifest.get("groups", []) if isinstance(g, dict)
                          for k in ("checkpoint", "config", "tokenizer")])
    root.mkdir(parents=True, exist_ok=True)
    name = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:10]
    run_dir = root / name
    run_dir.mkdir()
    try:
        code = execute(manifest, run_dir, args.smoke, args.preflight)
    except Exception as error:
        write_json(run_dir / "serial_summary.json", {"status": "failed", "all_enabled_succeeded": False,
                                                     "error": f"{type(error).__name__}: {error}"})
        code = 1
    print(f"Summary: {run_dir / 'serial_summary.json'}", flush=True)
    return code


if __name__ == "__main__":
    sys.exit(main())
