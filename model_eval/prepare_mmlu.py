#!/usr/bin/env python
"""Prepare one auditable JSON pack, offline by default. No model imports."""
import argparse
from pathlib import Path

from mmlu_core import REVISION, SUBJECTS, digest, file_hash, protect_output, validate_data, write_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument("--source", help="Local pinned HF snapshot directory with subject/dev,test parquet")
    source.add_argument("--download", action="store_true", help="Explicitly download ONLY the pinned dev/test files")
    p.add_argument("--source-revision", choices=[REVISION], help="Required offline provenance assertion")
    p.add_argument("--out", required=True, help="New JSON data pack (will not overwrite)")
    args = p.parse_args()
    dest = Path(args.out).expanduser().resolve()
    if dest.exists():
        p.error("--out already exists; do not overwrite a data pack")
    if args.source and args.source_revision != REVISION:
        p.error("Offline import requires --source-revision " + REVISION + "; verify origin before asserting")
    # pyarrow/huggingface_hub are preparation-only dependencies, not inference requirements.
    import pyarrow.parquet as pq

    if args.download:
        from huggingface_hub import snapshot_download
        root = Path(snapshot_download(repo_id="cais/mmlu", repo_type="dataset", revision=REVISION,
                                     allow_patterns=[f"{s}/{split}-*.parquet" for s in SUBJECTS
                                                     for split in ("dev", "test")]))
    else:
        root = Path(args.source).expanduser().resolve(strict=True)
    protect_output(dest, [str(root)])
    data = {"schema_version": 1, "dataset": "cais/mmlu", "revision": REVISION,
            "provenance": {"method": "pinned_hf_download" if args.download else "operator_asserted_local_snapshot",
                           "source": str(root), "files": {}}, "subjects": {}}
    for subject in SUBJECTS:
        data["subjects"][subject] = {}
        for split in ("dev", "test"):
            # The pinned revision has one shard per split per subject. Reject partial/mixed snapshots.
            files = sorted((root / subject).glob(f"{split}-*.parquet"))
            if [f.name for f in files] != [f"{split}-00000-of-00001.parquet"]:
                raise ValueError(f"{subject}/{split}: expected exactly one pinned parquet shard")
            path = files[0]
            data["provenance"]["files"][path.relative_to(root).as_posix()] = file_hash(path)
            rows = pq.read_table(path).to_pylist()
            normalized = []
            for i, row in enumerate(rows):
                if row.get("subject", subject) != subject:
                    raise ValueError(f"Wrong subject in {path}")
                normalized.append({"id": f"{subject}/{split}/{i}", "question": row["question"],
                                   "choices": row["choices"], "answer": row["answer"]})
            data["subjects"][subject][split] = normalized
    data["content_sha256"] = digest(data["subjects"])
    validate_data(data)
    dest.parent.mkdir(parents=True, exist_ok=True)
    # Reserve the name before atomic publication, so another preparer cannot overwrite it.
    with dest.open("x", encoding="utf-8"):
        pass
    write_json(dest, data)
    print(f"Prepared 57 subjects / 285 dev / 14042 test: {dest}")
    print(f"Content SHA256: {data['content_sha256']}")


if __name__ == "__main__":
    main()
