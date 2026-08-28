#!/usr/bin/env python

"""Audit and plan the diverse real-robot acquisition pilot."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from lerobot.datasets.diverse_pilot import (
    AdmissionError,
    attach_payload_sizes,
    download_manifest_payload,
    generate_review_proxies,
    load_source_specs,
    nominate_lerobot_candidates,
    resolve_lerobot_payload,
    run_source_audits,
    safe_cleanup_payload,
    validate_packed_v3_dataset,
    validate_staging_capacity,
    write_json,
    write_packed_v3_dataset,
)


def _audit(args: argparse.Namespace) -> None:
    specs = load_source_specs(args.config)
    audits = run_source_audits(
        specs,
        args.metadata_root,
        args.output_root / "audits",
        fetch=not args.no_fetch,
        token=os.environ.get("HF_TOKEN"),
    )
    summary = {
        spec.name: {
            "repo_id": spec.repo_id,
            "passed": audit["admission"]["passed"],
            "failures": audit["admission"]["failures"],
        }
        for spec, audit in zip(specs, audits, strict=True)
    }
    write_json(args.output_root / "audit_summary.json", summary)
    print(json.dumps(summary, indent=2))


def _nominate(args: argparse.Namespace) -> None:
    slots = []
    for spec in load_source_specs(args.config):
        audit_path = args.output_root / "audits" / f"{spec.name}.json"
        if not audit_path.exists():
            raise FileNotFoundError(f"Run audit first; missing {audit_path}")
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        if not audit["admission"]["passed"]:
            slots.extend(
                {
                    "source": spec.name,
                    "slot": slot,
                    "status": "blocked_by_source_audit",
                    "failures": audit["admission"]["failures"],
                }
                for slot in range(spec.pilot_episodes)
            )
            continue
        candidates = nominate_lerobot_candidates(args.metadata_root / spec.name, count=spec.pilot_episodes)
        slots.extend({"source": spec.name, "slot": i, **candidate} for i, candidate in enumerate(candidates))
    write_json(args.output_root / "candidate_manifest.json", {"pilot_slots": slots})
    print(json.dumps({"pilot_slots": slots}, indent=2))


def _plan(args: argparse.Namespace) -> None:
    specs = {spec.name: spec for spec in load_source_specs(args.config)}
    spec = specs[args.source]
    audit = json.loads((args.output_root / "audits" / f"{spec.name}.json").read_text(encoding="utf-8"))
    try:
        manifest = resolve_lerobot_payload(spec, audit, args.metadata_root / spec.name, args.episodes)
        manifest = attach_payload_sizes(manifest, token=os.environ.get("HF_TOKEN"))
    except AdmissionError as error:
        raise SystemExit(str(error)) from error
    destination = args.output_root / "download_plans" / f"{spec.name}.json"
    write_json(destination, manifest)
    print(destination)


def _download(args: argparse.Namespace) -> None:
    manifest = json.loads(args.plan.read_text(encoding="utf-8"))
    validate_staging_capacity(manifest, args.staging_root)
    paths = download_manifest_payload(manifest, args.staging_root, token=os.environ.get("HF_TOKEN"))
    print("\n".join(str(path) for path in paths))


def _proxies(args: argparse.Namespace) -> None:
    manifest = json.loads(args.plan.read_text(encoding="utf-8"))
    paths = generate_review_proxies(manifest, args.staging_root, args.proxy_root)
    print("\n".join(str(path) for path in paths))


def _extract(args: argparse.Namespace) -> None:
    specs = {spec.name: spec for spec in load_source_specs(args.config)}
    spec = specs[args.source]
    audit = json.loads((args.output_root / "audits" / f"{spec.name}.json").read_text(encoding="utf-8"))
    manifest = json.loads(args.plan.read_text(encoding="utf-8"))
    report = write_packed_v3_dataset(
        spec,
        audit,
        manifest,
        args.metadata_root / spec.name,
        args.staging_root,
        args.annotations_root,
        args.dataset_root,
        output_repo_id=args.repo_id,
        stride_s=args.stride,
        min_chunks_per_episode=args.min_chunks_per_episode,
    )
    print(json.dumps(report, indent=2))


def _validate(args: argparse.Namespace) -> None:
    report = validate_packed_v3_dataset(args.dataset_root, args.repo_id)
    print(json.dumps(report, indent=2))


def _cleanup(args: argparse.Namespace) -> None:
    report_path = args.dataset_root / "meta/validation_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if not report.get("passed"):
        raise SystemExit(f"Cleanup requires a passing validation report: {report_path}")
    if not args.confirm_validated_output:
        raise SystemExit("Cleanup requires --confirm-validated-output after visual/annotation review")
    manifest = json.loads(args.plan.read_text(encoding="utf-8"))
    removed = safe_cleanup_payload(manifest, args.staging_root)
    print("\n".join(str(path) for path in removed))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--metadata-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    commands = parser.add_subparsers(dest="command", required=True)
    audit = commands.add_parser("audit", help="fetch metadata only and run admission audits")
    audit.add_argument("--no-fetch", action="store_true")
    audit.set_defaults(handler=_audit)
    nominate = commands.add_parser("nominate", help="nominate all eight pilot slots")
    nominate.set_defaults(handler=_nominate)
    plan = commands.add_parser("plan-download", help="resolve exact payload paths without downloading")
    plan.add_argument("--source", required=True)
    plan.add_argument("--episodes", type=int, nargs="+", required=True)
    plan.set_defaults(handler=_plan)
    download = commands.add_parser("download", help="download only the files in a reviewed plan")
    download.add_argument("--plan", type=Path, required=True)
    download.add_argument("--staging-root", type=Path, required=True)
    download.set_defaults(handler=_download)
    proxies = commands.add_parser("proxies", help="make full-episode review proxies and annotation templates")
    proxies.add_argument("--plan", type=Path, required=True)
    proxies.add_argument("--staging-root", type=Path, required=True)
    proxies.add_argument("--proxy-root", type=Path, required=True)
    proxies.set_defaults(handler=_proxies)
    extract = commands.add_parser("extract", help="write a reviewed packed LeRobot v3 dataset")
    extract.add_argument("--source", required=True)
    extract.add_argument("--plan", type=Path, required=True)
    extract.add_argument("--staging-root", type=Path, required=True)
    extract.add_argument("--annotations-root", type=Path, required=True)
    extract.add_argument("--dataset-root", type=Path, required=True)
    extract.add_argument("--repo-id", required=True)
    extract.add_argument("--stride", type=float, choices=(2.0, 3.0), default=2.0)
    extract.add_argument(
        "--min-chunks-per-episode",
        type=int,
        default=3,
        help="reject an ordinary episode when fewer full-window-approved chunks remain",
    )
    extract.set_defaults(handler=_extract)
    validate = commands.add_parser("validate", help="decode and numerically validate every packed row")
    validate.add_argument("--dataset-root", type=Path, required=True)
    validate.add_argument("--repo-id", required=True)
    validate.set_defaults(handler=_validate)
    cleanup = commands.add_parser("cleanup", help="remove only manifest payloads after validation")
    cleanup.add_argument("--plan", type=Path, required=True)
    cleanup.add_argument("--staging-root", type=Path, required=True)
    cleanup.add_argument("--dataset-root", type=Path, required=True)
    cleanup.add_argument("--confirm-validated-output", action="store_true")
    cleanup.set_defaults(handler=_cleanup)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
