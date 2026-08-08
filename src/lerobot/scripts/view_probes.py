# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Probe viewer: a local, dependency-free browser UI over a run's validation output.

    python -m lerobot.scripts.view_probes outputs/<run>

Scans ``<run>/validation/step_*/<probe>/`` on every request, so a checkpoint
that lands mid-session appears on refresh — nothing is baked in and there is no
export step.

The viewer knows nothing about any individual probe. Each probe describes itself
in an ``index.json`` (see ``lerobot.probes.manifest.write_index``): title,
documentation, headline metrics with thresholds, and a caption per figure. A
probe that hasn't adopted that yet still appears, with its files listed and its
module docstring read straight off disk — it just can't say which of its numbers
matter.
"""

from __future__ import annotations

import argparse
import json
import mimetypes
import re
import threading
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

from lerobot.probes.manifest import (
    GROUPS,
    SUITE_DOC,
    module_doc_from_source,
    panel_kind,
)

CHUNK_SIZE = 1024 * 1024
MEDIA_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".svg", ".html", ".htm", ".mp4", ".webm"}
# Probes fan out over episode x layer x timestep, so a flat cap would silently
# truncate whole facets. The UI faceting handles the count; this is only a guard
# against pathological directories.
MAX_FALLBACK_PANELS = 400
MAX_PANEL_DEPTH = 3
PROBE_SOURCE_DIR = Path(__file__).resolve().parent.parent / "probes"


# ── discovery ────────────────────────────────────────────────────────────────

def resolve_validation_dir(run_dir: str | Path) -> Path:
    """Accept a run dir, its validation dir, or a single step dir."""
    path = Path(run_dir).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"No such directory: {path}")
    if path.name.startswith("step_"):
        return path.parent
    if (path / "validation").is_dir():
        return path / "validation"
    if any(child.name.startswith("step_") for child in path.iterdir() if child.is_dir()):
        return path
    raise FileNotFoundError(
        f"{path} holds no step_* directories and no validation/ subdirectory. "
        f"Point this at a training run directory."
    )


def _natural_key(text: str) -> list:
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", text)]


def discover_steps(val_dir: Path) -> list[tuple[str, int]]:
    steps = []
    for child in sorted(val_dir.iterdir(), key=lambda p: _natural_key(p.name)):
        if child.is_dir() and child.name.startswith("step_"):
            digits = child.name.split("_")[-1]
            steps.append((child.name, int(digits) if digits.isdigit() else 0))
    return steps


def _fallback_panels(probe_dir: Path) -> list[dict]:
    """Every media file under the probe dir, naturally sorted.

    Subdirectories are kept in the path: probes fan out as ``ep0000_L00/*.mp4``
    or ``ep0000_t0p50/L28/*.mp4``, and the UI turns those levels into selectors.
    """
    found = []
    for path in probe_dir.rglob("*"):
        if path.is_file() and path.suffix.lower() in MEDIA_SUFFIXES:
            rel = path.relative_to(probe_dir)
            if len(rel.parts) <= MAX_PANEL_DEPTH:
                found.append(rel.as_posix())
    found.sort(key=_natural_key)
    # Without a manifest, the only signal for what matters is layout: probes put
    # summary figures at the top level and per-episode / per-layer fan-out in
    # subdirectories. Good enough to keep arrival uncluttered.
    return [
        {
            "file": rel, "caption": rel, "how": "", "kind": panel_kind(rel),
            "primary": "/" not in rel, "refs": [],
        }
        for rel in found[:MAX_FALLBACK_PANELS]
    ]


def _fallback_metrics(probe_dir: Path) -> list[dict]:
    """Scalars off the probe's own summary JSON, unlabelled and without direction."""
    for path in sorted(probe_dir.glob("*.json"), key=lambda p: _natural_key(p.name)):
        if path.name == "index.json":
            continue
        try:
            with path.open() as f:
                payload = json.load(f)
        except (OSError, ValueError):
            continue
        summary = payload.get("summary") if isinstance(payload, dict) else None
        if not isinstance(summary, dict):
            continue
        rows = []
        for key, value in summary.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                continue
            rows.append(
                {
                    "key": key, "label": key, "value": float(value), "good": "none",
                    "fmt": 0 if isinstance(value, int) else 4, "note": "",
                    "baseline": None, "warn": None, "bad": None, "status": "info",
                }
            )
            if len(rows) >= 10:
                break
        if rows:
            return rows
    return []


def read_probe(probe_dir: Path) -> dict:
    """One probe's description at one step: its manifest, or a best-effort stand-in."""
    has_log = (probe_dir / "probe.log").is_file()
    manifest_path = probe_dir / "index.json"
    if manifest_path.is_file():
        try:
            with manifest_path.open() as f:
                index = json.load(f)
            index["has_manifest"] = True
            index["has_log"] = has_log
            index.setdefault("metrics", [])
            index.setdefault("panels", [])
            return index
        except (OSError, ValueError):
            pass

    # No manifest: the probe crashed before writing one, or skipped itself (a
    # critic probe under skip_critic leaves nothing but its log). Identity is then
    # whatever the directory name and a same-named module can supply.
    probe_id = probe_dir.name
    source = PROBE_SOURCE_DIR / f"{probe_id}.py"
    return {
        "schema": 0,
        "id": probe_id,
        "title": probe_id.replace("_", " ").title(),
        "group": "Other",
        "claim": "",
        "doc": module_doc_from_source(str(source)) if source.is_file() else "",
        "status": "info",
        "metrics": _fallback_metrics(probe_dir),
        "panels": _fallback_panels(probe_dir),
        "extra": {},
        "has_manifest": False,
        "has_log": has_log,
    }




def build_index(val_dir: Path) -> dict:
    """Merge every step's probe manifests into the payload the UI renders."""
    steps = discover_steps(val_dir)
    probes: dict[str, dict] = {}

    for step_name, step_num in steps:
        for probe_dir in sorted((val_dir / step_name).iterdir(), key=lambda p: p.name):
            if not probe_dir.is_dir():
                continue
            index = read_probe(probe_dir)
            probe = probes.setdefault(
                index["id"],
                {
                    "id": index["id"], "title": index["title"], "group": index["group"],
                    "claim": index.get("claim", ""), "doc": index.get("doc", ""),
                    "has_manifest": index["has_manifest"],
                    "metrics": [], "panels": {}, "status": {}, "steps": [], "log": {},
                    "see_also": [], "extra": {},
                },
            )
            if index.get("has_log"):
                probe["log"][str(step_num)] = f"/asset/{step_name}/{index['id']}/probe.log"
            # A later step's manifest supersedes an earlier fallback.
            if index["has_manifest"] and not probe["has_manifest"]:
                probe.update(
                    title=index["title"], group=index["group"], claim=index.get("claim", ""),
                    doc=index.get("doc", ""), has_manifest=True,
                )
            probe["steps"].append(step_num)
            probe["status"][str(step_num)] = index.get("status", "info")
            probe["panels"][str(step_num)] = [
                {
                    "how": "", "primary": False, "refs": [], **panel,
                    "url": f"/asset/{step_name}/{index['id']}/{panel['file']}",
                }
                for panel in index["panels"]
            ]
            probe["see_also"] = index.get("see_also", probe.get("see_also", []))
            # Per step: a probe's provenance changes when the sampled episodes do.
            probe["extra"][str(step_num)] = index.get("extra", {})
            by_key = {metric["key"]: metric for metric in probe["metrics"]}
            for metric in index["metrics"]:
                row = by_key.get(metric["key"])
                if row is None:
                    row = {
                        "key": metric["key"], "label": metric["label"], "good": metric["good"],
                        "fmt": metric["fmt"], "baseline": metric.get("baseline"),
                        "warn": metric.get("warn"), "bad": metric.get("bad"),
                        "note": "", "values": {}, "statuses": {},
                        "primary": metric.get("primary", False),
                        "refs": list(metric.get("refs", [])),
                    }
                    probe["metrics"].append(row)
                    by_key[metric["key"]] = row
                row["values"][str(step_num)] = metric["value"]
                row["statuses"][str(step_num)] = metric.get("status", "info")
                row["primary"] = row.get("primary", False) or metric.get("primary", False)
                if metric.get("refs"):
                    row["refs"] = list(metric["refs"])
                if metric.get("note"):
                    row["note"] = metric["note"]

    # Sidebar order: by group as GROUPS declares them, then alphabetically inside a
    # group. A probe with no manifest sorts into "Other", at the end.
    ordered = sorted(
        probes.values(),
        key=lambda p: (
            GROUPS.index(p["group"]) if p["group"] in GROUPS else len(GROUPS),
            p["title"],
        ),
    )
    return {
        "run": {"name": val_dir.parent.name, "dir": str(val_dir.parent), "val_dir": str(val_dir)},
        "steps": [{"name": name, "step": num} for name, num in steps],
        "groups": list(GROUPS),
        "probes": ordered,
        "suite_doc": SUITE_DOC.strip(),
    }


# ── HTTP plumbing ────────────────────────────────────────────────────────────

def _json_response(handler: BaseHTTPRequestHandler, payload: dict, status: int = 200) -> None:
    body = json.dumps(payload).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.send_header("Cache-Control", "no-store")
    handler.end_headers()
    handler.wfile.write(body)


def _parse_range(range_header: str | None, size: int) -> tuple[int, int, bool]:
    if not range_header:
        return 0, size - 1, False
    match = re.fullmatch(r"bytes=(\d*)-(\d*)", range_header.strip())
    if not match:
        return 0, size - 1, False
    start_s, end_s = match.groups()
    if start_s:
        start = int(start_s)
        end = int(end_s) if end_s else size - 1
    else:
        suffix = int(end_s) if end_s else 0
        start = max(size - suffix, 0)
        end = size - 1
    if start >= size or end < start:
        raise ValueError("unsatisfiable range")
    return start, min(end, size - 1), True


def _send_file(handler: BaseHTTPRequestHandler, path: Path, head_only: bool = False) -> None:
    size = path.stat().st_size
    content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    try:
        start, end, partial = _parse_range(handler.headers.get("Range"), size)
    except ValueError:
        handler.send_response(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
        handler.send_header("Content-Range", f"bytes */{size}")
        handler.end_headers()
        return

    length = end - start + 1
    handler.send_response(HTTPStatus.PARTIAL_CONTENT if partial else HTTPStatus.OK)
    handler.send_header("Content-Type", content_type)
    handler.send_header("Accept-Ranges", "bytes")
    handler.send_header("Content-Length", str(length))
    handler.send_header("Cache-Control", "no-cache")
    if partial:
        handler.send_header("Content-Range", f"bytes {start}-{end}/{size}")
    handler.end_headers()
    if head_only:
        return

    with path.open("rb") as file:
        file.seek(start)
        remaining = length
        while remaining:
            chunk = file.read(min(CHUNK_SIZE, remaining))
            if not chunk:
                break
            handler.wfile.write(chunk)
            remaining -= len(chunk)


def make_handler(val_dir: Path) -> type[BaseHTTPRequestHandler]:
    def asset_path(rel: str) -> Path:
        path = (val_dir / unquote(rel)).resolve()
        if not path.is_relative_to(val_dir) or not path.is_file():
            raise FileNotFoundError(rel)
        return path

    class ProbeRequestHandler(BaseHTTPRequestHandler):
        server_version = "ProbeViewer/1.0"
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt: str, *args) -> None:
            return

        def do_HEAD(self) -> None:
            path = urlparse(self.path).path
            if path.startswith("/asset/"):
                self._handle_asset(path.removeprefix("/asset/"), head_only=True)
                return
            self.send_error(HTTPStatus.NOT_FOUND)

        def do_GET(self) -> None:
            path = urlparse(self.path).path
            if path == "/":
                self._send_html(INDEX_HTML)
            elif path == "/api/index":
                _json_response(self, build_index(val_dir))
            elif path.startswith("/asset/"):
                self._handle_asset(path.removeprefix("/asset/"), head_only=False)
            else:
                self.send_error(HTTPStatus.NOT_FOUND)

        def _send_html(self, html: str) -> None:
            body = html.encode("utf-8")
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _handle_asset(self, rel: str, head_only: bool) -> None:
            try:
                path = asset_path(rel)
            except (OSError, ValueError):
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            try:
                _send_file(self, path, head_only=head_only)
            except (BrokenPipeError, ConnectionResetError):
                # The browser abandoned the download — switching figure or probe
                # cancels an in-flight multi-megabyte PNG. Not an error, and the
                # socket is already gone, so there is nobody to send a status to.
                self.close_connection = True

    return ProbeRequestHandler


INDEX_HTML = r"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Probe viewer</title><style>
:root{--bg:#101318;--fg:#dfe4ea;--muted:#828d9a;--card:#161a20;--line:#232932;--accent:#5b9dff;
 --panel:#1a1f26;--good:#3ec48c;--warn:#e0a33c;--bad:#ef5f57;--info:#6a7684}
@media (prefers-color-scheme:light){:root{--bg:#eef0f3;--fg:#161a1f;--muted:#69727d;--card:#fff;
 --line:#dde1e6;--accent:#2563c9;--panel:#f1f3f6;--good:#159169;--warn:#b8791d;--bad:#cd423b;--info:#98a1ac}}
*,*::before,*::after{box-sizing:border-box}
body{margin:0;font:13px/1.55 ui-sans-serif,-apple-system,"Segoe UI",Roboto,sans-serif;background:var(--bg);
 color:var(--fg);height:100vh;overflow:hidden;display:flex;flex-direction:column;-webkit-font-smoothing:antialiased;
 scrollbar-width:thin;scrollbar-color:var(--line) transparent}
a{color:var(--accent)}
button,select,input[type=search]{font:inherit;background:var(--bg);color:var(--fg);border:1px solid var(--line);
 border-radius:6px;padding:4px 9px;cursor:pointer}
button:hover,select:hover{border-color:var(--accent)}
button:disabled{cursor:wait;opacity:.65}
button:focus-visible,select:focus-visible,input:focus-visible{outline:2px solid var(--accent);outline-offset:2px}
input[type=range]{width:180px;accent-color:var(--accent);cursor:pointer}
input[type=range][hidden]{display:none}
kbd{background:var(--panel);border:1px solid var(--line);border-bottom-width:2px;border-radius:4px;
 padding:1px 5px;font:11px ui-monospace,Menlo,monospace;color:var(--muted)}
.num{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:11px;color:var(--muted)}

/* ── header ─────────────────────────────────────────────────────────────── */
.top{display:flex;align-items:center;gap:14px;padding:9px 14px;border-bottom:1px solid var(--line);
 background:var(--card);flex:none}
.runblock{min-width:0}
.top .run{font-weight:640;max-width:36ch;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.top .sub{color:var(--muted);font-size:11.5px;max-width:44ch;overflow:hidden;text-overflow:ellipsis;
 white-space:nowrap}
.right{margin-left:auto;display:flex;align-items:center;gap:10px}
.hint{color:var(--muted);font-size:11.5px;display:flex;gap:6px;align-items:center}
.guideshort{display:none}
.refreshmsg{min-width:0;max-width:18ch;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}

/* ── two columns: probe list, then everything about that probe ──────────── */
.cols{flex:1;display:grid;grid-template-columns:236px minmax(0,1fr);min-height:0;
 transition:grid-template-columns .14s ease}
.cols.norail{grid-template-columns:0 minmax(0,1fr)}
.rail{grid-column:1}.main{grid-column:2}
.cols.norail .rail{display:none}
.rail{border-right:1px solid var(--line);overflow:auto;padding:9px;background:var(--card)}
.rail input{width:100%;margin-bottom:8px;cursor:text}
.g{font-size:10px;letter-spacing:.09em;text-transform:uppercase;color:var(--muted);margin:12px 0 3px;padding-left:7px}
.item{display:flex;align-items:center;gap:8px;width:100%;padding:5px 8px;border:0;border-radius:6px;
 background:transparent;text-align:left;cursor:pointer}
.item:hover{background:var(--panel)}
.item.on{background:var(--accent);color:#fff}
.item.on .num{color:#fff;opacity:.9}
.dot{width:7px;height:7px;border-radius:99px;flex:none}
.dot.good{background:var(--good)}.dot.warn{background:var(--warn)}.dot.bad{background:var(--bad)}
.dot.info{background:var(--info)}.dot.none{background:transparent;border:1px solid var(--muted)}
.item .nm{flex:1;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.statuslegend{display:flex;flex-wrap:wrap;gap:4px 9px;margin:16px 7px 5px;color:var(--muted);font-size:9.5px}
.statuslegend span{display:flex;align-items:center;gap:4px}
.statuslegend .dot{width:6px;height:6px}

/* ── main column ────────────────────────────────────────────────────────── */
.main{overflow:auto;padding:0 0 60px}
.probehead{padding:16px 20px 12px;border-bottom:1px solid var(--line);background:var(--card)}
.probehead h2{margin:0 0 3px;font-size:18px;letter-spacing:-.01em}
.probehead .claim{color:var(--muted);font-size:13.5px;margin:0}
.statusbadge{border-radius:99px;padding:0 5px;font-size:8px;letter-spacing:.05em;text-transform:uppercase;
 color:var(--muted);border:1px solid var(--line)}
.statusbadge.good{color:var(--good)}.statusbadge.warn{color:var(--warn)}.statusbadge.bad{color:var(--bad)}

/* ── headline tiles: the numbers a metrics-led probe is read for ─────────── */
/* Above the figures, because on a probe whose output IS the numbers, scrolling past
   four plots to reach them buries the result. The full table still sits below with
   the notes and thresholds; this is the primary metrics only. */
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:10px;
 padding:14px 20px 0}
.tile{background:var(--card);border:1px solid var(--line);border-radius:9px;padding:11px 13px}
.tile .lbl{font-size:11px;color:var(--muted);display:block;margin-bottom:5px;
 white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.tile .figure{display:flex;align-items:baseline;gap:7px}
.tile .v{font-size:25px;font-weight:640;letter-spacing:-.02em;
 font-variant-numeric:tabular-nums;line-height:1.1}
.tile .foot{display:flex;align-items:center;gap:7px;margin-top:6px}
.tile .dir{font-size:10px;color:var(--muted);white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
/* The definition travels with the number: these metrics are no longer repeated in the
   table below, so the tile is the only place the reader can learn what they are. */
.tile .tilenote{font-size:11px;line-height:1.5;color:var(--muted);margin-top:7px;
 border-top:1px solid var(--line);padding-top:7px}

/* ── data provenance ────────────────────────────────────────────────────── */
.prov{display:grid;grid-template-columns:auto minmax(0,1fr);gap:3px 12px;font-size:12px}
.prov dt{color:var(--muted)}
.prov dd{margin:0}
.prov .src{font-family:ui-monospace,Menlo,monospace;font-size:11px;word-break:break-all}
.spark{width:44px;height:15px;flex:none}
.d{font-size:11px;font-weight:600}
.d.up{color:var(--good)}.d.down{color:var(--bad)}.d.flat{color:var(--muted)}

/* ── figure bar + viewport ──────────────────────────────────────────────── */
.figbar{display:flex;gap:8px;align-items:center;flex-wrap:wrap;padding:11px 20px;
 border-bottom:1px solid var(--line);position:sticky;top:0;background:var(--bg);z-index:6}
.figbar label{display:flex;align-items:center;gap:6px;font-size:11.5px;color:var(--muted)}
.figbar select{max-width:min(46vw,560px)}
#figsel{font-weight:600;color:var(--fg)}
.figwrap{margin:14px 20px;background:var(--card);border:1px solid var(--line);border-radius:9px;padding:11px}
.figview{overflow:auto;background:var(--panel);border-radius:6px;max-height:78vh;cursor:grab}
.figview.dragging{cursor:grabbing}
.figview img{display:block}
/* Natural size, never upscaled: the attention overlays are 756px wide and stretching
   them across a wide window is both blurry and enormous. */
.figview video{display:block;width:auto;max-width:100%;height:auto;margin:0 auto}
/* The embedded plotly pages are ~860px tall; past that the frame is empty scroll and
   the figure's caption is pushed off the screen. */
.figview iframe{display:block;width:100%;height:min(74vh,880px);border:0}
.figmeta{display:flex;gap:9px;align-items:baseline;margin-top:8px;flex-wrap:wrap}
.figmeta .fn{font-family:ui-monospace,Menlo,monospace;font-size:10.5px;color:var(--muted);word-break:break-all}
/* Every figure carries its own caption and reading note, so a stack of key figures
   is still readable and nothing has to be matched up by position. */
.figguide{margin-top:10px;border-top:1px solid var(--line);padding-top:9px}
.figguide .capline{font-size:14px;font-weight:600;margin:0 0 5px}
.figguide .doc{color:var(--muted)}
.figguide .doc p:last-child{margin-bottom:0}
.badge{background:var(--panel);border-radius:99px;padding:1px 8px;font-size:9.5px;letter-spacing:.06em;
 text-transform:uppercase;color:var(--accent);border:1px solid var(--line)}
.natsize{font-size:10.5px;color:var(--muted);margin-left:auto}

/* ── explanation below the figure ───────────────────────────────────────── */
.below{display:grid;grid-template-columns:repeat(auto-fit,minmax(360px,1fr));gap:14px;margin:0 20px}
.box{background:var(--card);border:1px solid var(--line);border-radius:9px;padding:14px 16px}
.box h4{margin:0 0 8px;font-size:10px;letter-spacing:.09em;text-transform:uppercase;color:var(--muted)}
.box h4+h4{margin-top:18px}
.doc{font-size:12.5px;line-height:1.65}
.doc p{margin:0 0 9px}.doc ul,.doc ol{margin:0 0 9px;padding-left:20px}.doc li{margin-bottom:4px}
.doc code{background:var(--panel);border-radius:3px;padding:0 4px;font-size:11.5px;
 font-family:ui-monospace,Menlo,monospace}
.doc h3{font-size:13px;margin:15px 0 6px}
/* Indented docstring blocks are usually a command line: wrap it rather than hiding
   half of it behind a horizontal scrollbar. */
.doc pre.eq{background:var(--panel);border-radius:6px;padding:9px 11px;overflow-x:auto;font-size:11.5px;
 margin:0 0 9px;font-family:ui-monospace,Menlo,monospace;white-space:pre-wrap;word-break:break-word}
math{font-size:1.08em}
math[display="block"]{display:block;margin:4px 0;overflow-x:auto;text-align:center;font-size:1.2em}
.eqblock{background:var(--panel);border-radius:6px;padding:10px 12px;margin:0 0 11px;overflow-x:auto}
/* A metric's note explains the number; it is not a warning, and colouring every one
   amber made a healthy probe look alarming. */
.note{color:var(--muted);font-size:11.5px;margin:5px 0}
.warnbox{background:var(--panel);border-left:2px solid var(--warn);padding:9px 11px;border-radius:5px;
 font-size:12px;color:var(--muted);margin-bottom:12px}
.refs{display:flex;gap:6px;flex-wrap:wrap;margin-top:8px}
.ref{font-size:11px;color:var(--accent);border:1px solid var(--line);border-radius:99px;padding:2px 9px;
 cursor:pointer;background:var(--panel)}
.ref:hover{border-color:var(--accent)}
.mrow{display:flex;align-items:center;gap:8px;padding:6px 0;border-bottom:1px solid var(--line)}
.mrow .lbl{flex:1;min-width:0;font-size:12px}
.mrow .lbl .dir{display:block;font-size:10.5px;color:var(--muted)}
.mrow .val{font-family:ui-monospace,Menlo,monospace;font-weight:600;font-size:12.5px}
pre.log{background:var(--panel);border:1px solid var(--line);border-radius:6px;padding:9px;font-size:11px;
 line-height:1.45;max-height:300px;overflow:auto;white-space:pre-wrap;word-break:break-word;margin:8px 0 0}
.empty{color:var(--muted);padding:44px;text-align:center}

/* ── overlays ───────────────────────────────────────────────────────────── */
#overlay{position:fixed;inset:0;background:rgba(0,0,0,.55);z-index:50;display:none;padding:5vh 0}
#overlay.on{display:block}
#overlay .sheet{max-width:780px;margin:0 auto;background:var(--card);border:1px solid var(--line);
 border-radius:11px;padding:26px 30px;max-height:90vh;overflow:auto}
#overlay h2{margin:0 0 10px;font-size:19px}
.lightbox{position:fixed;inset:0;background:rgba(0,0,0,.92);z-index:60;display:none;overflow:auto;cursor:zoom-out}
.lightbox.on{display:block}
.lightbox img{display:block;margin:auto}
@media(max-width:1100px){
 .hint{display:none}
 .top .sub{max-width:28ch}
 .right{gap:7px}
 .guidelong{display:none}
 .guideshort{display:inline}
}
@media(max-width:960px){
 .cols{position:relative;grid-template-columns:0 minmax(0,1fr)}
 .cols.norail{grid-template-columns:0 minmax(0,1fr)}
 .rail{position:absolute;inset:0 auto 0 0;z-index:20;width:236px;box-shadow:8px 0 24px rgba(0,0,0,.22)}
 .cols.norail .rail{display:none}
 .main{grid-column:2}
 .top{gap:8px}
 input[type=range]{width:120px}
 .figwrap{margin-left:12px;margin-right:12px}
 .below{margin-left:12px;margin-right:12px}
 .figbar{padding-left:12px;padding-right:12px}
}
@media(max-width:700px){
 .top{align-items:flex-start;flex-wrap:wrap;padding:7px 10px}
 .runblock{width:100%}
 .right{width:100%;margin-left:0;justify-content:flex-end}
 .refreshmsg{display:none}
}
</style></head><body>
<div class="top">
  <div class="runblock"><div class="run" id="runname">loading…</div><div class="sub" id="runsub"></div></div>
  <div class="hint"><kbd>jk</kbd>probe <kbd>[]</kbd>step <kbd>n</kbd>/<kbd>p</kbd>figure <kbd>f</kbd>wide
    <kbd>?</kbd>docs</div>
  <div class="right">
    <button id="railbtn" onclick="toggleRail()" title="Hide the probe list (f)" aria-expanded="true">&#9666;</button>
    <button onclick="openDocs()"><span class="guidelong">Reading this suite</span><span class="guideshort">Guide</span></button>
    <button id="refreshbtn" onclick="reload()" title="Rescan the output directory">Refresh</button>
    <span class="num refreshmsg" id="refreshmsg" aria-live="polite"></span>
    <span class="num" id="steplab"></span>
    <input type="range" id="steprange" min="0" max="0" value="0" aria-label="Validation step">
  </div>
</div>
<div class="cols">
  <div class="rail"><input type="search" id="filter" placeholder="filter probes…" aria-label="Filter probes">
    <div id="list"></div>
    <div class="statuslegend" aria-label="Probe status legend">
      <span><i class="dot good"></i>pass</span><span><i class="dot warn"></i>warn</span>
      <span><i class="dot bad"></i>fail</span><span><i class="dot info"></i>no threshold</span>
    </div>
  </div>
  <div class="main" id="main"></div>
</div>
<div id="overlay" onclick="if(event.target===this)closeDocs()"><div class="sheet">
  <h2>Reading this suite</h2><div class="doc" id="suitedoc"></div></div></div>
<div class="lightbox" id="lightbox" onclick="this.classList.remove('on')"><img id="lightimg" alt=""></div>
<script>
let DATA = null, si = 0, cur = null, pick = null, facet = [], zoom = "auto";
const FACET_THRESHOLD = 12;
let refreshTimer = null, railUserSet = false;
const KEYS = "__keys__";          // dropdown option: every key figure, stacked
const ANY  = "✱any";              // facet option: don't filter on this axis
// Downscaling past this makes axis labels and annotations illegible, so a figure
// that would shrink further opens at 100% in a scrollable viewport instead.
const LEGIBLE_SCALE = 0.55;

// ── LaTeX subset -> MathML ───────────────────────────────────────────────────
const TEX_ID = {alpha:"α",beta:"β",gamma:"γ",delta:"δ",epsilon:"ε",zeta:"ζ",eta:"η",theta:"θ",
 kappa:"κ",lambda:"λ",mu:"μ",nu:"ν",xi:"ξ",pi:"π",rho:"ρ",sigma:"σ",tau:"τ",phi:"φ",chi:"χ",
 psi:"ψ",omega:"ω",Gamma:"Γ",Delta:"Δ",Theta:"Θ",Lambda:"Λ",Pi:"Π",Sigma:"Σ",Phi:"Φ",Psi:"Ψ",
 Omega:"Ω",ell:"ℓ",hbar:"ℏ",varepsilon:"ε",vartheta:"ϑ",varphi:"φ"};
const TEX_OP = {cdot:"⋅",times:"×",div:"÷",pm:"±",mp:"∓",approx:"≈",neq:"≠",le:"≤",ge:"≥",ll:"≪",
 gg:"≫",sim:"∼",propto:"∝",to:"→",rightarrow:"→",mapsto:"↦",infty:"∞",partial:"∂",nabla:"∇",
 sum:"∑",prod:"∏",int:"∫",in:"∈",notin:"∉",subset:"⊂",cup:"∪",cap:"∩",forall:"∀",exists:"∃",
 langle:"⟨",rangle:"⟩",lVert:"‖",rVert:"‖",vert:"|",equiv:"≡",circ:"∘",star:"⋆",ldots:"…",
 cdots:"⋯",leftarrow:"←",Rightarrow:"⇒",implies:"⇒"};
const TEX_FUN = ["log","exp","sin","cos","tan","min","max","argmin","argmax","mean","std","var",
 "det","dim","ker","deg","lim","sup","inf","MSE","RMSE"];
// Alphabet-switching commands. Without these the fallback at the end of texAtom prints
// the command name itself, so `\mathcal{L}` reads "mathcalL" on the page.
const TEX_VARIANT = {mathcal:"script", mathbb:"double-struck", mathbf:"bold", mathsf:"sans-serif",
 boldsymbol:"bold", mathfrak:"fraktur"};
const TEX_ACCENT = {bar:"‾", overline:"‾", hat:"^", tilde:"~", vec:"→", dot:"˙", ddot:"¨"};
const TEX_SPACE = {quad:"1em", qquad:"2em", ",":"0.17em", ";":"0.28em", ":":"0.22em", "!":"-0.17em"};

function texTokens(src){
  const out = []; let i = 0;
  while(i < src.length){
    const c = src[i];
    if(c === "\\"){
      const m = /^\\([a-zA-Z]+|.)/.exec(src.slice(i));
      // \text{...} keeps its spaces, so lift it verbatim before the tokenizer
      // throws whitespace away.
      if(["text","mathrm","operatorname","mathit"].includes(m[1]) && src[i + m[0].length] === "{"){
        let j = i + m[0].length + 1, depth = 1, buf = "";
        while(j < src.length && depth > 0){
          const ch = src[j];
          if(ch === "{") depth++; else if(ch === "}" && --depth === 0){ j++; break; }
          buf += ch; j++;
        }
        out.push({t:"text", v:buf}); i = j; continue;
      }
      out.push({t:"cmd", v:m[1]}); i += m[0].length;
    }
    else if("{}^_".includes(c)){ out.push({t:c}); i++; }
    else if(/\s/.test(c)){ i++; }
    else if(/[0-9]/.test(c)){ const m = /^[0-9]+(\.[0-9]+)?/.exec(src.slice(i)); out.push({t:"num", v:m[0]}); i += m[0].length; }
    else if(/[a-zA-Z]/.test(c)){ out.push({t:"id", v:c}); i++; }
    else { out.push({t:"op", v:c}); i++; }
  }
  return out;
}
function texAtom(tk, tokens){
  if(tk.t === "{") return `<mrow>${texParse(tokens, true)}</mrow>`;
  if(tk.t === "num") return `<mn>${tk.v}</mn>`;
  if(tk.t === "text") return `<mtext>${esc(tk.v)}</mtext>`;
  if(tk.t === "id") return `<mi>${tk.v}</mi>`;
  if(tk.t === "op") return `<mo>${esc(tk.v)}</mo>`;
  if(tk.t === "cmd"){
    const c = tk.v;
    if(c === "frac" || c === "tfrac" || c === "dfrac")
      return `<mfrac>${texArg(tokens)}${texArg(tokens)}</mfrac>`;
    if(c === "sqrt") return `<msqrt>${texArg(tokens)}</msqrt>`;
    if(TEX_VARIANT[c]){
      const arg = texArg(tokens), v = TEX_VARIANT[c];
      // MathML Core wants the variant on the <mi> itself; <mstyle mathvariant> is
      // deprecated. Single letters are the whole use case, so rewrite those directly.
      // texArg wraps a braced argument in <mrow>, so `\mathcal{L}` arrives wrapped.
      const one = /^(?:<mrow>)?<mi>(.)<\/mi>(?:<\/mrow>)?$/.exec(arg);
      return one ? `<mi mathvariant="${v}">${one[1]}</mi>` : `<mstyle mathvariant="${v}">${arg}</mstyle>`;
    }
    if(TEX_ACCENT[c])
      return `<mover accent="true">${texArg(tokens)}<mo>${TEX_ACCENT[c]}</mo></mover>`;
    if(TEX_SPACE[c]) return `<mspace width="${TEX_SPACE[c]}"/>`;
    if(c === "left" || c === "right" || c === "big" || c === "Big"){
      const n = tokens.shift(); if(!n) return "";
      const glyph = n.t === "cmd" ? (TEX_OP[n.v] || n.v) : n.v;
      return glyph === "." ? "" : `<mo stretchy="true">${esc(glyph)}</mo>`;
    }
    if(c === "|") return `<mo stretchy="true">‖</mo>`;
    if(TEX_ID[c]) return `<mi>${TEX_ID[c]}</mi>`;
    if(TEX_OP[c]) return `<mo>${TEX_OP[c]}</mo>`;
    if(TEX_FUN.includes(c)) return `<mi mathvariant="normal">${c}</mi>`;
    return `<mi>${esc(c)}</mi>`;
  }
  return "";
}
function texArg(tokens){ const tk = tokens.shift(); return tk ? texAtom(tk, tokens) : "<mrow/>"; }
function texParse(tokens, untilBrace){
  let out = "";
  while(tokens.length){
    const tk = tokens.shift();
    if(tk.t === "}"){ if(untilBrace) break; continue; }
    let node = texAtom(tk, tokens), sub = null, sup = null;
    while(tokens.length && (tokens[0].t === "^" || tokens[0].t === "_")){
      const kind = tokens.shift().t;
      if(kind === "^") sup = texArg(tokens); else sub = texArg(tokens);
    }
    if(sub && sup) node = `<msubsup>${node}${sub}${sup}</msubsup>`;
    else if(sup) node = `<msup>${node}${sup}</msup>`;
    else if(sub) node = `<msub>${node}${sub}</msub>`;
    out += node;
  }
  return out;
}
function mathml(tex, display){
  try {
    return `<math xmlns="http://www.w3.org/1998/Math/MathML" display="${display?"block":"inline"}">` +
           `<mrow>${texParse(texTokens(tex), false)}</mrow></math>`;
  } catch(e){ return `<code>${esc(tex)}</code>`; }
}

// ── tiny markdown ────────────────────────────────────────────────────────────
function esc(s){ return String(s).replace(/[&<>]/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;"}[c])); }
function attr(s){ return esc(s).replace(/"/g, "&quot;"); }
function inline(s){
  // U+E000 is private-use: it cannot occur in a docstring, survives escaping, and
  // can't be confused with a number that happens to be in the prose.
  const held = [];
  const hold = html => "" + (held.push(html) - 1) + "";
  s = String(s)
    .replace(/\$\$([\s\S]+?)\$\$/g, (_, t) => hold(mathml(t, true)))
    .replace(/\$([^$\n]+)\$/g, (_, t) => hold(mathml(t, false)));
  return esc(s)
    .replace(/``([^`]+)``/g, "<code>$1</code>").replace(/`([^`]+)`/g, "<code>$1</code>")
    .replace(/\*\*(.+?)\*\*/g, "<b>$1</b>").replace(/\*([^*\s][^*]*)\*/g, "<i>$1</i>")
    .replace(/(\d+)/g, (_, i) => held[+i]);
}
function md(text){
  if(!text) return "";
  return text.split(/\n\s*\n/).map(block => {
    const raw = block.split("\n").filter(l => l.trim());
    const lines = block.split("\n").map(l => l.trim());
    if(lines[0].startsWith("## ")) return `<h3>${inline(lines[0].slice(3))}</h3>` +
      (lines.length > 1 ? `<p>${inline(lines.slice(1).join(" "))}</p>` : "");
    // Blocks are already split on blank lines, so a block that opens with a marker is a
    // list and every later line is either a new item or the wrapped tail of the last one.
    // Testing the trimmed lines instead lost every list whose items wrap.
    const list = (marker, tag) => {
      const start = raw.findIndex(l => marker.test(l));
      if(start < 0) return "";
      const items = [];
      raw.slice(start).forEach(l => {
        const m = marker.exec(l);
        if(m) items.push(m[1]); else if(items.length) items[items.length-1] += " " + l.trim();
      });
      // A marker part-way down only starts a list if it really is one; a lone dash in
      // running prose is not.
      if(start > 0 && items.length < 2) return "";
      const lead = start ? `<p>${inline(raw.slice(0, start).join(" ").trim())}</p>` : "";
      return `${lead}<${tag}>${items.map(i => `<li>${inline(i)}</li>`).join("")}</${tag}>`;
    };
    const bulleted = list(/^\s*[-*]\s+(.*)/, "ul") || list(/^\s*\d+\.\s+(.*)/, "ol");
    if(bulleted) return bulleted;
    // An indented run is a command, an aligned table or an equation — never prose. It
    // often follows a lead line ("Outputs under <dir>/:"), which stays a paragraph.
    // Reflowing those into prose is how a file listing became one unreadable sentence.
    const indent = raw.findIndex(l => /^\s\s/.test(l));
    if(indent >= 0 && raw.slice(indent).every(l => /^\s\s/.test(l))){
      const body = raw.slice(indent);
      const strip = Math.min(...body.map(l => l.match(/^\s*/)[0].length));
      const text = body.map(l => l.slice(strip)).join("\n");
      const lead = indent ? `<p>${inline(raw.slice(0, indent).join(" ").trim())}</p>` : "";
      return lead + (/\$/.test(text) ? `<div class="eqblock">${inline(text)}</div>`
                                     : `<pre class="eq">${esc(text)}</pre>`);
    }
    // Probe docstrings list emitted files as "name.mp4 — what it is", one per line
    // and without bullets. Joining those into a paragraph is unreadable.
    const dashed = lines.filter(l => /\s(—|--)\s/.test(l));
    if(dashed.length >= 2 && dashed.length >= lines.length - 1){
      const items = []; lines.forEach(l => {
        if(/\s(—|--)\s/.test(l)) items.push(l);
        else if(items.length) items[items.length-1] += " " + l; else items.push(l); });
      return `<ul>${items.map(i => `<li>${inline(i)}</li>`).join("")}</ul>`;
    }
    return `<p>${inline(lines.join(" "))}</p>`;
  }).join("");
}

// ── metrics ──────────────────────────────────────────────────────────────────
function stepKey(i){ return String(DATA.steps[i].step); }
function fmt(v, d){ return v === null || v === undefined ? "—" : (d === 0 ? String(Math.round(v)) : v.toFixed(d)); }
function presentSteps(m){ return DATA.steps.map((s, i) => i)
  .filter(i => m.values[stepKey(i)] !== undefined && m.values[stepKey(i)] !== null); }
function delta(m){
  const idx = presentSteps(m).filter(i => i <= si);
  if(idx.length < 2) return null;
  const d = m.values[stepKey(idx[idx.length-1])] - m.values[stepKey(idx[idx.length-2])];
  if(Math.abs(d) < 1e-12) return {d:0, dir:"flat"};
  return {d, dir: m.good === "none" ? "flat" : ((d > 0) === (m.good === "high") ? "up" : "down")};
}
function spark(m){
  const idx = presentSteps(m);
  if(idx.length < 2) return `<span class="spark"></span>`;
  const vals = idx.map(i => m.values[stepKey(i)]);
  const extra = (m.baseline === null || m.baseline === undefined) ? [] : [m.baseline];
  const lo = Math.min(...vals, ...extra), hi = Math.max(...vals, ...extra), rg = (hi - lo) || 1;
  const W = 44, H = 15, X = i => (i/(idx.length-1))*(W-4)+2, Y = v => H-2-((v-lo)/rg)*(H-6);
  const d = vals.map((v, i) => (i?"L":"M") + X(i).toFixed(1) + " " + Y(v).toFixed(1)).join(" ");
  const at = idx.indexOf(idx.filter(i => i <= si).pop());
  const base = extra.length ? `<line x1="0" y1="${Y(m.baseline).toFixed(1)}" x2="${W}"
    y2="${Y(m.baseline).toFixed(1)}" stroke="var(--muted)" stroke-width="1" stroke-dasharray="2 2"
    opacity=".6"/>` : "";
  const head = at >= 0 ? `<circle cx="${X(at).toFixed(1)}" cy="${Y(vals[at]).toFixed(1)}" r="2.3"
    fill="var(--accent)"/>` : "";
  return `<svg class="spark" viewBox="0 0 ${W} ${H}">${base}<path d="${d}" fill="none"
    stroke="var(--muted)" stroke-width="1.4"/>${head}</svg>`;
}
function dirLabel(m){
  if(m.good === "none") return "descriptive";
  const th = [];
  if(m.warn !== null && m.warn !== undefined) th.push("warn " + m.warn);
  if(m.bad !== null && m.bad !== undefined) th.push("bad " + m.bad);
  if(m.baseline !== null && m.baseline !== undefined) th.push("baseline " + m.baseline);
  return (m.good === "high" ? "higher is better" : "lower is better") +
         (th.length ? " · " + th.join(", ") : "");
}
function statusLabel(s){ return ({good:"pass", warn:"warn", bad:"fail", info:"no threshold"})[s] || "no threshold"; }
function orderedMetrics(p){ return [...p.metrics].sort((a, b) => (b.primary ? 1 : 0) - (a.primary ? 1 : 0)); }

// Primary metrics as stat tiles above the figures. Label, value, delta against the
// previous step that has one, sparkline across steps, threshold status.
function headlineTiles(p){
  const primary = p.metrics.filter(m => m.primary);
  if(!primary.length) return "";
  return `<div class="tiles">${primary.map(m => {
    const v = m.values[stepKey(si)], d = delta(m);
    const status = m.statuses ? (m.statuses[stepKey(si)] || "info") : "info";
    return `<div class="tile">
      <span class="lbl" title="${esc(m.label)}">${esc(m.label)}</span>
      <div class="figure"><span class="v">${fmt(v, m.fmt)}</span>
        <span class="d ${d?d.dir:"flat"}">${d && d.d ? (d.d>0?"+":"")+fmt(d.d, m.fmt) : ""}</span></div>
      <div class="foot">${spark(m)}
        <span class="statusbadge ${status}">${statusLabel(status)}</span>
        <span class="dir" title="${esc(dirLabel(m))}">${esc(dirLabel(m))}</span></div>
      ${m.note ? `<div class="tilenote">${inline(m.note)}</div>` : ""}
    </div>`;
  }).join("")}</div>`;
}

// Where the numbers came from. Rendered from index.json `extra.provenance`, so any
// probe gains this section by declaring one — nothing here is objective-specific.
function splitLine(s){
  if(!s) return "";
  const src = (s.sources || []).map(x =>
    `<div class="src">${esc(x.name)} · ${x.n_episodes} ep ${JSON.stringify(x.episodes||[])}
     · ${x.n_frames} frames<br>${esc(x.root || "")}</div>`).join("");
  return `<b>${s.n_frames} frames</b> over ${s.n_episodes} episodes${src}`;
}
function provenanceBox(p){
  const prov = (p.extra && p.extra[stepKey(si)] || {}).provenance;
  if(!prov) return "";
  const rows = [
    ["Held out", splitLine(prov.val)],
    ["Training", splitLine(prov.train)],
    ["Per episode", `${prov.frames_per_episode} frames, evenly spaced` +
      (prov.image_stride ? `, snapped onto the stride-${prov.image_stride} grid` : "")],
    ["Episode budget", prov.episode_budget === null || prov.episode_budget === undefined
      ? "every episode" : `${prov.episode_budget}, divided across the training sources`],
    ["Per forward", `batch ${prov.batch_size}, ${prov.timesteps_per_forward} flow timesteps,` +
      ` chunk ${prov.chunk_size}`],
    ["Total", `${prov.forwards} forwards`],
  ].filter(r => r[1]);
  return `<div class="box"><h4>Data behind these numbers</h4>
    <dl class="prov">${rows.map(([k, v]) => `<dt>${esc(k)}</dt><dd>${v}</dd>`).join("")}</dl>
    ${prov.sampling ? `<div class="note" style="margin-top:9px">${inline(prov.sampling)}</div>` : ""}
  </div>`;
}
function refChips(ids){
  if(!ids || !ids.length) return "";
  return `<div class="refs">${ids.map(id => {
    const other = DATA.probes.find(x => x.id === id);
    return other ? `<button type="button" class="ref" onclick="sel('${id}')">→ ${esc(other.title)}</button>`
                 : `<span class="ref" style="opacity:.5">→ ${esc(id)} (not in this run)</span>`;
  }).join("")}</div>`;
}

// ── facets ───────────────────────────────────────────────────────────────────
function facetLabel(v){ return v === ANY ? "everything" : (v === "" ? "top level" : v); }
function natcmp(a, b){ return String(a).localeCompare(String(b), undefined, {numeric:true}); }
function dirOf(file){ const i = file.lastIndexOf("/"); return i < 0 ? "" : file.slice(0, i); }
function components(dir){ return dir.split("/").flatMap(seg => seg.split("_")); }
function alphaPrefix(part){ const m = String(part).match(/^[a-zA-Z]+/); return m ? m[0] : ""; }
function facetPlan(panels){
  const dirs = [...new Set(panels.map(pn => dirOf(pn.file)))];
  if(dirs.length < 2) return {axes: [], dirs};
  const split = dirs.map(components);
  const width = split[0].length;
  const uniform = split.every(p => p.length === width &&
    p.every((part, i) => alphaPrefix(part) === alphaPrefix(split[0][i])));
  if(!uniform) return {axes:[{label:"view", options:[ANY, ...dirs.slice().sort(natcmp)]}], dirs, whole:true};
  const axes = [];
  for(let i = 0; i < width; i++){
    const options = [...new Set(split.map(p => p[i]))].sort(natcmp);
    if(options.length > 1) axes.push({label: alphaPrefix(split[0][i]) || `part ${i+1}`, options, index:i});
  }
  return {axes, dirs, split};
}
function matchesFacet(pn, plan){
  if(!plan.axes.length) return true;
  const dir = dirOf(pn.file);
  if(plan.whole) return facet[0] === ANY || dir === facet[0];
  const parts = components(dir);
  return plan.axes.every((axis, a) => facet[a] === ANY || parts[axis.index] === facet[a]);
}
function normalizeFacet(plan){
  plan.axes.forEach((axis, a) => { if(!axis.options.includes(facet[a])) facet[a] = axis.options[0]; });
  facet.length = plan.axes.length;
}
// Choosing a figure wins over the selectors: stepping with n/p, or landing on a probe
// whose key figure lives in another episode, otherwise filtered the chosen figure out
// and the page claimed the probe had written nothing.
function snapFacet(pn, plan){
  if(!pn || !plan.axes.length) return;
  const dir = dirOf(pn.file);
  if(plan.whole){ facet[0] = plan.axes[0].options.includes(dir) ? dir : ANY; return; }
  const parts = components(dir);
  plan.axes.forEach((axis, a) => { if(parts[axis.index] !== undefined) facet[a] = parts[axis.index]; });
}

// ── state helpers ────────────────────────────────────────────────────────────
function probe(){ return DATA.probes.find(x => x.id === cur) || null; }
function probesAt(){ return DATA.probes.filter(p => p.panels[stepKey(si)] !== undefined); }
function allPanels(){ const p = probe(); return p ? (p.panels[stepKey(si)] || []) : []; }
function keyPanels(){ const k = allPanels().filter(pn => pn.primary); return k.length ? k : allPanels(); }
function shownPanels(){
  const all = allPanels();
  if(pick === KEYS) return keyPanels();
  const hit = all.find(pn => pn.file === pick);
  return hit ? [hit] : keyPanels().slice(0, 1);
}

// ── rendering ────────────────────────────────────────────────────────────────
function drawList(){
  const q = document.getElementById("filter").value.toLowerCase();
  const here = probesAt();
  document.getElementById("list").innerHTML = DATA.groups.map(g => {
    const items = here.filter(p => p.group === g &&
      (!q || (p.title + p.id + p.claim).toLowerCase().includes(q)));
    if(!items.length) return "";
    return `<div class="g">${esc(g)}</div>` + items.map(p => {
      const m = orderedMetrics(p)[0], v = m ? m.values[stepKey(si)] : undefined;
      const status = p.status[stepKey(si)] || "info";
      const shown = m && v !== undefined && v !== null ? fmt(v, m.fmt) : "";
      const tip = `${p.title} — ${statusLabel(status)}${m && shown ? ` · ${m.label}: ${shown}` : ""}`;
      return `<button type="button" class="item ${p.id===cur?"on":""}" onclick="sel('${p.id}')"
        title="${attr(tip)}" aria-current="${p.id===cur?"true":"false"}"><span class="dot ${status}"></span>
        <span class="nm">${esc(p.title)}</span><span class="num">${shown}</span></button>`;
    }).join("");
  }).join("") || `<div class="empty">no probes at this step</div>`;
}

function figureSelect(plan){
  const all = allPanels(), keyed = allPanels().filter(pn => pn.primary);
  if(!all.length) return `<span class="num">No figures</span>`;
  // One figure needs no menu: its caption is printed under it like every other.
  if(all.length === 1) return `<span class="num">1 figure</span>`;
  const inFacet = pn => matchesFacet(pn, plan);
  const opt = pn => `<option value="${esc(pn.file)}" ${pick===pn.file?"selected":""}>${
    esc(pn.caption && pn.caption !== pn.file ? pn.caption : pn.file)}</option>`;
  const keyOpts = keyed.filter(inFacet), restOpts = all.filter(pn => !pn.primary && inFacet(pn));
  return `<label>Figure<select id="figsel" onchange="setPick(this.value)">
    ${keyed.length > 1 ? `<option value="${KEYS}" ${pick===KEYS?"selected":""}>All key figures (${keyed.length})</option>` : ""}
    ${keyOpts.length ? `<optgroup label="Key figures">${keyOpts.map(opt).join("")}</optgroup>` : ""}
    ${restOpts.length ? `<optgroup label="Everything else (${restOpts.length})">${restOpts.map(opt).join("")}</optgroup>` : ""}
  </select></label>`;
}

function figureHtml(pn){
  if(pn.kind === "image")
    return `<div class="figview" data-pan="1"><img src="${pn.url}" alt="${esc(pn.caption)}"
      onload="fitImage(this)" ondblclick="light('${pn.url}')"></div>`;
  if(pn.kind === "video")
    return `<div class="figview"><video controls preload="metadata" playsinline src="${pn.url}"></video></div>`;
  if(pn.kind === "html")
    return `<div class="figview"><iframe src="${pn.url}" loading="lazy"></iframe></div>`;
  return `<div class="figview"><div class="empty">${esc(pn.file)} —
    <a href="${pn.url}" target="_blank">open</a></div></div>`;
}

// A figure only gets scaled down while it stays readable; past that it opens at
// 100% and you scroll, which is how you actually read a 7000px figure.
function fitImage(img){
  const wrap = img.parentElement;
  const avail = wrap.clientWidth;
  const nat = img.naturalWidth || avail;
  let scale;
  if(zoom === "auto") scale = (avail / nat >= LEGIBLE_SCALE) ? Math.min(1, avail / nat) : 1;
  else if(zoom === "fit") scale = Math.min(1, avail / nat);
  else scale = parseFloat(zoom);
  img.style.width = Math.round(nat * scale) + "px";
  img.style.maxWidth = "none";
  const tag = wrap.parentElement.querySelector(".natsize");
  if(tag) tag.textContent = `${nat}×${img.naturalHeight} native · ${Math.round(scale*100)}%` +
    (scale < 1 ? " · double-click for full size" : "");
}

function drawMain(){
  const main = document.getElementById("main"), p = probe();
  if(!p){ main.innerHTML = `<div class="empty">select a probe</div>`; return; }

  const all = allPanels();
  const plan = all.length > FACET_THRESHOLD ? facetPlan(all) : {axes: []};
  normalizeFacet(plan);
  if(pick !== KEYS && !all.some(pn => pn.file === pick)){
    const keyed = keyPanels();
    pick = keyed.length > 1 && all.filter(pn => pn.primary).length > 1 ? KEYS
         : (keyed[0] ? keyed[0].file : null);
  }
  snapFacet(all.find(pn => pn.file === pick), plan);
  const shown = shownPanels().filter(pn => matchesFacet(pn, plan));

  const facets = plan.axes.map((axis, a) => `<label>${esc(axis.label)}
    <select onchange="setFacet(${a}, this.value)">${axis.options.map(o =>
      `<option ${o===facet[a]?"selected":""} value="${esc(o)}">${esc(facetLabel(o))}</option>`
      ).join("")}</select></label>`).join("");

  const figures = shown.length ? shown.map(pn => `<div class="figwrap">
      ${figureHtml(pn)}
      <div class="figguide">
        <p class="capline">${inline(pn.caption)}</p>
        ${pn.how ? `<div class="doc">${md(pn.how)}</div>`
                 : `<p class="note">No reading note for this figure — what the probe measures
                    is described below.</p>`}
        ${refChips(pn.refs)}
      </div>
      <div class="figmeta">${pn.primary ? `<span class="badge">key</span>` : ""}
        <span class="num">figure ${all.indexOf(pn) + 1} of ${all.length}</span>
        <span class="fn">${esc(pn.file)}</span><span class="natsize"></span></div>
    </div>`).join("")
    : `<div class="figwrap"><div class="empty">${all.length
        ? "no figure matches the selectors above — widen one of them"
        : "this probe wrote no figures at this step" + (p.log[stepKey(si)] ? " — its log is below" : "")
      }</div></div>`;

  // Only what the tiles above don't already carry — a metric shown twice on one page
  // reads as two different numbers.
  const tabled = orderedMetrics(p).filter(m => !m.primary);
  const metricsBox = tabled.length ? `<div class="box">
    <h4>${p.metrics.some(m => m.primary) ? "Supporting metrics" : "Metrics"} at step
      ${DATA.steps[si].step.toLocaleString()}</h4>` +
      tabled.map(m => { const v = m.values[stepKey(si)], d = delta(m);
        const status = m.statuses ? (m.statuses[stepKey(si)] || "info") : "info";
        return `<div class="mrow"><span class="lbl">${esc(m.label)}
          <span class="dir">${esc(dirLabel(m))}</span></span>${spark(m)}
          <span class="val">${fmt(v, m.fmt)}</span>
          <span class="statusbadge ${status}">${statusLabel(status)}</span>
          <span class="d ${d?d.dir:"flat"}">${d && d.d ? (d.d>0?"+":"")+fmt(d.d, m.fmt) : "—"}</span></div>
          ${m.note ? `<div class="note">${inline(m.note)}</div>` : ""}${refChips(m.refs)}`;
      }).join("") + `</div>` : "";

  const docBox = `<div class="box">
    ${p.has_manifest ? "" : `<div class="warnbox">No <code>index.json</code> — every file in the
      directory is listed, documentation is the module docstring read off disk, and the numbers are
      raw summary keys with no labels, direction or thresholds.</div>`}
    <h4>What this probe measures</h4>
    <div class="doc">${md(p.doc) || "<p>no module docstring</p>"}</div>
    ${p.see_also && p.see_also.length ? `<h4>Read alongside</h4>${refChips(p.see_also)}` : ""}
    ${p.log[stepKey(si)] ? `<h4>Probe log</h4>
      <button onclick="showLog('${p.log[stepKey(si)]}')">Load probe.log</button>
      <pre class="log" id="logbox"></pre>` : ""}
  </div>`;

  const canZoom = shown.some(pn => pn.kind === "image");
  main.innerHTML = `
    <div class="probehead"><h2>${esc(p.title)}</h2>
      ${p.claim ? `<p class="claim">${inline(p.claim)}</p>` : ""}</div>
    <div class="figbar">${figureSelect(plan)}${facets}
      ${canZoom ? `<label>Zoom<select onchange="setZoom(this.value)">
        <option value="auto" ${zoom==="auto"?"selected":""}>auto</option>
        <option value="fit" ${zoom==="fit"?"selected":""}>fit width</option>
        <option value="0.5" ${zoom==="0.5"?"selected":""}>50%</option>
        <option value="1" ${zoom==="1"?"selected":""}>100%</option>
        <option value="2" ${zoom==="2"?"selected":""}>200%</option></select></label>` : ""}
      ${all.length > 1 ? `<span class="num">${all.length} figures in this probe</span>` : ""}</div>
    ${headlineTiles(p)}
    ${figures}
    <div class="below">${provenanceBox(p)}${metricsBox}${docBox}</div>`;
  wirePanning();
}

function wirePanning(){
  document.querySelectorAll('.figview[data-pan]').forEach(view => {
    let down = false, x0 = 0, y0 = 0, l0 = 0, t0 = 0;
    view.addEventListener("pointerdown", e => {
      down = true; x0 = e.clientX; y0 = e.clientY; l0 = view.scrollLeft; t0 = view.scrollTop;
      view.classList.add("dragging"); view.setPointerCapture(e.pointerId);
    });
    view.addEventListener("pointermove", e => {
      if(!down) return;
      view.scrollLeft = l0 - (e.clientX - x0); view.scrollTop = t0 - (e.clientY - y0);
    });
    const stop = () => { down = false; view.classList.remove("dragging"); };
    view.addEventListener("pointerup", stop); view.addEventListener("pointercancel", stop);
  });
}
function light(url){
  const img = document.getElementById("lightimg");
  img.style.width = "auto"; img.src = url;
  document.getElementById("lightbox").classList.add("on");
}
async function showLog(url){
  const box = document.getElementById("logbox");
  box.textContent = "loading…";
  const text = await (await fetch(url)).text();
  box.textContent = text.length > 20000 ? "…" + text.slice(-20000) : text;
}

function draw(){ drawList(); drawMain();
  document.getElementById("steplab").textContent = "step " + DATA.steps[si].step.toLocaleString(); }
function sel(id){ cur = id; pick = null; facet = []; draw(); document.getElementById("main").scrollTop = 0; }
function setPick(v){ pick = v; drawMain(); }
function setFacet(a, v){ facet[a] = v; facet.length = a + 1; pick = null; drawMain(); }
function setZoom(v){ zoom = v; document.querySelectorAll(".figview img").forEach(fitImage); }
function stepFigure(delta){
  const all = allPanels(); if(!all.length) return;
  const order = [...all.filter(pn => pn.primary), ...all.filter(pn => !pn.primary)];
  const at = order.findIndex(pn => pn.file === pick);
  setPick(order[Math.max(0, Math.min(order.length-1, at + delta))].file);
}
function updateRailButton(){
  const hidden = document.querySelector(".cols").classList.contains("norail");
  const button = document.getElementById("railbtn");
  button.innerHTML = hidden ? "&#9656;" : "&#9666;";
  button.title = (hidden ? "Show" : "Hide") + " the probe list (f)";
  button.setAttribute("aria-expanded", String(!hidden));
}
function syncResponsiveRail(){
  if(!railUserSet) document.querySelector(".cols").classList.toggle("norail", innerWidth <= 960);
  updateRailButton();
}
function toggleRail(){
  railUserSet = true; document.querySelector(".cols").classList.toggle("norail"); updateRailButton();
}
function openDocs(){ document.getElementById("overlay").classList.add("on"); }
function closeDocs(){ document.getElementById("overlay").classList.remove("on"); }
function flashRefresh(message){
  const label = document.getElementById("refreshmsg");
  label.textContent = message; clearTimeout(refreshTimer);
  refreshTimer = setTimeout(() => { label.textContent = ""; }, 2400);
}

async function reload(){
  const keep = cur, previous = DATA;
  const selectedStep = previous && previous.steps[si] ? previous.steps[si].step : null;
  const wasLatest = !previous || si === previous.steps.length - 1;
  const button = document.getElementById("refreshbtn");
  button.disabled = true; button.textContent = "Refreshing…";
  try {
    const response = await fetch("/api/index");
    if(!response.ok) throw new Error(`index request failed: ${response.status}`);
    DATA = await response.json();
    document.getElementById("runname").textContent = DATA.run.name;
    document.getElementById("runname").title = DATA.run.name;
    document.getElementById("runsub").textContent = DATA.run.val_dir;
    document.getElementById("runsub").title = DATA.run.val_dir;
    document.getElementById("suitedoc").innerHTML = md(DATA.suite_doc);
    const range = document.getElementById("steprange");
    if(!DATA.steps.length){
      range.hidden = true; document.getElementById("steplab").textContent = "";
      document.getElementById("list").innerHTML = "";
      document.getElementById("main").innerHTML = `<div class="empty">No step_* directories under
        ${esc(DATA.run.val_dir)} yet.</div>`;
      flashRefresh("No steps yet"); return;
    }
    range.max = DATA.steps.length - 1;
    range.hidden = DATA.steps.length <= 1;
    if(wasLatest) si = DATA.steps.length - 1;
    else {
      const same = DATA.steps.findIndex(s => s.step === selectedStep);
      si = same >= 0 ? same : Math.min(si, DATA.steps.length - 1);
    }
    range.value = si;
    const here = probesAt();
    cur = (keep && here.some(p => p.id === keep)) ? keep : (here[0] ? here[0].id : null);
    pick = null; draw();
    const added = previous ? Math.max(0, DATA.steps.length - previous.steps.length) : 0;
    flashRefresh(added ? `${added} new step${added === 1 ? "" : "s"}` : (previous ? "Up to date" : ""));
  } catch(error) {
    console.error(error); flashRefresh("Refresh failed");
    if(!DATA) document.getElementById("main").innerHTML = `<div class="empty">Could not load probe output.</div>`;
  } finally {
    button.disabled = false; button.textContent = "Refresh";
  }
}

document.getElementById("steprange").oninput = e => { si = +e.target.value; draw(); };
document.getElementById("filter").oninput = drawList;
addEventListener("resize", () => {
  syncResponsiveRail(); document.querySelectorAll(".figview img").forEach(fitImage);
});
addEventListener("keydown", e => {
  if(e.key === "Escape"){ closeDocs(); document.getElementById("lightbox").classList.remove("on"); }
  if(e.target.tagName === "INPUT" || e.target.tagName === "SELECT"){
    if(e.key === "Escape") e.target.blur();
    return;
  }
  const here = probesAt(), i = here.findIndex(x => x.id === cur);
  if(e.key === "j" && here.length) sel(here[Math.min(here.length-1, i+1)].id);
  if(e.key === "k" && here.length) sel(here[Math.max(0, i-1)].id);
  if(e.key === "n") stepFigure(1);
  if(e.key === "p") stepFigure(-1);
  if(e.key === "]"){ si = Math.min(DATA.steps.length-1, si+1);
    document.getElementById("steprange").value = si; draw(); }
  if(e.key === "["){ si = Math.max(0, si-1); document.getElementById("steprange").value = si; draw(); }
  if(e.key === "f") toggleRail();
  if(e.key === "?") openDocs();
  if(e.key === "/"){ e.preventDefault(); document.getElementById("filter").focus(); }
});
syncResponsiveRail();
reload();
</script></body></html>
"""


def _make_server(val_dir: Path, host: str, port: int, retries: int = 20) -> ThreadingHTTPServer:
    handler = make_handler(val_dir)
    last_error: OSError | None = None
    for candidate in range(port, port + retries + 1):
        try:
            return ThreadingHTTPServer((host, candidate), handler)
        except OSError as exc:
            last_error = exc
            if port == 0:
                break
    raise RuntimeError(f"Could not bind probe viewer server: {last_error}") from last_error


def view_probes(
    run_dir: str | Path,
    host: str = "127.0.0.1",
    port: int = 7870,
    open_browser: bool = True,
) -> None:
    """Launch the probe viewer and block until interrupted."""
    val_dir = resolve_validation_dir(run_dir)
    server = _make_server(val_dir, host, port)
    url = f"http://{server.server_address[0]}:{server.server_address[1]}/"
    index = build_index(val_dir)
    missing = [p["id"] for p in index["probes"] if not p["has_manifest"]]
    print(f"Probe viewer: {url}")
    print(f"Serving: {val_dir}")
    print(f"{len(index['steps'])} step(s), {len(index['probes'])} probe(s)")
    if missing:
        print(f"Without index.json (raw file listing only): {', '.join(missing)}")
    if open_browser:
        threading.Timer(0.35, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping probe viewer.")
    finally:
        server.server_close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Browse probe output from a training run.")
    parser.add_argument("run_dir", help="Training run directory (or its validation/ or a step_* dir).")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7870)
    parser.add_argument("--no-open", action="store_true", help="Do not open a browser tab.")
    args = parser.parse_args()
    view_probes(args.run_dir, host=args.host, port=args.port, open_browser=not args.no_open)


if __name__ == "__main__":
    main()
