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

"""Compare probe output from one to four runs or checkpoints in a browser.

Examples::

    python -m lerobot.scripts.compare_probes outputs/ckpt400 outputs/ckpt800 outputs/ckpt1200
    python -m lerobot.scripts.compare_probes outputs/run/validation/step_00000400 \
        outputs/run/validation/step_00000800 --label step-400 --label step-800

The existing :mod:`lerobot.scripts.view_probes` remains the detailed single-run
viewer. This companion compares the selected checkpoint of each run side by side,
matching figures by their exact probe-relative filename. With a fixed validation set, paths such as
``ep0000_L14/overlay_img_top_summary.mp4`` identify the same sampled view. The page
(``_compare_probes/app.js``) parses those filenames into axes, one chip row each, so a
probe with hundreds of outputs (attention: episode x layer x camera x view) is navigated
by clicking a value on one axis while the others stay put; a wall mode fans the camera
axis out per run. Probes the suite no longer reads are hidden from the rail by the
``HIDDEN`` pattern in ``app.js``; their output is still served.

Probes that pick their examples by percentile break that assumption: the band is
fixed but the frame it lands on is not, so the same figure is called
``examples/frame_p10_ep0002_fr004248.png`` in one run and
``examples/frame_p10_ep0002_fr001632.png`` in the next, and under exact-name
matching it is unique to its run and hidden by the shared-files filter. Those
panels carry an ``align`` key naming the band instead; older manifests, written
before the key existed, get the same key derived from the filename.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import threading
import webbrowser
from collections import Counter
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

from lerobot.scripts.view_probes import _json_response, _send_file, build_index, resolve_validation_dir

MAX_RUNS = 4

# ``frame_p10_ep0002_fr004248.png`` — the trailing episode and frame are what makes an
# otherwise identical panel unique to its run.
SAMPLE_SUFFIX = re.compile(r"_ep\d+_fr\d+(?=\.[^.]+$)")


@dataclass(frozen=True)
class CompareSource:
    """A validation tree and the column metadata derived from one CLI input."""

    input_path: Path
    val_dir: Path
    label: str
    initial_step: int | None


def _step_number(path: Path) -> int | None:
    if not path.name.startswith("step_"):
        return None
    suffix = path.name.rsplit("_", 1)[-1]
    return int(suffix) if suffix.isdigit() else None


def resolve_compare_sources(
    run_dirs: list[str | Path], labels: list[str] | None = None
) -> list[CompareSource]:
    """Resolve one to four inputs while retaining an explicitly selected step."""
    if not 1 <= len(run_dirs) <= MAX_RUNS:
        raise ValueError(f"Expected between 1 and {MAX_RUNS} run paths, got {len(run_dirs)}.")
    labels = labels or []
    if len(labels) > len(run_dirs):
        raise ValueError(f"Got {len(labels)} labels for {len(run_dirs)} run paths.")

    sources = []
    for index, raw in enumerate(run_dirs):
        input_path = Path(raw).expanduser().resolve()
        val_dir = resolve_validation_dir(input_path)
        requested_step = _step_number(input_path)
        default_label = (
            f"{val_dir.parent.name} · {input_path.name}"
            if requested_step is not None
            else val_dir.parent.name
        )
        label = labels[index].strip() if index < len(labels) and labels[index].strip() else default_label
        sources.append(
            CompareSource(
                input_path=input_path,
                val_dir=val_dir,
                label=label,
                initial_step=requested_step,
            )
        )
    return sources


def _alignment_keys(panels: list[dict]) -> dict[str, str]:
    """Map each panel file to the name it answers to in every run.

    A probe states the key itself with ``Panel(align=...)``. Manifests written before
    that field existed get the same key by dropping the episode/frame suffix, which is
    only sound when it still names one figure: probes that write several examples per
    band collapse onto one key, and a key that names two panels in the same run names
    nothing, so those keep their exact filename and stay unaligned as before.
    """
    derived = {
        panel["file"]: panel.get("align") or SAMPLE_SUFFIX.sub("", panel["file"])
        for panel in panels
    }
    counts = Counter(derived.values())
    return {file: key if counts[key] == 1 else file for file, key in derived.items()}


def _interactive_samples(
    probe_dir: Path, probe_id: str, panel_file: str
) -> tuple[str, list[dict]] | None:
    """Return the internal frame order for inspectors the comparison UI can coordinate."""
    if probe_id == "action_trace" and panel_file == "action_trace.html":
        metrics_path = probe_dir / "metrics.csv"
        if not metrics_path.is_file():
            return None
        with metrics_path.open(newline="") as file:
            rows = list(csv.DictReader(file))
        samples = [
            {
                "key": f"{row['episode']}:{row['frame']}",
                "label": f"episode {row['episode']} · frame {row['frame']}",
                "episode": int(row["episode"]),
                "frame": int(row["frame"]),
                "global_idx": int(row["global_idx"]),
            }
            for row in rows
        ]
        return "frame", samples

    if probe_id == "objective" and panel_file == "action_exemplars.html":
        summary_path = probe_dir / "objective.json"
        if not summary_path.is_file():
            return None
        try:
            with summary_path.open() as file:
                summary = json.load(file)
        except (OSError, ValueError):
            return None
        samples = []
        for band in summary.get("exemplars", []):
            for row in band.get("frames", []):
                samples.append(
                    {
                        "key": f"{row['episode_idx']}:{row['frame_idx']}",
                        "label": (
                            f"p{band['percentile']} {band['label']} · "
                            f"episode {row['episode_idx']} · frame {row['frame_idx']}"
                        ),
                        "episode": int(row["episode_idx"]),
                        "frame": int(row["frame_idx"]),
                        "global_idx": int(row["global_idx"]),
                    }
                )
        return "rank", samples
    return None


def build_compare_index(sources: list[CompareSource]) -> dict:
    """Build each normal viewer index and namespace its asset URLs by column."""
    runs = []
    groups: list[str] = []
    for source_index, source in enumerate(sources):
        index = build_index(source.val_dir)
        for group in index["groups"]:
            if group not in groups:
                groups.append(group)
        for probe in index["probes"]:
            for panels in probe["panels"].values():
                keys = _alignment_keys(panels)
                for panel in panels:
                    panel["align"] = keys[panel["file"]]
                    relative = panel["url"].removeprefix("/asset/")
                    step_name = relative.split("/", 1)[0]
                    interactive = _interactive_samples(
                        source.val_dir / step_name / probe["id"], probe["id"], panel["file"]
                    )
                    if interactive is not None:
                        panel["sample_alignment"], panel["samples"] = interactive
                    panel["url"] = f"/asset/{source_index}/{relative}"
        available_steps = {step["step"] for step in index["steps"]}
        initial_step = source.initial_step if source.initial_step in available_steps else None
        if initial_step is None and index["steps"]:
            initial_step = index["steps"][-1]["step"]
        index["run"].update(
            label=source.label,
            input_path=str(source.input_path),
            source_index=source_index,
            initial_step=initial_step,
            explicit_step=source.initial_step is not None,
        )
        runs.append(index)
    return {"runs": runs, "groups": groups, "max_runs": MAX_RUNS}


def make_compare_handler(sources: list[CompareSource]) -> type[BaseHTTPRequestHandler]:
    def asset_path(source_index: int, relative: str) -> Path:
        if not 0 <= source_index < len(sources):
            raise FileNotFoundError(relative)
        root = sources[source_index].val_dir
        path = (root / unquote(relative)).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise FileNotFoundError(relative)
        return path

    class CompareRequestHandler(BaseHTTPRequestHandler):
        server_version = "ProbeCompare/1.0"
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt: str, *args) -> None:
            return

        def do_HEAD(self) -> None:
            path = urlparse(self.path).path
            if path == "/":
                self._send_html(COMPARE_HTML, head_only=True)
            elif path in COMPARE_UI_ASSETS:
                self._handle_ui(path, head_only=True)
            elif path.startswith("/asset/"):
                self._handle_asset(path.removeprefix("/asset/"), head_only=True)
            else:
                self.send_error(HTTPStatus.NOT_FOUND)

        def do_GET(self) -> None:
            path = urlparse(self.path).path
            if path == "/":
                self._send_html(COMPARE_HTML)
            elif path in COMPARE_UI_ASSETS:
                self._handle_ui(path, head_only=False)
            elif path == "/api/index":
                _json_response(self, build_compare_index(sources))
            elif path.startswith("/asset/"):
                self._handle_asset(path.removeprefix("/asset/"), head_only=False)
            else:
                self.send_error(HTTPStatus.NOT_FOUND)

        def _handle_ui(self, path: str, head_only: bool) -> None:
            try:
                _send_file(self, COMPARE_UI_ASSETS[path], head_only=head_only)
            except (BrokenPipeError, ConnectionResetError):
                self.close_connection = True

        def _send_html(self, html: str, head_only: bool = False) -> None:
            body = html.encode("utf-8")
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            if not head_only:
                self.wfile.write(body)

        def _handle_asset(self, relative: str, head_only: bool) -> None:
            try:
                source_text, asset_relative = relative.split("/", 1)
                path = asset_path(int(source_text), asset_relative)
            except (FileNotFoundError, OSError, ValueError):
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            try:
                _send_file(self, path, head_only=head_only)
            except (BrokenPipeError, ConnectionResetError):
                self.close_connection = True

    return CompareRequestHandler


# Keep the browser assets beside this module so source checkouts and installed wheels
# serve the same interface, independently of the migration design gallery.
COMPARE_UI_DIR = Path(__file__).with_name("_compare_probes")
COMPARE_HTML = (COMPARE_UI_DIR / "index.html").read_text(encoding="utf-8")
COMPARE_UI_ASSETS = {
    f"/ui/compare-probes/{name}": COMPARE_UI_DIR / name
    for name in ("evidence.css", "styles.css", "model.js", "app.js")
}


def _make_server(
    sources: list[CompareSource], host: str, port: int, retries: int = 20
) -> ThreadingHTTPServer:
    handler = make_compare_handler(sources)
    last_error: OSError | None = None
    for candidate in range(port, port + retries + 1):
        try:
            return ThreadingHTTPServer((host, candidate), handler)
        except OSError as exc:
            last_error = exc
            if port == 0:
                break
    raise RuntimeError(f"Could not bind probe comparison server: {last_error}") from last_error


def compare_probes(
    run_dirs: list[str | Path],
    labels: list[str] | None = None,
    host: str = "127.0.0.1",
    port: int = 7871,
    open_browser: bool = True,
) -> None:
    """Launch the comparison viewer and block until interrupted."""
    sources = resolve_compare_sources(run_dirs, labels)
    server = _make_server(sources, host, port)
    url = f"http://{server.server_address[0]}:{server.server_address[1]}/"
    index = build_compare_index(sources)
    print(f"Probe comparison: {url}")
    for position, run in enumerate(index["runs"], start=1):
        print(f"  {position}. {run['run']['label']}: {run['run']['val_dir']}")
    print(f"{len(sources)} run(s), {len(index['groups'])} probe group(s)")
    if open_browser:
        threading.Timer(0.35, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping probe comparison.")
    finally:
        server.server_close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare probe output across one to four runs or checkpoints.")
    parser.add_argument(
        "run_dirs",
        nargs="+",
        help="Run, validation, or step_* directories to compare (one to four paths).",
    )
    parser.add_argument(
        "--label",
        action="append",
        default=[],
        help="Column label, repeated in the same order as the paths.",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7871)
    parser.add_argument("--no-open", action="store_true", help="Do not open a browser tab.")
    args = parser.parse_args()
    try:
        compare_probes(
            args.run_dirs,
            labels=args.label,
            host=args.host,
            port=args.port,
            open_browser=not args.no_open,
        )
    except (FileNotFoundError, ValueError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
