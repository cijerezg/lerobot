#!/usr/bin/env python

"""
Static server with HTTP Range support, for the annotation review page.

`python -m http.server` ignores Range entirely: it answers a range request with 200 and the
whole body. That makes video seeking impossible, which matters because episodes are packed
many-to-a-file -- rebot_sorting_clothes_v1 ep7's wrist view starts 516s into a 197MB mp4, so
without Range the browser must fetch the entire file before it can show the right frame, and
the video looks frozen. With Range it seeks instantly.

    uv run lerobot/src/lerobot/data_processing/annotate/serve_review.py \
        --data-dir outputs/rebot_sorting_clothes_v1
"""

import argparse
import functools
import http.server
import os
import re
import socketserver


class RangeHandler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Accept-Ranges", "bytes")
        super().end_headers()

    def do_GET(self):
        rng = self.headers.get("Range")
        path = self.translate_path(self.path)
        if not rng or not os.path.isfile(path):
            return super().do_GET()

        m = re.match(r"bytes=(\d*)-(\d*)$", rng.strip())
        if not m:
            return super().do_GET()

        size = os.path.getsize(path)
        start = int(m.group(1)) if m.group(1) else 0
        end = int(m.group(2)) if m.group(2) else size - 1
        end = min(end, size - 1)
        if start > end or start >= size:
            self.send_response(416)
            self.send_header("Content-Range", f"bytes */{size}")
            self.end_headers()
            return

        self.send_response(206)
        self.send_header("Content-Type", self.guess_type(path))
        self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.send_header("Content-Length", str(end - start + 1))
        self.end_headers()

        remaining = end - start + 1
        with open(path, "rb") as f:
            f.seek(start)
            while remaining > 0:
                chunk = f.read(min(64 * 1024, remaining))
                if not chunk:
                    break
                try:
                    self.wfile.write(chunk)
                except (BrokenPipeError, ConnectionResetError):
                    return          # browser cancelled the range mid-seek; normal
                remaining -= len(chunk)

    def log_message(self, fmt, *args):
        pass                        # a scrubbing video emits a range request per seek


def main():
    ap = argparse.ArgumentParser(description="Serve a dataset directory with Range support")
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()

    handler = functools.partial(RangeHandler, directory=args.data_dir)
    with socketserver.ThreadingTCPServer(("127.0.0.1", args.port), handler) as httpd:
        httpd.daemon_threads = True
        print(f"serving {args.data_dir} on http://localhost:{args.port}")
        print(f"  open http://localhost:{args.port}/review_annotations.html")
        print("  ctrl-c to stop")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nstopped")


if __name__ == "__main__":
    main()
