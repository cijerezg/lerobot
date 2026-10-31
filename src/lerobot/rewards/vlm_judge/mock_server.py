"""Stand-in for both Spark services, for dry runs without the models:

    uv run python -m lerobot.rewards.vlm_judge.mock_server --port 8011   # /v1/chat/completions -> fixed judge (+ assigner "step") JSON
                                                                         # /score -> fixed Robometer progress/success

The judge answer walks the stages with the request count so a timeline has some shape; the Robometer answer ramps
progress over the frames. Kill it when the test ends."""

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


class Handler(BaseHTTPRequestHandler):
    count = 0

    def do_POST(self) -> None:
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        Handler.count += 1
        if self.path.endswith("/chat/completions"):
            n_images = sum(1 for p in body["messages"][1]["content"] if p["type"] == "image_url")
            k = Handler.count // 4
            # ids stay inside every spec (>= 4 subtasks, >= 4 stages) and every assigner window (>= 2 offered steps)
            answer = dict(subtask_id=min(k, 3), stage_id=min(k, 3), progress_0_1=min(0.2 * k + 0.05, 1.0), confidence=0.8,
                          evidence=f"mock answer {Handler.count} on {n_images} images", step=k % 2)
            out = dict(id="mock", object="chat.completion", model=body.get("model"),
                       choices=[dict(index=0, message=dict(role="assistant", content=json.dumps(answer)), finish_reason="stop")],
                       usage=dict(prompt_tokens=0, completion_tokens=0, total_tokens=0))
        elif self.path.endswith("/score"):
            n = len(body["frames"])
            out = dict(progress=[round((i + 1) / n, 3) for i in range(n)], success=[round(0.1 + 0.8 * (i + 1) / n, 3) for i in range(n)],
                       task=body["task"], n_frames=n)
        else:
            self.send_error(404)
            return
        data = json.dumps(out).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *args) -> None:
        pass


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8011)
    args = ap.parse_args()
    ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
