"""Continuously mirror one remote run dashboard and serve it read-only."""

from __future__ import annotations

import argparse
import http.server
import subprocess
import threading
import time
from pathlib import Path


def pull_loop(run_id: str, destination: Path, interval: float) -> None:
    remote = f"/jailbreaks/runs/{run_id}"
    destination.mkdir(parents=True, exist_ok=True)
    while True:
        for name in ("dashboard.html", "dashboard_history.jsonl"):
            subprocess.run(
                ["modal", "volume", "get", "bt-outputs", f"{remote}/{name}", str(destination / name)],
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        time.sleep(interval)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_id")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--interval", type=float, default=5.0)
    args = parser.parse_args()
    if "/" in args.run_id or args.run_id in {".", ".."}:
        raise SystemExit("invalid run id")
    destination = Path(__file__).resolve().parent / "live" / args.run_id
    threading.Thread(target=pull_loop, args=(args.run_id, destination, args.interval), daemon=True).start()
    handler = lambda *a, **kw: http.server.SimpleHTTPRequestHandler(*a, directory=str(destination), **kw)
    print(f"http://127.0.0.1:{args.port}/dashboard.html", flush=True)
    http.server.ThreadingHTTPServer(("127.0.0.1", args.port), handler).serve_forever()


if __name__ == "__main__":
    main()
