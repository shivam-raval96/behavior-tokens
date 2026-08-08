from __future__ import annotations

import argparse
import subprocess
import threading
import time
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


VOLUME = "bt-outputs"
REMOTE_ROOT = "/steering_suffix_optimization/runs"


def sync_dashboard(run_id: str, local_root: Path) -> None:
    destination = local_root / run_id
    destination.mkdir(parents=True, exist_ok=True)
    for name in ("dashboard.html", "dashboard_history.jsonl"):
        subprocess.run(
            [
                "modal",
                "volume",
                "get",
                "--force",
                VOLUME,
                f"{REMOTE_ROOT}/{run_id}/{name}",
                str(destination / name),
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )


def mirror_forever(run_id: str, local_root: Path, interval: float) -> None:
    while True:
        try:
            sync_dashboard(run_id, local_root)
        except subprocess.CalledProcessError:
            pass
        time.sleep(interval)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Mirror a Volume-backed experiment dashboard to localhost"
    )
    parser.add_argument("run_id")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--interval", type=float, default=5.0)
    parser.add_argument(
        "--local-root", type=Path, default=Path(__file__).parent / "runs"
    )
    args = parser.parse_args()
    sync_dashboard(args.run_id, args.local_root)
    threading.Thread(
        target=mirror_forever,
        args=(args.run_id, args.local_root, args.interval),
        daemon=True,
    ).start()

    def handler(*handler_args, **kwargs):
        return SimpleHTTPRequestHandler(
            *handler_args, directory=str(args.local_root), **kwargs
        )

    print(f"http://127.0.0.1:{args.port}/{args.run_id}/dashboard.html", flush=True)
    ThreadingHTTPServer(("127.0.0.1", args.port), handler).serve_forever()


if __name__ == "__main__":
    main()
