from __future__ import annotations

import argparse
import socket
import threading
import time
import urllib.request

import uvicorn

from kannaadi.api.app import app


def announce_when_serving(port: int) -> None:
    health_url = f"http://127.0.0.1:{port}/health"
    for _ in range(100):
        try:
            with urllib.request.urlopen(health_url, timeout=0.5) as response:
                if response.status == 200:
                    print(f"KANNAADI_READY:{port}", flush=True)
                    return
        except OSError:
            time.sleep(0.05)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="Kannaadi local model service")
    result.add_argument("--host", default="127.0.0.1")
    result.add_argument("--port", type=int, default=0)
    result.add_argument("--token", required=True)
    return result


def main() -> None:
    args = parser().parse_args()
    if args.host not in {"127.0.0.1", "localhost"}:
        raise SystemExit("Kannaadi sidecar must bind to loopback")
    if len(args.token) < 32:
        raise SystemExit("Sidecar token is too short")
    app.state.api_token = args.token
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener.bind(("127.0.0.1", args.port))
    listener.listen(2048)
    port = listener.getsockname()[1]
    threading.Thread(target=announce_when_serving, args=(port,), daemon=True).start()
    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="info", access_log=False)
    uvicorn.Server(config).run(sockets=[listener])


if __name__ == "__main__":
    main()
