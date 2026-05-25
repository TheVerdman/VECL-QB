from __future__ import annotations

import argparse
import json
import mimetypes
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, cast
from urllib.parse import unquote, urlparse

from vecl.ui.cockpit import CockpitBackend, parse_json_object

STATIC_ROOT = Path(__file__).with_name("static")


class CockpitHTTPServer(ThreadingHTTPServer):
    def __init__(self, server_address: tuple[str, int], backend: CockpitBackend) -> None:
        super().__init__(server_address, CockpitRequestHandler)
        self.backend = backend


class CockpitRequestHandler(BaseHTTPRequestHandler):
    server_version = "VECLQBCockpit/0.1"

    def do_GET(self) -> None:  # noqa: N802
        try:
            self._handle_get()
        except KeyError as exc:
            self._send_json({"error": str(exc).strip("'")}, status=404)
        except ValueError as exc:
            self._send_json({"error": str(exc)}, status=400)
        except Exception as exc:
            self._send_json({"error": str(exc)}, status=500)

    def do_POST(self) -> None:  # noqa: N802
        try:
            self._handle_post()
        except KeyError as exc:
            self._send_json({"error": str(exc).strip("'")}, status=404)
        except ValueError as exc:
            self._send_json({"error": str(exc)}, status=400)
        except Exception as exc:
            self._send_json({"error": str(exc)}, status=500)

    def log_message(self, format: str, *args: object) -> None:
        return

    @property
    def backend(self) -> CockpitBackend:
        return cast(CockpitHTTPServer, self.server).backend

    def _handle_get(self) -> None:
        path = urlparse(self.path).path
        if path in {"/", "/index.html"}:
            self._send_static(STATIC_ROOT / "index.html")
            return
        if path.startswith("/static/"):
            self._send_static(STATIC_ROOT / unquote(path.removeprefix("/static/")))
            return
        if path == "/api/health":
            self._send_json(self.backend.health())
            return
        if path == "/api/drivers":
            self._send_json({"drivers": self.backend.list_drivers()})
            return
        if path == "/api/specialists":
            self._send_json({"specialists": self.backend.list_specialists()})
            return
        if path == "/api/sessions":
            self._send_json({"sessions": self.backend.list_sessions()})
            return
        if path.startswith("/api/runs/") and path.endswith("/events"):
            run_id = unquote(path.removeprefix("/api/runs/").removesuffix("/events"))
            self._send_json({"events": self.backend.get_run_events(run_id)})
            return
        if path.startswith("/api/runs/"):
            run_id = unquote(path.removeprefix("/api/runs/"))
            self._send_json({"run": self.backend.get_run(run_id)})
            return
        if path.startswith("/api/artifacts/"):
            artifact_id = unquote(path.removeprefix("/api/artifacts/"))
            self._send_json(self.backend.get_artifact_content(artifact_id))
            return
        self._send_json({"error": "not found"}, status=404)

    def _handle_post(self) -> None:
        path = urlparse(self.path).path
        body = self._read_json()
        if path == "/api/sessions":
            title = str(body.get("title") or "New conversation")
            self._send_json({"session": self.backend.create_session(title)})
            return
        if path == "/api/chat":
            payload_override = _payload_override_from_body(body)
            run = self.backend.submit_chat(
                prompt=str(body.get("prompt") or ""),
                driver_provider=str(body.get("driver") or body.get("provider") or "openai"),
                session_id=(
                    str(body["session_id"]) if body.get("session_id") is not None else None
                ),
                payload_override=payload_override,
            )
            self._send_json({"run": run}, status=202)
            return
        self._send_json({"error": "not found"}, status=404)

    def _read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length") or "0")
        if length <= 0:
            return {}
        raw = self.rfile.read(length).decode()
        loaded = json.loads(raw)
        if not isinstance(loaded, dict):
            raise ValueError("request JSON must be an object")
        return loaded

    def _send_json(self, payload: dict[str, Any], *, status: int = 200) -> None:
        raw = json.dumps(payload, sort_keys=True, default=str).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(raw)

    def _send_static(self, path: Path) -> None:
        resolved = path.resolve()
        root = STATIC_ROOT.resolve()
        if not resolved.is_file() or root not in resolved.parents and resolved != root:
            self._send_json({"error": "not found"}, status=404)
            return
        raw = resolved.read_bytes()
        content_type = mimetypes.guess_type(str(resolved))[0] or "application/octet-stream"
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)


def _payload_override_from_body(body: dict[str, Any]) -> dict[str, Any] | None:
    payload = body.get("payload")
    if isinstance(payload, dict):
        return dict(payload)
    payload_json = body.get("payload_json")
    if isinstance(payload_json, str) and payload_json.strip():
        return parse_json_object(payload_json)
    return None


def run(host: str = "127.0.0.1", port: int = 8765) -> None:
    backend = CockpitBackend()
    server = CockpitHTTPServer((host, port), backend)
    actual_host = str(server.server_address[0])
    actual_port = int(server.server_address[1])
    print(f"VECL-QB cockpit listening at http://{actual_host}:{actual_port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nVECL-QB cockpit stopped.", flush=True)
    finally:
        server.server_close()


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the local VECL-QB research cockpit.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    run(host=str(args.host), port=int(args.port))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
