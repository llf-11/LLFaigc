import json
import time
from urllib import error, request


class RunningHubError(RuntimeError):
    pass


class RunningHubClient:
    def __init__(self, api_key, base_url="https://www.runninghub.cn", timeout=120):
        self.api_key = api_key.strip()
        self.base_url = base_url.rstrip("/")
        self.timeout = int(timeout)

    def post(self, path, payload):
        if not self.api_key:
            raise RunningHubError("RunningHub API key is required.")

        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        req = request.Request(
            self.base_url + path,
            data=body,
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json",
                "Authorization": "Bearer " + self.api_key,
                "x-api-key": self.api_key,
            },
            method="POST",
        )

        try:
            with request.urlopen(req, timeout=self.timeout) as resp:
                data = resp.read().decode("utf-8")
        except error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RunningHubError(f"HTTP {exc.code}: {detail}") from exc
        except error.URLError as exc:
            raise RunningHubError(str(exc)) from exc

        try:
            return json.loads(data)
        except json.JSONDecodeError:
            return {"raw": data}

    def create_task(self, endpoint_path, payload):
        return self.post(endpoint_path, payload)

    def submit_openapi(self, endpoint, payload):
        payload = dict(payload)
        payload.setdefault("appCode", "comfyui_llfaigc")
        result = self.post("/" + endpoint.strip("/"), payload)
        task_id = (
            result.get("taskId")
            or result.get("task_id")
            or result.get("data", {}).get("taskId")
            or result.get("data", {}).get("task_id")
        )
        if not task_id:
            raise RunningHubError("RunningHub response did not include taskId: " + json.dumps(result, ensure_ascii=False))
        return str(task_id), result

    def query_openapi(self, task_id):
        return self.post("/query", {"taskId": task_id})

    def poll_openapi(self, task_id, interval=5, max_wait=600):
        started = time.time()
        last_result = {}
        while True:
            last_result = self.query_openapi(task_id)
            status = str(last_result.get("status", "")).upper()
            if status in {"SUCCESS", "FAILED", "CANCEL"}:
                return last_result
            if time.time() - started >= max_wait:
                return last_result
            time.sleep(max(1, int(interval)))

    def poll_task(self, status_path, task_id, interval=3, max_wait=600):
        started = time.time()
        while True:
            result = self.post(status_path, {"taskId": task_id})
            status = str(
                result.get("status")
                or result.get("data", {}).get("status", "")
                or result.get("state", "")
            ).lower()
            if status in {"success", "succeeded", "finished", "completed", "failed", "error"}:
                return result
            if time.time() - started >= max_wait:
                return result
            time.sleep(max(1, int(interval)))
