"""Concise diagnostics for evaluation errors."""

from contextlib import contextmanager
from json import JSONDecodeError

_STAGE_NOTE = "AMB stage: "
_SUBJECT_NOTE = "AMB subject: "

_HTTP_HINTS = {
    401: "鉴权失败，请检查 API Key",
    403: "访问被拒绝，请检查 Key 和服务权限",
    404: "资源不存在，请检查 API 地址、模型 ID 或 Bank ID",
    429: "请求被限流，请检查并发或服务配额",
}


@contextmanager
def evaluation_step(stage: str, subject: str = ""):
    try:
        yield
    except Exception as exc:
        if not _error_note(exc, _STAGE_NOTE):
            exc.add_note(_STAGE_NOTE + stage)
        if subject:
            exc.add_note(_SUBJECT_NOTE + subject)
        raise


def _error_note(exc: Exception, prefix: str) -> str:
    return next((note[len(prefix):] for note in reversed(getattr(exc, "__notes__", ())) if note.startswith(prefix)), "")


def error_location(exc: Exception, default: str) -> str:
    stage = _error_note(exc, _STAGE_NOTE) or default
    subject = _error_note(exc, _SUBJECT_NOTE)
    return f"{stage} · {subject}" if subject else stage


def error_reason(exc: Exception) -> str:
    cause: BaseException | None = exc
    seen: set[int] = set()
    while cause is not None and id(cause) not in seen:
        seen.add(id(cause))
        response = getattr(cause, "response", None)
        status = getattr(cause, "status_code", None) or getattr(response, "status_code", None) or getattr(cause, "code", None)
        if isinstance(status, int) and 400 <= status <= 599:
            hint = _HTTP_HINTS.get(status, "上游服务出错" if status >= 500 else "服务拒绝了请求")
            return f"HTTP {status}，{hint}"
        if isinstance(cause, JSONDecodeError):
            return f"JSON 格式错误（第 {cause.lineno} 行，第 {cause.colno} 列）：{cause.msg}"
        kind = type(cause).__name__
        if isinstance(cause, TimeoutError) or "Timeout" in kind:
            return "连接或请求超时，请检查服务地址和网络"
        if isinstance(cause, ConnectionError) or kind in {"ConnectError", "APIConnectionError", "URLError"}:
            return "无法连接服务，请检查地址和网络"
        cause = cause.__cause__
    lines = str(exc).splitlines()
    detail = lines[0] if lines else ""
    return f"{type(exc).__name__}: {detail}" if detail else type(exc).__name__
