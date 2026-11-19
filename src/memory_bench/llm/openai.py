import json
import os
import time

from .base import LLM, Schema

_MAX_RETRIES = 6
_RETRY_BASE_DELAY = 5
_JSON_FIELD_TYPES = {
    "string": (str,),
    "boolean": (bool,),
    "integer": (int,),
    "number": (int, float),
    "array": (list,),
    "object": (dict,),
    "null": (type(None),),
}


def _parse_response(text: str | None, schema: Schema, *, allow_fence: bool) -> dict:
    content = (text or "").strip()
    if allow_fence:
        lines = content.splitlines()
        if len(lines) >= 3 and lines[0].strip().lower() in {"```json", "```"} and lines[-1].strip() == "```":
            content = "\n".join(lines[1:-1])
    data = json.loads(content)
    if not isinstance(data, dict) or any(field not in data for field in schema.required):
        raise json.JSONDecodeError("Model response is missing required JSON fields", content, 0)
    for field, spec in schema.properties.items():
        kind = spec.get("type")
        expected_types = _JSON_FIELD_TYPES.get(kind) if isinstance(kind, str) else None
        if field in data and expected_types and type(data[field]) not in expected_types:
            raise json.JSONDecodeError(f"Invalid JSON type for {field!r}", content, 0)
    return data


def _log_bad_json(model: str, response, text: str) -> None:
    if log := os.environ.get("AMB_BAD_JSON_LOG"):
        with open(log, "a", encoding="utf-8") as stream:
            entry = {
                "model": model,
                "finish_reason": getattr(response.choices[0], "finish_reason", None),
                "usage": getattr(getattr(response, "usage", None), "completion_tokens", None),
                "head": text[:400], "tail": text[-400:], "len": len(text),
            }
            stream.write(json.dumps(entry) + "\n")


class OpenAILLM(LLM):
    DEFAULT_MODEL = "gpt-4o"

    def __init__(self, model: str = DEFAULT_MODEL, *, api_key: str | None = None, base_url: str | None = None):
        from openai import OpenAI
        self._client = OpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY"), base_url=base_url)
        self._model = model
        self._use_prompt_for_json = False

    @property
    def model_id(self) -> str:
        return f"openai:{self._model}"

    def generate(self, prompt: str, schema: Schema) -> dict:
        schema_json = {
            "type": "object",
            "properties": schema.properties,
            "required": schema.required,
            "additionalProperties": False,
        }
        response_format = {
            "type": "json_schema",
            "json_schema": {"name": "response", "schema": schema_json, "strict": True},
        }
        json_prompt = (
            f"{prompt}\n\n"
            "Final output format: return only one valid JSON object matching this JSON Schema. "
            "Use the exact field names and value types. Do not include markdown or prose outside the object.\n"
            f"{json.dumps(schema_json, ensure_ascii=False)}"
        )
        use_schema = not self._use_prompt_for_json
        delay = _RETRY_BASE_DELAY
        for attempt in range(_MAX_RETRIES):
            can_retry = attempt < _MAX_RETRIES - 1
            try:
                response = self._client.chat.completions.create(
                    model=self._model,
                    messages=[{"role": "user", "content": prompt if use_schema else json_prompt}],
                    **({"response_format": response_format} if use_schema else {}),
                )
            except Exception as exc:
                status = getattr(exc, "status_code", None)
                if can_retry and use_schema and status in (400, 422):
                    message = str(exc).lower()
                    if any(term in message for term in ("response_format", "response format", "json_schema", "json schema")):
                        use_schema = False
                        continue
                transient = status == 429 or (isinstance(status, int) and 500 <= status < 600) or (
                    type(exc).__name__ in {"APIConnectionError", "APITimeoutError"}
                )
                if can_retry and transient:
                    time.sleep(delay)
                    delay *= 2
                    continue
                raise

            choice = response.choices[0]
            content = getattr(choice.message, "content", None)
            text = content if isinstance(content, str) else ""
            try:
                data = _parse_response(text, schema, allow_fence=not use_schema)
            except json.JSONDecodeError:
                _log_bad_json(self._model, response, text)
                if not can_retry:
                    raise
                if getattr(choice, "finish_reason", None) not in {"length", "content_filter"} and not getattr(
                    choice.message, "refusal", None,
                ):
                    use_schema = False
                continue
            if not use_schema:
                # Reuse confirmed compatibility mode for subsequent calls on this client.
                self._use_prompt_for_json = True
            return data
        raise RuntimeError("OpenAI request exhausted its retry budget.")
