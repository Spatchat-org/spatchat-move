# llm_utils.py
import json
import os
import random
import time
from pathlib import Path
from typing import Iterator

from huggingface_hub import InferenceClient
from together import Together

try:
    from together.error import InvalidRequestError, RateLimitError, ServiceUnavailableError
except Exception:  # pragma: no cover
    InvalidRequestError = type("InvalidRequestError", (Exception,), {})
    ServiceUnavailableError = type("ServiceUnavailableError", (Exception,), {})
    try:
        from together import RateLimitError  # type: ignore
    except Exception:  # pragma: no cover
        RateLimitError = type("RateLimitError", (Exception,), {})


HF_MODEL_DEFAULT = "meta-llama/Llama-3.1-8B-Instruct"
TOGETHER_MODEL_DEFAULT = "meta-llama/Llama-3.3-70B-Instruct-Turbo"


def _load_local_env() -> None:
    root = Path(__file__).resolve().parent
    for name in (".env", ".env.local"):
        path = root / name
        if not path.exists():
            continue
        try:
            for raw_line in path.read_text(encoding="utf-8").splitlines():
                line = raw_line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip()
                if not key or key in os.environ:
                    continue
                if (value.startswith('"') and value.endswith('"')) or (value.startswith("'") and value.endswith("'")):
                    value = value[1:-1]
                os.environ[key] = value
        except Exception:
            continue


_load_local_env()


def _choice_content(choice):
    msg = getattr(choice, "message", None) or (choice.get("message") if isinstance(choice, dict) else None)
    content = None if msg is None else (msg.get("content") if isinstance(msg, dict) else getattr(msg, "content", None))
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                parts.append(part.get("text", ""))
            elif isinstance(part, str):
                parts.append(part)
        content = "".join(parts)
    return content or ""


def _delta_text(delta):
    return delta.get("content", "") if isinstance(delta, dict) else getattr(delta, "content", "")


def _stream_choice_text(chunk) -> str:
    choices = getattr(chunk, "choices", None) or (chunk.get("choices") if isinstance(chunk, dict) else None) or []
    if not choices:
        return ""
    delta = getattr(choices[0], "delta", None) or (choices[0].get("delta") if isinstance(choices[0], dict) else None)
    if delta is None:
        return ""
    return _delta_text(delta)


class _SpacedCallLimiter:
    def __init__(self, min_interval_seconds: float):
        self.min = float(min_interval_seconds)
        self._last = 0.0

    def wait(self):
        now = time.monotonic()
        gap = now - self._last
        if gap < self.min:
            time.sleep(self.min - gap)
        self._last = time.monotonic()


class UnifiedLLM:
    def __init__(self):
        token = (os.getenv("HF_TOKEN") or "").strip()
        endpoint_url = (os.getenv("HF_ENDPOINT_URL") or os.getenv("HF_BASE_URL") or "").strip()
        self.hf_model = (os.getenv("HF_MODEL") or HF_MODEL_DEFAULT).strip()
        self.hf_token = token
        self.hf_default = InferenceClient(token=token, timeout=300)

        if endpoint_url:
            self.hf = InferenceClient(base_url=endpoint_url, token=token, timeout=300)
            self.hf_chat_enabled = True
        else:
            self.hf = InferenceClient(model=self.hf_model, token=token, timeout=300)
            self.hf_chat_enabled = False

        self.together = None
        self.together_model = (os.getenv("TOGETHER_MODEL") or TOGETHER_MODEL_DEFAULT).strip()
        self.together_model_candidates = []
        for model_name in (
            self.together_model,
            "meta-llama/Llama-3.3-70B-Instruct-Turbo",
            "Qwen/Qwen2.5-7B-Instruct-Turbo",
        ):
            if model_name and model_name not in self.together_model_candidates:
                self.together_model_candidates.append(model_name)

        tk = (os.getenv("TOGETHER_API_KEY") or "").strip()
        if tk:
            self.together = Together(api_key=tk)
            min_interval = float((os.getenv("TOGETHER_MIN_INTERVAL_SECONDS") or "0.0").strip() or "0.0")
            self._lim = _SpacedCallLimiter(min_interval)

    @staticmethod
    def _is_not_found_error(exc) -> bool:
        response = getattr(exc, "response", None)
        status_code = getattr(response, "status_code", None)
        return status_code == 404 or "404" in str(exc)

    @staticmethod
    def _messages_to_prompt(messages):
        parts = []
        for m in messages:
            role = m.get("role", "user")
            content = m.get("content", "")
            parts.append(f"<|{role}|>\n{content}\n")
        parts.append("<|assistant|>\n")
        return "".join(parts)

    def _hf_chat(self, messages, max_tokens=256, temperature=0.0):
        prompt = self._messages_to_prompt(messages)

        if not self.hf_chat_enabled and not self.hf_token:
            return self.hf_default.text_generation(
                prompt,
                max_new_tokens=max_tokens,
                temperature=temperature,
                stream=False,
                return_full_text=False,
            )

        tries, delay, last = 3, 2.5, None
        for _ in range(tries):
            try:
                if self.hf_chat_enabled and hasattr(self.hf, "chat_completion"):
                    resp = self.hf.chat_completion(
                        messages=messages,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        stream=False,
                    )
                    return _choice_content(resp.choices[0])
                return self.hf.text_generation(
                    prompt,
                    model=None if self.hf_chat_enabled else self.hf_model,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    stream=False,
                    return_full_text=False,
                )
            except Exception as exc:
                if self._is_not_found_error(exc):
                    return self.hf_default.text_generation(
                        prompt,
                        max_new_tokens=max_tokens,
                        temperature=temperature,
                        stream=False,
                        return_full_text=False,
                    )
                last = exc
                time.sleep(delay)
                delay *= 1.8
        raise last

    def _hf_chat_stream(self, messages, max_tokens=256, temperature=0.0) -> Iterator[str]:
        prompt = self._messages_to_prompt(messages)

        if not self.hf_chat_enabled and not self.hf_token:
            for token in self.hf_default.text_generation(
                prompt,
                max_new_tokens=max_tokens,
                temperature=temperature,
                stream=True,
                return_full_text=False,
            ):
                if token:
                    yield str(token)
            return

        if self.hf_chat_enabled and hasattr(self.hf, "chat_completion"):
            resp = self.hf.chat_completion(
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                stream=True,
            )
            for chunk in resp:
                text = _stream_choice_text(chunk)
                if text:
                    yield text
            return

        try:
            for token in self.hf.text_generation(
                prompt,
                model=self.hf_model,
                max_new_tokens=max_tokens,
                temperature=temperature,
                stream=True,
                return_full_text=False,
            ):
                if token:
                    yield str(token)
            return
        except Exception as exc:
            if self._is_not_found_error(exc):
                for token in self.hf_default.text_generation(
                    prompt,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    stream=True,
                    return_full_text=False,
                ):
                    if token:
                        yield str(token)
                return
            raise

    def _together_chat(self, messages, temperature=0.0, max_tokens=256):
        if not self.together:
            raise RuntimeError("Together client is not configured")

        last = None
        for model_name in self.together_model_candidates:
            self._lim.wait()
            backoff = 12.0
            for i in range(4):
                try:
                    resp = self.together.chat.completions.create(
                        model=model_name,
                        messages=messages,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        stream=False,
                    )
                    self.together_model = model_name
                    return _choice_content(resp.choices[0])
                except InvalidRequestError as exc:
                    last = exc
                    break
                except (RateLimitError, ServiceUnavailableError) as exc:
                    last = exc
                    if i == 3:
                        break
                    time.sleep(backoff + random.uniform(0, 3))
                    backoff *= 1.8
                except Exception as exc:
                    last = exc
                    break
        raise last if last is not None else RuntimeError("Together chat failed without an error")

    def _together_chat_stream(self, messages, temperature=0.0, max_tokens=256) -> Iterator[str]:
        if not self.together:
            raise RuntimeError("Together client is not configured")

        last = None
        for model_name in self.together_model_candidates:
            self._lim.wait()
            backoff = 12.0
            for i in range(4):
                try:
                    resp = self.together.chat.completions.create(
                        model=model_name,
                        messages=messages,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        stream=True,
                    )
                    self.together_model = model_name
                    for chunk in resp:
                        text = _stream_choice_text(chunk)
                        if text:
                            yield text
                    return
                except InvalidRequestError as exc:
                    last = exc
                    break
                except (RateLimitError, ServiceUnavailableError) as exc:
                    last = exc
                    if i == 3:
                        break
                    time.sleep(backoff + random.uniform(0, 3))
                    backoff *= 1.8
                except Exception as exc:
                    last = exc
                    break
        raise last if last is not None else RuntimeError("Together chat stream failed without an error")

    def chat(self, messages, temperature=0.0, max_tokens=256, stream=False):
        del stream
        if self.together:
            try:
                return self._together_chat(messages, temperature=temperature, max_tokens=max_tokens)
            except Exception:
                pass
        try:
            return self._hf_chat(messages, max_tokens=max_tokens, temperature=temperature)
        except Exception:
            if not self.together:
                raise
            return self._together_chat(messages, temperature=temperature, max_tokens=max_tokens)

    def stream_chat(self, messages, temperature=0.0, max_tokens=256) -> Iterator[str]:
        if self.together:
            try:
                yield from self._together_chat_stream(messages, temperature=temperature, max_tokens=max_tokens)
                return
            except Exception:
                pass
        try:
            yield from self._hf_chat_stream(messages, max_tokens=max_tokens, temperature=temperature)
            return
        except Exception:
            if not self.together:
                raise
        yield from self._together_chat_stream(messages, temperature=temperature, max_tokens=max_tokens)


SYSTEM_PROMPT = """
You are SpatChat, a wildlife movement and home-range analysis expert.

Your role:
- Answer general questions naturally and conversationally when they do not depend on the uploaded data.
- Answer wildlife movement, home-range, estimator, and movement-analysis questions as a domain expert.
- Help the user use this app for MCP, KDE, AKDE, LoCoH, dBBMM, displacement, step lengths, turning angles,
  autocorrelation diagnostics, and hidden Markov model state identification.

You are given a JSON object called dataset_context with facts about the currently loaded dataset
(columns, counts, ranges, and small samples). When the user asks anything about the uploaded data,
answer STRICTLY using dataset_context. If the answer is not present or cannot be derived from it,
say you do not know and briefly suggest what the user could do to compute it.

Rules:
- For general knowledge or casual chat, answer normally in <= 3 sentences.
- After a general knowledge or casual chat answer, end with one short sentence redirecting the user back to this room's purpose.
- For dataset-specific questions, do not fabricate numbers, files, columns, or results.
- When relevant, connect your answer back to a concrete action the user can take in this room.
- Keep replies concise and practical.
""".strip()

FALLBACK_PROMPT = """
You are SpatChat, a wildlife movement expert. If you can't map to a tool, answer naturally in <=3 sentences.
""".strip()

_llm = UnifiedLLM()


def ask_llm(chat_history, user_input, context=None):
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    if context is not None:
        messages.append({
            "role": "system",
            "content": "dataset_context:\n" + json.dumps(context, ensure_ascii=False)
        })
    for m in chat_history:
        messages.append({"role": m["role"], "content": m["content"]})
    messages.append({"role": "user", "content": user_input})

    resp = _llm.chat(messages, temperature=0.0, max_tokens=256, stream=False)
    try:
        call = json.loads(resp)
        return call, resp
    except Exception:
        return None, resp


def ask_llm_stream(chat_history, user_input, context=None):
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    if context is not None:
        messages.append({
            "role": "system",
            "content": "dataset_context:\n" + json.dumps(context, ensure_ascii=False)
        })
    for m in chat_history:
        messages.append({"role": m["role"], "content": m["content"]})
    messages.append({"role": "user", "content": user_input})

    yield from _llm.stream_chat(messages, temperature=0.0, max_tokens=256)
