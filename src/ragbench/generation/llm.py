"""LLM generation backend supporting both Ollama and OpenAI-compatible APIs."""

import json
import os
from collections.abc import Iterator

import httpx
import ollama

API_URL = os.environ.get("API_URL", "http://localhost:11434")
API_AUTH_BEARER = os.environ.get("API_AUTH_BEARER")
API_SRC = os.environ.get("API_SRC", "ollama")


class _OpenAIBackend:
    """Minimal OpenAI-compatible chat backend using httpx."""

    def __init__(
        self, base_url: str, model: str, temperature: float, auth_bearer: str | None
    ) -> None:
        self.base_url = base_url
        self.model = model
        self.temperature = temperature
        self.headers = {"Content-Type": "application/json"}
        if auth_bearer:
            self.headers["Authorization"] = f"Bearer {auth_bearer}"

    def _url(self, path: str) -> str:
        return f"{self.base_url}/v1{path}"

    def health_check(self) -> bool:
        """Return True if the /v1/models endpoint is reachable."""
        try:
            with httpx.Client(timeout=10, follow_redirects=True) as client:
                r = client.get(self._url("/models"), headers=self.headers)
                r.raise_for_status()
            return True
        except Exception:
            return False

    def chat(self, messages: list[dict], temperature: float | None = None, **_):
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature if temperature is not None else self.temperature,
        }
        with httpx.Client(timeout=120, follow_redirects=True) as client:
            r = client.post(self._url("/chat/completions"), headers=self.headers, json=payload)
            r.raise_for_status()
            data = r.json()
        return data["choices"][0]["message"]["content"].strip()

    def stream(self, messages: list[dict], temperature: float | None = None, **_):
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature if temperature is not None else self.temperature,
            "stream": True,
        }
        with httpx.Client(timeout=120, follow_redirects=True) as client:
            with client.stream(
                "POST", self._url("/chat/completions"), headers=self.headers, json=payload
            ) as r:
                r.raise_for_status()
                for line in r.iter_lines():
                    if line.startswith("data: "):
                        chunk = line[6:]
                        if chunk == "[DONE]":
                            break
                        try:
                            obj = json.loads(chunk)
                            delta = obj["choices"][0]["delta"]
                            if "content" in delta and delta["content"]:
                                yield delta["content"]
                        except (json.JSONDecodeError, KeyError, IndexError):
                            continue


class _OllamaBackend:
    """Native Ollama backend via the official ollama Python client."""

    def __init__(self, host: str, model: str, temperature: float, auth_bearer: str | None) -> None:
        headers = {"Authorization": f"Bearer {auth_bearer}"} if auth_bearer else None
        self._client = ollama.Client(host=host, headers=headers)
        self.model = model
        self.temperature = temperature

    def health_check(self) -> bool:
        """Return True if Ollama responds on /api/tags."""
        try:
            self._client.list()
            return True
        except Exception:
            return False

    def chat(self, messages: list[dict], temperature: float | None = None, **_):
        response = self._client.chat(
            model=self.model,
            messages=messages,
            options={"temperature": temperature if temperature is not None else self.temperature},
        )
        return response["message"]["content"].strip()

    def stream(self, messages: list[dict], temperature: float | None = None, **_):
        for chunk in self._client.chat(
            model=self.model,
            messages=messages,
            options={"temperature": temperature if temperature is not None else self.temperature},
            stream=True,
        ):
            yield chunk["message"]["content"]


class Generator:
    """Unified generator that transparently talks to Ollama *or* OpenAI-compatible endpoints.

    The backend is chosen automatically:
      - If ``API_URL`` contains ``/v1/`` (or is not localhost:11434) the
        OpenAI-compatible backend is used.
      - Otherwise the native Ollama SDK backend is used.

    Auth header (``Authorization: Bearer <token>``) is injected when
    ``API_AUTH_BEARER`` is set, regardless of backend.
    """

    def __init__(
        self,
        model: str = "qwen2.5:3b-instruct",
        temperature: float = 0.0,
        host: str = API_URL,
        auth_bearer: str | None = API_AUTH_BEARER,
        api_src: str = API_SRC,
    ) -> None:
        self.model = model
        self.temperature = temperature

        match api_src:
            case "ollama":
                self._backend = _OllamaBackend(
                    host=host, model=model, temperature=temperature, auth_bearer=auth_bearer
                )
            case "openai":
                self._backend = _OpenAIBackend(
                    base_url=host, model=model, temperature=temperature, auth_bearer=auth_bearer
                )
            case _:
                raise ValueError(f"Unrecognized API source: {api_src}")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(self, prompt: str) -> str:
        """Blocking call; returns the full response string. Used by the evaluator."""
        return self._backend.chat(
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
        )

    def stream(self, prompt: str) -> Iterator[str]:
        """Streaming call; yields content chunks as they arrive. Used by the chat UI."""
        yield from self._backend.stream(
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
        )
