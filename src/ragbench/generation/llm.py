from typing import Iterator

import ollama


class Generator:
    def __init__(self, model: str = "qwen2.5:3b-instruct", temperature: float = 0.0) -> None:
        self.model = model
        self.temperature = temperature

    def generate(self, prompt: str) -> str:
        """Blocking call; returns the full response string. Used by the evaluator."""
        response = ollama.chat(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": self.temperature},
        )
        return response["message"]["content"].strip()

    def stream(self, prompt: str) -> Iterator[str]:
        """Streaming call; yields content chunks as they arrive. Used by the chat UI."""
        for chunk in ollama.chat(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": self.temperature},
            stream=True,
        ):
            yield chunk["message"]["content"]