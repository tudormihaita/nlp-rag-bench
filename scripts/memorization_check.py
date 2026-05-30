#!/usr/bin/env python3
"""
Verify that the chosen LLM does not memorize MuSiQue answers.

Runs the no-RAG baseline on 50 randomly sampled questions from the full dev set
(drawn independently of the main 500-question sample) and reports average token F1.

If F1 > 0.20, the model has too much parametric knowledge of MuSiQue answers and
will produce an artificially high no-RAG baseline, compressing the visible RAG gains.
Consider switching to a smaller model variant.
"""
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import json

from loguru import logger
from rich.console import Console
from tqdm import tqdm

from ragbench.config import settings
from ragbench.data.loader import load_musique
from ragbench.evaluation.metrics import token_f1
from ragbench.generation.llm import Generator
from ragbench.generation.prompts import NO_RAG_PROMPT

_CHECK_SEED = 0 # intentionally different from the main sampling seed
_CHECK_N = 50
_F1_THRESHOLD = 0.20

console = Console()


def main() -> None:
    logger.info(f"Loading full dev set from {settings.musique_file}")
    all_questions = load_musique(settings.musique_file)
    logger.info(f"Loaded {len(all_questions)} questions")

    rng = random.Random(_CHECK_SEED)
    sample = rng.sample(all_questions, _CHECK_N)

    gen = Generator(settings.generator_model)
    logger.info(f"Running no-RAG baseline with {settings.generator_model} on {_CHECK_N} questions…")

    f1_scores: list[float] = []
    for q in tqdm(sample, desc="memorization check"):
        prompt = NO_RAG_PROMPT.format(question=q["question"])
        prediction = gen.generate(prompt)
        references = [q["answer"]] + q.get("answer_aliases", [])
        f1_scores.append(token_f1(prediction, references))

    avg_f1 = sum(f1_scores) / len(f1_scores)

    console.rule("Memorization Check Result")
    console.print(f"Model:    {settings.generator_model}")
    console.print(f"Avg F1:   [bold]{avg_f1:.3f}[/bold]  (threshold: {_F1_THRESHOLD})")

    if avg_f1 > _F1_THRESHOLD:
        console.print(
            f"[bold red]WARNING:[/bold red] F1 = {avg_f1:.3f} > {_F1_THRESHOLD}. "
            "The model may be memorizing MuSiQue answers. "
            "Consider switching to a smaller variant (e.g. llama3.2:3b, qwen2.5:3b-instruct)."
        )
        sys.exit(1)
    else:
        console.print(
            f"[bold green]OK:[/bold green] F1 = {avg_f1:.3f} ≤ {_F1_THRESHOLD}. "
            "Parametric memory is within acceptable range."
        )


if __name__ == "__main__":
    main()
