# Author: Tudor Mihaita
RAG_PROMPT = """\
Answer the question using the context passages below.
The question may require chaining facts across multiple passages — follow each step to reach the final answer, not an intermediate one.
Think through the reasoning chain silently, then write only the final answer: a name, place, date, or brief phrase.
Only write "I don't know" if the answer cannot be inferred from any passage even after careful reasoning.

Context:
{context}

Question: {question}

Answer:"""

NO_RAG_PROMPT = """\
Answer the question with a short, direct answer — a name, place, date, or brief phrase.
If you don't know, say "I don't know."

Question: {question}

Answer:"""

INTERMEDIATE_PROMPT = """\
Answer the question with a single name, place, date, or short phrase based on the passages below.
If the answer is not present, write "unknown".

Context:
{context}

Question: {question}

Answer:"""

REWRITE_PROMPT = """\
Rewrite the follow-up question as a self-contained, specific question using the known facts.
Replace vague references like "the actor", "that place", or "the person above" with the actual name or entity from the known facts.
Output only the rewritten question, nothing else.

Example:
Follow-up question: Who is the spouse of the actor?
Known facts: Paul Bettany
Rewritten question: Who is Paul Bettany's spouse?

Follow-up question: {question}
Known facts: {facts}

Rewritten question:"""

DECOMPOSITION_PROMPT = """\
Decompose the multi-hop question into 2-4 simpler sub-questions, each answerable by looking up a single fact.
Each sub-question must be self-contained — do NOT use references like "the person above", "that place", or "the answer to #1".

Output ONLY the sub-questions, one per line, numbered.

Example:
Question: In which city was the director of 'The Dark Knight' born?
1. Who directed 'The Dark Knight'?
2. In which city was Christopher Nolan born?

Question: {question}

Sub-questions:"""
