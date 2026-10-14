# Agent: builds the prompt, calls the Dartmouth API, parses the reply.
# It never changes the task state.

import os
import time
from dotenv import load_dotenv
from openai import OpenAI

from prompt_builder import build_prompt, PROMPT_VERSION
from response_parser import parse_decision

load_dotenv()

MAX_ATTEMPTS = 3
REASONING_CHARS = 3000   # truncate saved reasoning traces

client = OpenAI(
    api_key=os.environ["DARTMOUTH_API_KEY"],
    base_url="https://chat.dartmouth.edu/api",
)


def get_reasoning(message):
    """Reasoning models return their thinking in a non-standard extra field."""
    extra = message.model_extra or {}
    text = extra.get("reasoning_content")
    return str(text)[:REASONING_CHARS] if text else None


def get_decision(state, model, bet_options, temperature=1.0, max_tokens=4000):
    """Ask the model for one decision. Retry up to MAX_ATTEMPTS with the SAME prompt."""
    prompt = build_prompt(state)
    last_error = None
    last_finish = None
    reasoning = None
    text = ""

    for attempt in range(1, MAX_ATTEMPTS + 1):
        start = time.perf_counter()
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        latency = time.perf_counter() - start
        message = response.choices[0].message
        text = message.content or ""
        last_finish = response.choices[0].finish_reason
        reasoning = get_reasoning(message)

        try:
            bet, color = parse_decision(text, bet_options)
            return {
                "bet": bet,
                "color": color,
                "n_attempts": attempt,
                "parse_failed": False,
                "raw_reply": text,
                "reasoning": reasoning,
                "prompt_version": PROMPT_VERSION,
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "latency_s": round(latency, 3),
                "finish_reason": last_finish,
            }
        except ValueError as e:
            last_error = e

    return {
        "bet": None,
        "color": None,
        "n_attempts": MAX_ATTEMPTS,
        "parse_failed": True,
        "raw_reply": text,
        "reasoning": reasoning,
        "prompt_version": PROMPT_VERSION,
        "error": str(last_error),
        "finish_reason": last_finish,
    }