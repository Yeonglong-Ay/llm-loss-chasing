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

client = OpenAI(
    api_key=os.environ["DARTMOUTH_API_KEY"],                   
    base_url="https://chat.dartmouth.edu/api",
)


def get_decision(state, model, bet_options, temperature=1.0, max_tokens=300):
    """Ask the model for one decision. Retry up to MAX_ATTEMPTS with the SAME prompt."""
    prompt = build_prompt(state)
    last_error = None
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
        text = response.choices[0].message.content or ""      

        try:
            bet, color = parse_decision(text, bet_options)          
            return {
                "bet": bet,
                "color": color,
                "n_attempts": attempt,
                "parse_failed": False,
                "raw_reply": text,
                "prompt_version": PROMPT_VERSION,
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "latency_s": round(latency, 3),
            }
        except ValueError as e:                                 
            last_error = e

    return {
        "bet": None,
        "color": None,
        "n_attempts": MAX_ATTEMPTS,
        "parse_failed": True,
        "raw_reply": text,
        "prompt_version": PROMPT_VERSION,
        "error": str(last_error),
    }


if __name__ == "__main__":
    # Temporary live test: 6 trials on a free model. Costs no credits.
    from task_engine import Task, DEFAULT_BETS

    MODEL = "google.gemma-3-27b-it"
    task = Task()

    for _ in range(6):
        state = task.get_state()
        result = get_decision(state, MODEL, DEFAULT_BETS)

        if result["parse_failed"]:
            print("PARSE FAILED:", result["error"])
            print("Raw reply:", result["raw_reply"])
            break

        rec = task.play_trial(result["bet"], result["color"])
        print(
            f"trial {rec['trial']:>2} | streak in: {rec['streak_before']} | "
            f"bet ${rec['bet']} on {rec['predicted']:<5} | {rec['outcome']} | "
            f"balance ${rec['balance_after']} | attempts {result['n_attempts']}"
        )