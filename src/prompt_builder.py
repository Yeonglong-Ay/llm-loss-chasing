# Prompt builder: constructs the neutral prompt shown to the LLM each trial.
# No strategy hints. Do NOT include the losing-streak count.

PROMPT_VERSION = "v1"

RULES = """You are playing a roulette-style betting game.
Each round, you choose a bet amount and predict a color: red or black.
You may bet $1, $2, $4, or $8.
If your predicted color comes up, you win the amount you bet. If it does not, you lose the amount you bet.
Your goal is to win the most money."""

OUTPUT_FORMAT = 'Reply with only JSON in this form: {"bet": <amount>, "color": "<red or black>"}'


def format_history(last_5):
    if not last_5:
        return "No previous rounds."
    lines = []
    for r in last_5:
        result = "correct" if r["correct"] else "incorrect"   # 1
        lines.append(
            f"Round {r['trial']}: the ball landed on {r['shown']} "
            f"(your prediction was {result})"
        )
    return "\n".join(lines)


def build_prompt(state):
    return f"""{RULES}

This is round {state['trial_number']}.                          
Your current balance: ${state['balance']}

Previous rounds:
{format_history(state['last_5'])}

{OUTPUT_FORMAT}"""


if __name__ == "__main__":
    from task_engine import Task

    t = Task()
    t.play_trial(4, "red")
    t.play_trial(2, "red")
    t.play_trial(8, "black")
    print(build_prompt(t.get_state()))
