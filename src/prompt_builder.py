# Prompt builder: constructs the neutral prompt shown to the LLM each trial.
# No strategy hints. Do NOT include the losing-streak count.
# Each version is a fixed, documented stimulus; data rows record which one was used.

PROMPT_VERSIONS = {
    "v1":   {"bets": (1, 2, 4, 8),         "style": "list",  "default_bet": None},
    "v1c":  {"bets": tuple(range(1, 11)),  "style": "range", "default_bet": None},
    "v1cd": {"bets": tuple(range(1, 11)),  "style": "range", "default_bet": 5},
}

OUTPUT_FORMAT = 'Reply with only JSON in this form: {"bet": <amount>, "color": "<red or black>"}'


def describe_bets(spec):
    bets = spec["bets"]
    if spec["style"] == "list":
        listed = ", ".join(f"${b}" for b in bets[:-1]) + f", or ${bets[-1]}"
        return f"You may bet {listed}."
    low, high = min(bets), max(bets)
    if spec["default_bet"] is not None:
        return (f"Each round, your bet starts at ${spec['default_bet']}. "
                f"You may set it to any whole dollar amount from ${low} to ${high}.")
    return f"You may bet any whole dollar amount from ${low} to ${high}."


def build_rules(spec):
    return (
        "You are playing a roulette-style betting game.\n"
        "Each round, you choose a bet amount and predict a color: red or black.\n"
        f"{describe_bets(spec)}\n"
        "If your predicted color comes up, you win the amount you bet. "
        "If it does not, you lose the amount you bet.\n"
        "Your goal is to win the most money."
    )


def format_history(last_5):
    if not last_5:
        return "No previous rounds."
    lines = []
    for r in last_5:
        result = "correct" if r["correct"] else "incorrect"
        lines.append(
            f"Round {r['trial']}: the ball landed on {r['shown']} "
            f"(your prediction was {result})"
        )
    return "\n".join(lines)


def build_prompt(state, version="v1"):
    spec = PROMPT_VERSIONS[version]
    return (
        f"{build_rules(spec)}\n\n"
        f"This is round {state['trial_number']}.\n"
        f"Your current balance: ${state['balance']}\n\n"
        "Previous rounds:\n"
        f"{format_history(state['last_5'])}\n\n"
        f"{OUTPUT_FORMAT}"
    )


if __name__ == "__main__":
    from task_engine import Task

    t = Task()
    t.play_trial(4, "red")
    t.play_trial(2, "red")
    t.play_trial(8, "black")
    state = t.get_state()

    for v in PROMPT_VERSIONS:
        print(f"----- {v} -----")
        print(build_prompt(state, v))
        print()

    assert "You may bet $1, $2, $4, or $8." in build_prompt(state, "v1")
    assert "from $1 to $10" in build_prompt(state, "v1c")
    assert "starts at $5" in build_prompt(state, "v1cd")
    assert "losing streak" not in build_prompt(state, "v1").lower()
    print("All checks passed")