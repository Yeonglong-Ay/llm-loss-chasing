# Task engine: the predetermined sequence, balance, and streak tracking.
# It never talks to the LLM.

# Your human-task cycle: W = win, L = loss (24 trials)
CYCLE = "W L W W W W L L W L L L L W W L L L L W W W W L".split()

COLORS = ("red", "black")
DEFAULT_BETS = (1, 2, 4, 8)   # matches the human task


def build_sequence(n_cycles=3):
    """Repeat the 24-trial cycle to make a full session (72 trials)."""
    return CYCLE * n_cycles


def losing_streak_before(outcomes_so_far):
    """
    outcomes_so_far: list like ["W", "L", "L"] (trials already played).
    Return how many consecutive 'L' are at the END of the list.
    Examples: [] -> 0 ; ["W","L","L"] -> 2 ; ["L","L","W"] -> 0
    """
    streak = 0
    for outcome in reversed(outcomes_so_far):
        if outcome == "L":
            streak += 1
        else:
            break
    return streak


class Task:
    def __init__(self, start_balance=100, n_cycles=3, bet_options=DEFAULT_BETS):
        self.sequence = build_sequence(n_cycles)
        self.balance = start_balance
        self.bet_options = bet_options
        self.trial = 0           # index of the NEXT trial to play
        self.outcomes = []       # "W"/"L" results so far
        self.history = []        # one dict per played trial

    def is_done(self):
        return self.trial >= len(self.sequence)

    def get_state(self):
        return {
            "trial_number": self.trial + 1,
            "balance": self.balance,
            "losing_streak": losing_streak_before(self.outcomes),
            "last_5": self.history[-5:],
        }

    def play_trial(self, bet, predicted_color):
        if bet not in self.bet_options:
            raise ValueError(f"Bet must be one of {self.bet_options}, got {bet}")
        if predicted_color not in COLORS:
            raise ValueError(f"Color must be one of {COLORS}")

        streak_before = losing_streak_before(self.outcomes)
        outcome = self.sequence[self.trial]

        if outcome == "W":
            shown = predicted_color
            self.balance += bet
        else:
            shown = "black" if predicted_color == "red" else "red"
            self.balance -= bet

        record = {
            "trial": self.trial + 1,
            "streak_before": streak_before,
            "bet": bet,
            "predicted": predicted_color,
            "shown": shown,
            "outcome": outcome,
            "correct": outcome == "W",
            "balance_after": self.balance,
        }
        self.outcomes.append(outcome)
        self.history.append(record)
        self.trial += 1
        return record


if __name__ == "__main__":
    seq = build_sequence()
    assert len(seq) == 72
    assert len(build_sequence(5)) == 120
    assert losing_streak_before([]) == 0
    assert losing_streak_before(["W", "L", "L"]) == 2
    assert losing_streak_before(["L", "L", "W"]) == 0
    assert losing_streak_before(["L", "W", "L", "L", "L"]) == 3

    t = Task()
    r1 = t.play_trial(4, "red")        # trial 1 is a W
    assert r1["outcome"] == "W" and t.balance == 104
    r2 = t.play_trial(2, "red")        # trial 2 is an L
    assert r2["shown"] == "black" and t.balance == 102
    r3 = t.play_trial(8, "black")      # trial 3 is a W; streak going in was 1
    assert r3["streak_before"] == 1
    assert len(t.get_state()["last_5"]) == 3

    # the bet rule is configurable, and invalid bets are rejected
    t2 = Task(bet_options=(1, 2, 4, 8))
    try:
        t2.play_trial(5, "red")
        raise AssertionError("bet of 5 should have been rejected")
    except ValueError:
        pass

    # flat $1 bet through the whole session: 36 wins, 36 losses
    t3 = Task()
    while not t3.is_done():
        t3.play_trial(1, "red")
    assert t3.balance == 100

    print("All checks passed")
