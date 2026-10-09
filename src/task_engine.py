# Task engine: the predetermined sequence, balance, and streak tracking.
# It never talks to the LLM.

# Your human-task cycle: W = win, L = loss (24 trials)

import random

CYCLE = "W L W W W W L L W L L L L W W L L L L W W W W L".split()

COLORS = ("red", "black")
DEFAULT_BETS = (1, 2, 4, 8)   # matches the human task


LOSS_RUN_LENGTHS = [1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5]      # 45 losses
WIN_RUN_LENGTHS = [1, 1, 1, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6]    # 55 wins


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



def generate_sequence(seed):
    """Shuffle the runs, then alternate win run / loss run. Same seed -> same sequence."""
    rng = random.Random(seed)                  
    loss_runs = LOSS_RUN_LENGTHS.copy()
    win_runs = WIN_RUN_LENGTHS.copy()
    rng.shuffle(loss_runs)                        
    rng.shuffle(win_runs)

    sequence = []
    for i in range(len(loss_runs)):
        sequence += ["W"] * win_runs[i]     
        sequence += ["L"] * loss_runs[i]            
    sequence += ["W"] * win_runs[-1]         
    return sequence


class Task:
    def __init__(self, start_balance=100, n_cycles=3, bet_options=DEFAULT_BETS, sequence=None):
        self.sequence = sequence if sequence is not None else build_sequence(n_cycles)
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

        # generated sequences
    from collections import Counter

    g1 = generate_sequence(seed=1)
    assert len(g1) == 100
    assert g1.count("L") == 45
    assert g1[0] == "W" and g1[-1] == "W"
    assert generate_sequence(1) == g1                  # same seed -> same sequence
    assert generate_sequence(2) != g1                  # different seed -> different order

    # streak exposure is identical for every seed
    for seed in (1, 2, 3):
        s = generate_sequence(seed)
        counts = Counter(losing_streak_before(s[:i]) for i in range(len(s)))
        assert counts[1] == 15 and counts[2] == 12 and counts[3] == 9
        assert counts[4] == 6 and counts[5] == 3 and counts[0] == 55

    # Task can run on a generated sequence
    tg = Task(sequence=g1)
    while not tg.is_done():
        tg.play_trial(1, "red")
    assert tg.trial == 100

    print("All checks passed")
