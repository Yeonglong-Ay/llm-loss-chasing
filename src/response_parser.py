import json
import re


def parse_decision(text, bet_options, colors=("red", "black")):
    """Turn the LLM's reply into (bet, color). Raise ValueError if invalid."""
    match = re.search(r"\{.*?\}", text, re.DOTALL)
    if match is None:
        raise ValueError("No JSON object found in reply")

    try:
        data = json.loads(match.group(0))          
        raw_bet = float(data["bet"])            
        color = str(data["color"]).strip().lower()
    except (KeyError, TypeError, json.JSONDecodeError) as e:
        raise ValueError(f"Malformed decision: {e}")

    if not raw_bet.is_integer():
        raise ValueError(f"Bet must be a whole number, got {raw_bet}")
    bet = int(raw_bet)

    if bet not in bet_options:                            
        raise ValueError(f"Bet {bet} not in {bet_options}")
    if color not in colors:                          
        raise ValueError(f"Color {color!r} not in {colors}")

    return bet, color


if __name__ == "__main__":
    opts = (1, 2, 4, 8)
    assert parse_decision('{"bet": 4, "color": "red"}', opts) == (4, "red")
    assert parse_decision('Sure! {"bet": "8", "color": "Black"}', opts) == (8, "black")
    assert parse_decision('```json\n{"bet": 2.0, "color": "red"}\n```', opts) == (2, "red")

    for bad in ['no json here', '{"bet": 5, "color": "red"}',
                '{"bet": 4, "color": "green"}', '{"bet": 4.5, "color": "red"}',
                '{"color": "red"}', '{"bet": 4, "color": ']:
        try:
            parse_decision(bad, opts)
            raise AssertionError(f"should have rejected: {bad}")
        except ValueError:
            pass

    print("All checks passed")