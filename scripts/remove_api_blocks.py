#!/usr/bin/env python3
"""Remove extracted API blocks from main.py (Step 5).

Removes 3 blocks:
- Block A: lines 250-1270 (caches, constants, standings helpers)
- Block B: lines 1361-1620 (_train_model_background, _warmup_standings_cache, _predictions_catchup)
- Block C: lines 1662-EOF (Pydantic models + 39 endpoints)

Each replaced with a comment marker. Order: C first (highest), then B, then A.
"""

MAIN_PY = "app/main.py"

BLOCK_A_START = 250
BLOCK_A_END = 1270
BLOCK_B_START = 1361
BLOCK_B_END = 1620
BLOCK_C_START = 1662
BLOCK_C_END = None  # to end of file


def remove_blocks():
    with open(MAIN_PY, "r") as f:
        lines = f.readlines()

    total_before = len(lines)
    print(f"Before: {total_before} lines")
    block_c_end = BLOCK_C_END or total_before

    # Verify boundaries
    line_a = lines[BLOCK_A_START - 1].strip()
    print(f"Block A start ({BLOCK_A_START}): {line_a[:80]}")
    assert "predictions_cache" in line_a or "Simple in-memory" in line_a, f"Unexpected: {line_a}"

    line_b = lines[BLOCK_B_START - 1].strip()
    print(f"Block B start ({BLOCK_B_START}): {line_b[:80]}")
    assert "_train_model_background" in line_b or "async def" in line_b, f"Unexpected: {line_b}"

    line_c = lines[BLOCK_C_START - 1].strip()
    print(f"Block C start ({BLOCK_C_START}): {line_c[:80]}")
    assert "ETLSync" in line_c or "class" in line_c, f"Unexpected: {line_c}"

    # Remove Block C first (highest line numbers)
    replacement_c = "\n# Pydantic models + 39 API endpoints moved to app/routes/api.py\n"
    new_lines = lines[:BLOCK_C_START - 1] + [replacement_c]

    # Remove Block B
    replacement_b = "\n# Startup helpers (_train_model_background, _warmup_standings_cache, _predictions_catchup) moved to app/routes/api.py\n"
    new_lines = new_lines[:BLOCK_B_START - 1] + [replacement_b] + new_lines[BLOCK_B_END:]

    # Remove Block A
    replacement_a = "\n# Caches, constants, and standings/prediction helpers moved to app/routes/api.py\n"
    new_lines = new_lines[:BLOCK_A_START - 1] + [replacement_a] + new_lines[BLOCK_A_END:]

    total_after = len(new_lines)
    removed = total_before - total_after
    print(f"After: {total_after} lines (removed {removed})")

    with open(MAIN_PY, "w") as f:
        f.writelines(new_lines)

    print("Done.")


if __name__ == "__main__":
    remove_blocks()
