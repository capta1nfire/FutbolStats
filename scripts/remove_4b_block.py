#!/usr/bin/env python3
"""Remove 4b block from main.py (Step 4b).

The entire section from line 5102 (PIT DASHBOARD separator) to end of file
is replaced with a single comment marker.
"""

MAIN_PY = "app/main.py"
BLOCK_START = 5102  # # ====...PIT DASHBOARD


def remove_block():
    with open(MAIN_PY, "r") as f:
        lines = f.readlines()

    total_before = len(lines)
    print(f"Before: {total_before} lines")

    # Verify boundary
    line_start = lines[BLOCK_START - 1].strip()
    print(f"Block start ({BLOCK_START}): {line_start[:80]}")
    assert "===" in line_start, f"Unexpected: {line_start}"

    # Keep lines before block + replacement
    replacement = "\n# Dashboard views (PIT, TITAN, tables, predictions, analytics) moved to app/dashboard/dashboard_views_routes.py\n"
    new_lines = lines[:BLOCK_START - 1] + [replacement]

    total_after = len(new_lines)
    removed = total_before - total_after
    print(f"After: {total_after} lines (removed {removed})")

    with open(MAIN_PY, "w") as f:
        f.writelines(new_lines)

    print("Done.")


if __name__ == "__main__":
    remove_block()
