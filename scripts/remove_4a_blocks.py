#!/usr/bin/env python3
"""Remove Block A and Block B from main.py (Step 4a).

Block A: lines 6977-10764 (OPS DASHBOARD core + helpers + debug endpoints + ops.json)
Block B: lines 13604-16691 (ops logs, triggers, login, daily counts, alerts, incidents, debug log)

Replaces each block with a comment marker.
"""

MAIN_PY = "app/main.py"

BLOCK_A_START = 6977  # # ====...OPS DASHBOARD
BLOCK_A_END = 10764   # empty line after ops.json endpoint
BLOCK_B_START = 13604 # @app.get("/dashboard/ops/logs.json")
BLOCK_B_END = 16691   # last line of file (return {"ok": ...})


def remove_blocks():
    with open(MAIN_PY, "r") as f:
        lines = f.readlines()

    total_before = len(lines)
    print(f"Before: {total_before} lines")

    # Verify boundaries
    line_a = lines[BLOCK_A_START - 1].strip()
    print(f"Block A start ({BLOCK_A_START}): {line_a[:80]}")
    assert "===" in line_a, f"Unexpected Block A start: {line_a}"

    line_b = lines[BLOCK_B_START - 1].strip()
    print(f"Block B start ({BLOCK_B_START}): {line_b[:80]}")
    assert "dashboard/ops/logs" in line_b, f"Unexpected Block B start: {line_b}"

    # Remove Block B first (higher line numbers) to preserve Block A line positions
    # Keep lines before Block B (1..13603) + replacement + lines after Block B (16692+)
    block_b_replacement = "\n# Ops endpoints (triggers, login, alerts, incidents) moved to app/dashboard/ops_routes.py\n"
    new_lines = lines[:BLOCK_B_START - 1] + [block_b_replacement] + lines[BLOCK_B_END:]

    # Now remove Block A
    block_a_replacement = "\n# OPS DASHBOARD (core + helpers + debug endpoints) moved to app/dashboard/ops_routes.py\n\n"
    new_lines = new_lines[:BLOCK_A_START - 1] + [block_a_replacement] + new_lines[BLOCK_A_END:]

    total_after = len(new_lines)
    removed = total_before - total_after
    print(f"After: {total_after} lines (removed {removed})")

    with open(MAIN_PY, "w") as f:
        f.writelines(new_lines)

    print("Done.")


if __name__ == "__main__":
    remove_blocks()
