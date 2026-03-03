#!/usr/bin/env python3
"""
Task Scheduler using PuLP (Mixed Integer Linear Programming)

Usage:
    python task_scheduler.py --operations ops.csv --process process1.csv process2.csv [--output schedule.csv]

Operations CSV format:
    module, operation, duration_seconds

Job CSV format:
    module, operation, parameters
    (parameters column is optional; used for wait operations with duration=<minutes>)
"""

import argparse
import sys
import math
import logging
import pandas as pd
import pulp


# ─────────────────────────────────────────────────────────────────────────────
# CSV loading helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_operations(path: str) -> dict:
    """
    Load the operations catalogue (csv file).
    Returns a dict keyed by operation -> {"module": module, "duration_seconds": dur}
    The special wait operation has key 'wait'.
    """
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]

    req_columns = {"module", "operation", "duration_seconds"}
    if not req_columns.issubset(set(df.columns)):
        sys.exit(
            f"[ERROR] Operations file must contain columns: {req_columns}\n"
            f"        Found: {list(df.columns)}"
        )

    op_catalogue = {}
    for _, row in df.iterrows():
        module = None if pd.isna(row["module"]) or str(row["module"]).strip() == "" else str(row["module"]).strip()
        op = str(row["operation"]).strip()
        dur = float(row["duration_seconds"])
        op_catalogue[op] = {"module": module, "duration_seconds": dur}

    return op_catalogue


def parse_wait_duration(parameters: str) -> float:
    """
    Parse 'duration=<minutes>' from the parameters field.
    Returns duration in seconds.
    """
    if pd.isna(parameters) or str(parameters).strip() == "":
        raise ValueError("Wait operation requires a parameters value like 'duration=5'")
    for part in str(parameters).split(","):
        part = part.strip()
        if part.lower().startswith("duration="):
            minutes = float(part.split("=", 1)[1].strip().removesuffix("min"))
            return minutes * 60.0
    raise ValueError(f"Could not parse duration from parameters: '{parameters}'")


def load_process(path: str, op_catalogue: dict) -> list:
    """
    Load a single process CSV.
    Returns a list of dicts:
        {process_file, step, module, operation, duration_seconds}
    """
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]

    required = {"module", "operation", "parameters"}
    if not required.issubset(set(df.columns)):
        sys.exit(
            f"[ERROR] Process file '{path}' must contain columns: module, operation, parameters\n"
            f"        Found: {list(df.columns)}"
        )

    steps = []
    for idx, row in df.iterrows():
        module = None if pd.isna(row["module"]) or str(row["module"]).strip() == "" else str(row["module"]).strip()
        op = str(row["operation"]).strip()
        params = row["parameters"]

        # Determine duration
        if op.lower() == "wait":
            # Special wait operation – duration comes from parameters
            duration = parse_wait_duration(params)
            module = None  # wait has no module
        else:
            if op not in op_catalogue:
                sys.exit(
                    f"[ERROR] Operation {op!r} in '{path}' row {idx+2} "
                    f"not found in operations catalogue."
                )
            duration = op_catalogue[op]["duration_seconds"]

        steps.append({
            "process_file": path,
            "step": idx,
            "module": module if module else "",
            "operation": op,
            "duration_seconds": duration,
            "parameters": str(params) if not pd.isna(params) else "",
        })

    return steps


# ─────────────────────────────────────────────────────────────────────────────
# Scheduler (MILP via PuLP)
# ─────────────────────────────────────────────────────────────────────────────

def build_schedule(all_jobs: list[list[dict]]) -> pd.DataFrame:
    """
    Solve the job scheduling problem using MILP (PuLP).

    Decision variables
    ------------------
    start[j][s]  : continuous, start time of step s in job j  (seconds)
    order[j1,s1,j2,s2] : binary, 1 if task (j1,s1) precedes (j2,s2) on the
                         same module (only created when module is non-empty).

    Objective
    ---------
    Minimise makespan (= max finish time across all tasks).
    """

    prob = pulp.LpProblem("TaskScheduler", pulp.LpMinimize)

    # Flatten tasks
    tasks = []  # (job_idx, step_idx, module, op, duration)
    for j, job in enumerate(all_jobs):
        for s, step in enumerate(job):
            tasks.append((j, s, step["module"], step["operation"], step["duration_seconds"]))

    # Big-M  (sum of all durations is a safe upper bound)
    M = sum(t[4] for t in tasks) + 1

    # ── Start-time variables ──────────────────────────────────────────────────
    start = {}
    for j, s, mod, op, dur in tasks:
        start[(j, s)] = pulp.LpVariable(f"start_{j}_{s}", lowBound=0)

    # ── Makespan variable ─────────────────────────────────────────────────────
    makespan = pulp.LpVariable("makespan", lowBound=0)
    prob += makespan  # objective

    # ── Precedence within each job (sequential steps) ─────────────────────────
    for j, job in enumerate(all_jobs):
        for s in range(len(job) - 1):
            dur_s = job[s]["duration_seconds"]
            prob += start[(j, s + 1)] >= start[(j, s)] + dur_s, f"prec_{j}_{s}"

    # ── Makespan lower bound ──────────────────────────────────────────────────
    for j, s, mod, op, dur in tasks:
        prob += makespan >= start[(j, s)] + dur, f"mkspan_{j}_{s}"

    # ── No-overlap on shared modules ──────────────────────────────────────────
    # Group tasks by module (skip empty-module = wait tasks)
    from collections import defaultdict
    module_tasks = defaultdict(list)
    for j, s, mod, op, dur in tasks:
        if mod:  # skip wait / no-module tasks
            module_tasks[mod].append((j, s, dur))

    order_vars = {}
    for mod, mod_tasks in module_tasks.items():
        n = len(mod_tasks)
        for i in range(n):
            for k in range(i + 1, n):
                j1, s1, d1 = mod_tasks[i]
                j2, s2, d2 = mod_tasks[k]
                if j1 == j2 and abs(s1 - s2) <= 1:
                    # Same job adjacent steps — already covered by precedence;
                    # but we still add no-overlap if they share the module
                    pass
                # Binary: y=1 means (j1,s1) before (j2,s2)
                y = pulp.LpVariable(f"ord_{j1}_{s1}_{j2}_{s2}", cat="Binary")
                order_vars[(j1, s1, j2, s2)] = y
                # (j1,s1) finishes before (j2,s2) starts  OR  vice versa
                prob += start[(j1, s1)] + d1 <= start[(j2, s2)] + M * (1 - y), \
                       f"noovlp_a_{j1}_{s1}_{j2}_{s2}"
                prob += start[(j2, s2)] + d2 <= start[(j1, s1)] + M * y, \
                       f"noovlp_b_{j1}_{s1}_{j2}_{s2}"

    # ── Solve ─────────────────────────────────────────────────────────────────
    solver = pulp.getSolver("PULP_CBC_CMD", msg=False, timeLimit=120)
    status = prob.solve(solver)

    if pulp.LpStatus[status] not in ("Optimal", "Not Solved"):
        print(f"[WARNING] Solver status: {pulp.LpStatus[status]}")

    # ── Extract solution ──────────────────────────────────────────────────────
    rows = []
    for j, job in enumerate(all_jobs):
        for s, step in enumerate(job):
            t_start = pulp.value(start[(j, s)])
            if t_start is None:
                t_start = 0.0
            t_start = round(t_start, 3)
            t_end = round(t_start + step["duration_seconds"], 3)
            rows.append({
                "process_file": step["process_file"],
                "job_index": j,
                "step": (s+1), # 1-based step index for nicer output
                "module": step["module"],
                "operation": step["operation"],
                "parameters": step["parameters"],
                "duration_seconds": step["duration_seconds"],
                "start_time_seconds": t_start,
                "end_time_seconds": t_end,
            })

    df = pd.DataFrame(rows)
    df.sort_values(["start_time_seconds", "job_index", "step"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Pretty-print schedule
# ─────────────────────────────────────────────────────────────────────────────

def print_schedule(df: pd.DataFrame):
    import os
    col_widths = {
        "process_file": max(8, df["process_file"].apply(lambda x: len(os.path.basename(x))).max()),
        "module": max(6, df["module"].str.len().max()),
        "operation": max(9, df["operation"].str.len().max()),
        "duration_seconds": 17,
        "start_time_seconds": 19,
        "end_time_seconds": 17,
    }
    import os as _os
    header = (
        f"{'Process File':<{col_widths['process_file']}}  "
        f"{'Step':>4}  "
        f"{'Module':<{col_widths['module']}}  "
        f"{'Operation':<{col_widths['operation']}}  "
        f"{'Duration(s)':>{col_widths['duration_seconds']}}  "
        f"{'Start(s)':>{col_widths['start_time_seconds']}}  "
        f"{'End(s)':>{col_widths['end_time_seconds']}}"
    )
    sep = "─" * len(header)
    print("\n" + sep)
    print(header)
    print(sep)
    for _, row in df.iterrows():
        jf = _os.path.basename(row["process_file"])
        print(
            f"{jf:<{col_widths['process_file']}}  "
            f"{row['step']:>4}  "
            f"{row['module']:<{col_widths['module']}}  "
            f"{row['operation']:<{col_widths['operation']}}  "
            f"{row['duration_seconds']:>{col_widths['duration_seconds']}.2f}  "
            f"{row['start_time_seconds']:>{col_widths['start_time_seconds']}.2f}  "
            f"{row['end_time_seconds']:>{col_widths['end_time_seconds']}.2f}"
        )
    print(sep)
    makespan = df["end_time_seconds"].max()
    print(f"  Makespan: {makespan:.2f} s  ({makespan/60:.2f} min)\n")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Optimised task scheduler using Mixed Integer Linear Programming (PuLP).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples
--------
  # Two processes, default output
  python task_scheduler.py --operations operations.csv --process process1.csv process2.csv

  # Save schedule to file
  python task_scheduler.py --operations operations.csv --process process1.csv process2.csv --output schedule.csv

Operations CSV columns : module, operation, duration_seconds
Process CSV columns        : module, operation[, parameters]
  - Use operation='wait' (no module) with parameters='duration=<minutes>' for wait steps.
""",
    )
    parser.add_argument(
        "--operations", "-ops",
        required=True,
        metavar="FILE",
        help="Path to operations catalogue CSV.",
    )
    parser.add_argument(
        "--process", "-j",
        required=True,
        nargs="+",
        metavar="FILE",
        help="One or more process CSV files.",
    )
    parser.add_argument(
        "--output", "-o",
        default=None,
        metavar="FILE",
        help="Optional output CSV file name (default: print to stdout only).",
    )
    parser.add_argument(
        "--print", "-p",
        action="store_true",
        dest="print_schedule",
        help="If specified, the computed schedule will be printed to stdout.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # configure basic logging to stdout
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

    logging.info(f"Loading operations catalogue: {args.operations}")
    ops_catalogue = load_operations(args.operations)
    logging.info(f"{len(ops_catalogue)} operation(s) loaded.")

    all_processes = []
    for process_path in args.process:
        logging.info(f"Loading process: {process_path}")
        this_proc = load_process(process_path, ops_catalogue)
        all_processes.append(this_proc)
        logging.info(f"{len(this_proc)} step(s) loaded.")

    logging.info(
        "\nSolving scheduling problem (%d tasks across %d processes) …",
        sum(len(j) for j in all_processes),
        len(all_processes),
    )
    schedule_df = build_schedule(all_processes)

    if args.print_schedule:
        print_schedule(schedule_df)

    if args.output:
        schedule_df.to_csv(args.output, index=False)
        logging.info(f"Schedule written to: {args.output}")

    return schedule_df


if __name__ == "__main__":
    main()

