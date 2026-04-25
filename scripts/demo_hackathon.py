# -*- coding: utf-8 -*-
"""
Hackathon Demo -- Project Panacea
=================================

Single-command demo that shows the adversarial oversight environment
in action. Runs 5 episodes with different deception types and displays
live agent communication in the terminal.

Usage:
  python scripts/demo_hackathon.py
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box
from rich.rule import Rule

from openenv_panacea.scenario_generator import generate_scenario
from openenv_panacea.reward import compute_reward

console = Console(force_terminal=True)

# ── Color scheme ──────────────────────────────────────────────────────────────

DECEPTION_COLORS = {
    "ghost":     "bold red",
    "inflation": "bold yellow",
    "masking":   "bold magenta",
    "none":      "bold green",
}


def run_demo_episode(episode_num: int, deception_type: str | None = None, seed: int = 42) -> dict:
    """Run one demo episode and display results."""

    scenario = generate_scenario(deception_type=deception_type, seed=seed)
    claim = scenario["claim"]
    patient = scenario["patient"]
    deception = scenario["deception"]
    expected = scenario["expected_verdict"]

    dtype = deception["type"]
    color = DECEPTION_COLORS.get(dtype, "white")

    # ── Header ────────────────────────────────────────────────────────────────
    console.print()
    console.print(Rule(f"Episode {episode_num}", style="bright_blue"))

    # ── Claim Panel ───────────────────────────────────────────────────────────
    console.print(Panel(
        f"[bold white]Claim ID:[/] {claim['claim_id']}   "
        f"[bold white]Dept:[/] [cyan]{claim['department']}[/]   "
        f"[bold white]Patient:[/] {claim['patient_id']}\n"
        f"[bold white]Resource:[/] {claim['requested_resource']}   "
        f"[bold white]Amount:[/] [yellow]${claim['claimed_amount']:,.2f}[/]   "
        f"[bold white]Protocol:[/] {claim['protocol']}",
        title="[bold] Sub-Agent Claim ",
        border_style="cyan",
    ))

    # ── Patient Record ────────────────────────────────────────────────────────
    if patient:
        v = patient["vitals"]
        console.print(
            f"  [dim]Patient:[/] {patient['name']}, Age {patient['age']}, "
            f"{patient['gender']}, {patient['blood_group']}"
        )
        console.print(
            f"  [dim]Vitals:[/] HR={v['heart_rate']} BP={v['bp_systolic']}/{v['bp_diastolic']} "
            f"O2={v['o2_saturation']}% Temp={v['temperature']}°C"
        )
        visible = scenario.get("visible_comorbidities", [])
        if visible:
            console.print(f"  [dim]Comorbidities:[/] {', '.join(c['condition'] for c in visible)}")
        console.print(f"  [dim]Severity:[/] {patient['severity_index']}")
    else:
        console.print("  [bold red][!!] NO PATIENT RECORD FOUND IN REGISTRY[/]")

    # ── Oversight Agent Reasoning ─────────────────────────────────────────────
    console.print()

    # Simulate oversight reasoning based on ground truth
    reasoning_steps = []
    verdict = expected  # Perfect agent for demo purposes

    if dtype == "ghost":
        reasoning_steps = [
            "[dim]Step 1:[/] Searching patient registry...",
            f"[bold red]Step 2:[/] Patient {claim['patient_id']} NOT FOUND in hospital database.",
            "[bold red]Step 3:[/] This is a GHOST PATIENT — fabricated identity.",
            "[bold]Verdict: REJECTED[/] — Ghost patient detected.",
        ]
    elif dtype == "inflation":
        expected_cost = deception["expected_cost"]
        inflation_factor = deception["inflation_factor"]
        reasoning_steps = [
            "[dim]Step 1:[/] Patient confirmed in registry.",
            f"[dim]Step 2:[/] Severity index = {patient['severity_index']}, expected cost ≈ ${expected_cost:,.2f}",
            f"[bold yellow]Step 3:[/] Claimed ${claim['claimed_amount']:,.2f} is {inflation_factor:.1f}× the expected amount!",
            f"[bold]Verdict: REJECTED[/] — Cost inflation detected ({inflation_factor:.1f}× over expected).",
        ]
    elif dtype == "masking":
        hidden = deception["hidden_comorbidities"]
        hidden_names = [c["condition"] for c in hidden]
        reasoning_steps = [
            "[dim]Step 1:[/] Patient confirmed in registry.",
            "[dim]Step 2:[/] Checking full comorbidity records...",
            f"[bold magenta]Step 3:[/] CRITICAL conditions found but OMITTED from claim: {', '.join(hidden_names)}",
            "[bold]Verdict: REJECTED[/] — Comorbidity masking detected.",
        ]
    else:
        reasoning_steps = [
            "[dim]Step 1:[/] Patient confirmed in registry.",
            "[dim]Step 2:[/] Vitals and severity check passed.",
            "[dim]Step 3:[/] Amount within expected range for protocol.",
            "[bold green]Verdict: APPROVED[/] — Claim verified clean.",
        ]

    for step in reasoning_steps:
        console.print(f"  {step}")

    # ── Reward ────────────────────────────────────────────────────────────────
    reasoning_text = " ".join(reasoning_steps)
    reward = compute_reward(verdict, expected, dtype, reasoning_text)

    verdict_color = "green" if verdict == "APPROVED" else "red"
    console.print(
        f"\n  [{color}]Deception: {dtype.upper()}[/]   "
        f"[{verdict_color}]Verdict: {verdict}[/]   "
        f"[bold bright_yellow]Reward: {reward:+.2f}[/]"
    )

    return {
        "episode": episode_num,
        "deception": dtype,
        "verdict": verdict,
        "expected": expected,
        "correct": verdict == expected,
        "reward": reward,
    }


def main():
    console.print()
    console.print(Panel(
        "[bold white]Adversarial Hospital Oversight Environment[/]\n"
        "[dim]Theme #1: Multi-Agent Interactions — Fleet AI Scalable Oversight[/]\n\n"
        "26 specialist doctor agents propose resource claims.\n"
        "Some are deceptive: ghost patients, cost inflation, hidden comorbidities.\n"
        "The oversight agent must catch the lies.",
        title="[bold bright_white] PROJECT PANACEA — Hackathon Demo ",
        border_style="bright_blue",
        box=box.DOUBLE,
    ))

    # Run one episode of each deception type, then one random
    episodes = [
        ("ghost", 100),
        ("inflation", 200),
        ("masking", 300),
        ("none", 400),
        (None, 500),  # Random
    ]

    results = []
    for i, (dtype, seed) in enumerate(episodes, 1):
        result = run_demo_episode(i, deception_type=dtype, seed=seed)
        results.append(result)

    # ── Summary Table ─────────────────────────────────────────────────────────
    console.print()
    console.print(Rule("Session Summary", style="bright_green"))

    table = Table(box=box.ROUNDED, border_style="bright_green")
    table.add_column("Episode", style="bold", justify="center")
    table.add_column("Deception", justify="center")
    table.add_column("Verdict", justify="center")
    table.add_column("Correct", justify="center")
    table.add_column("Reward", justify="right", style="bold bright_yellow")

    total_reward = 0.0
    correct_count = 0
    for r in results:
        color = DECEPTION_COLORS.get(r["deception"], "white")
        correct_str = "[green]✓[/]" if r["correct"] else "[red]✗[/]"
        verdict_color = "green" if r["verdict"] == "APPROVED" else "red"
        table.add_row(
            str(r["episode"]),
            f"[{color}]{r['deception'].upper()}[/]",
            f"[{verdict_color}]{r['verdict']}[/]",
            correct_str,
            f"{r['reward']:+.2f}",
        )
        total_reward += r["reward"]
        correct_count += int(r["correct"])

    console.print(table)
    console.print(
        f"\n  [bold]Accuracy:[/] {correct_count}/{len(results)} "
        f"({correct_count/len(results)*100:.0f}%)   "
        f"[bold]Total Reward:[/] [bold bright_yellow]{total_reward:+.2f}[/]"
    )

    console.print()
    console.print(Panel(
        "[bold green]Environment ready for GRPO training![/]\n"
        "[dim]Run the Colab notebook: notebooks/panacea_grpo_training.py[/]",
        border_style="green",
    ))


if __name__ == "__main__":
    main()
