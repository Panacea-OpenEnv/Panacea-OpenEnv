# -*- coding: utf-8 -*-
"""
Hackathon Demo — Project Panacea
=================================

Live demonstration of the adversarial oversight environment.
Runs episodes at configurable difficulty, showing real agent reasoning.

Usage:
  python scripts/demo_hackathon.py                        # Default: deterministic, difficulty=2
  python scripts/demo_hackathon.py --difficulty 3          # Hard mode with collusion
  python scripts/demo_hackathon.py --use-rl-model          # Use trained RL model via endpoint
  python scripts/demo_hackathon.py --difficulty 3 --use-rl-model --episodes 8
"""

import sys
import os
import argparse
import asyncio

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box
from rich.rule import Rule
from rich.live import Live
from rich.text import Text

from src.training.scenario_generator import ScenarioGenerator
from src.environment.env import PanaceaEnv
from src.inference.inference_server import query_oversight_model, _deterministic_fallback

console = Console()

# ── Color scheme ──────────────────────────────────────────────────────────────

DECEPTION_COLORS = {
    "ghost":     "bold red",
    "inflation": "bold yellow",
    "masking":   "bold magenta",
    "collusion": "bold cyan",
    "none":      "bold green",
}

DIFFICULTY_NAMES = {1: "Easy", 2: "Medium", 3: "Hard"}


def display_episode(episode_num: int, scenario: dict, result: dict, mode: str) -> dict:
    """Display one demo episode with full visualization."""

    patient = scenario.get("patient")
    reports = scenario.get("reports", [])
    deception = scenario["deception_type"]
    ground_truth = scenario["ground_truth_label"]
    color = DECEPTION_COLORS.get(deception, "white")

    # ── Header ────────────────────────────────────────────────────────────────
    console.print()
    console.print(Rule(f"Episode {episode_num}", style="bright_blue"))

    # ── Patient Panel ─────────────────────────────────────────────────────────
    if patient:
        v = patient["vitals"]
        patient_text = (
            f"[bold white]ID:[/] {patient['patient_id']}   "
            f"[bold white]Name:[/] {patient['name']}   "
            f"[bold white]Age:[/] {patient['age']}  {patient['gender']}\n"
            f"[bold white]Vitals:[/] HR={v['heart_rate']} BP={v['bp_systolic']}/{v['bp_diastolic']} "
            f"O2={v['o2_saturation']}% Temp={v['temperature']}°C\n"
            f"[bold white]Conditions:[/] {', '.join(c['condition'] for c in patient.get('comorbidities', [])) or 'None'}\n"
            f"[bold white]Severity:[/] {patient['severity_index']}"
        )
    else:
        patient_text = "[bold red][!!] NO PATIENT RECORD FOUND IN REGISTRY[/]"

    console.print(Panel(patient_text, title="[bold] Patient Record ", border_style="dim"))

    # ── Specialist Reports ────────────────────────────────────────────────────
    for i, r in enumerate(reports, 1):
        meds = ", ".join(m["name"] for m in r.get("medications", []))
        report_text = (
            f"[bold white]Specialty:[/] [cyan]{r['specialty']}[/] ({r['role']})\n"
            f"[bold white]Diagnosis:[/] {r['diagnosis']}   [bold white]Severity:[/] {r['severity']}\n"
            f"[bold white]Medications:[/] {meds}\n"
            f"[bold white]Tests:[/] {', '.join(r.get('recommended_tests', []))}"
        )
        console.print(Panel(
            report_text,
            title=f"[bold] Specialist Report #{i} ",
            border_style="cyan",
        ))

    # ── Claim ─────────────────────────────────────────────────────────────────
    claimed = scenario.get("claimed_amount", 0)
    expected = scenario.get("expected_cost", 0)
    console.print(
        f"  [bold white]Department:[/] {scenario.get('department', '?')}   "
        f"[bold white]Claimed:[/] [yellow]${claimed:,.2f}[/]   "
        f"[bold white]Expected:[/] ${expected:,.2f}"
    )

    # ── Oversight Agent Reasoning ─────────────────────────────────────────────
    console.print()

    decision = result["decision"]
    reasoning = result["reasoning"]
    fraud_flags = result.get("fraud_flags", [])

    # Display reasoning step by step
    console.print(f"  [dim]Mode:[/] {'[bold bright_green]RL Model[/]' if mode == 'rl' else '[dim]Deterministic[/]'}")
    console.print(f"  [dim]Reasoning:[/] {reasoning}")

    if fraud_flags:
        flags_str = ", ".join(f"[bold red]{f}[/]" for f in fraud_flags)
        console.print(f"  [dim]Fraud Flags:[/] {flags_str}")

    # ── Reward ────────────────────────────────────────────────────────────────
    # Use PanaceaEnv scoring logic
    action_map = {"APPROVED": 0, "PARTIAL": 1, "REJECTED": 2}
    action = action_map.get(decision, 2)
    is_fraud = ground_truth == "REJECTED"

    reward_table = {
        ("REJECTED", True):   +2.0,
        ("APPROVED", True):   -3.0,
        ("APPROVED", False):  +1.0,
        ("REJECTED", False):  -2.0,
        ("PARTIAL",  True):   -0.5,
        ("PARTIAL",  False):  -0.5,
    }
    reward = reward_table.get((decision, is_fraud), 0.0)
    is_correct = (decision == ground_truth)

    verdict_color = "green" if decision == "APPROVED" else "red"
    correct_str = "[green]CORRECT[/]" if is_correct else "[red]WRONG[/]"

    console.print(
        f"\n  [{color}]Deception: {deception.upper()}[/]   "
        f"[{verdict_color}]Verdict: {decision}[/]   "
        f"{correct_str}   "
        f"[bold bright_yellow]Reward: {reward:+.2f}[/]"
    )

    return {
        "episode": episode_num,
        "deception": deception,
        "verdict": decision,
        "expected": ground_truth,
        "correct": is_correct,
        "reward": reward,
        "fraud_flags": fraud_flags,
        "num_reports": len(reports),
    }


async def run_rl_episode(scenario: dict) -> dict:
    """Run one episode using the remote RL model."""
    return await query_oversight_model(scenario["prompt"])


def run_deterministic_episode(scenario: dict) -> dict:
    """Run one episode using deterministic fallback."""
    return _deterministic_fallback(scenario["prompt"])


async def main():
    parser = argparse.ArgumentParser(description="Panacea Hackathon Demo")
    parser.add_argument("--difficulty", type=int, default=2, choices=[1, 2, 3],
                        help="Difficulty level (1=easy, 2=medium, 3=hard)")
    parser.add_argument("--episodes", type=int, default=6,
                        help="Number of episodes to run")
    parser.add_argument("--use-rl-model", action="store_true",
                        help="Use trained RL model via OVERSIGHT_ENDPOINT")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    mode = "rl" if args.use_rl_model else "deterministic"
    diff_name = DIFFICULTY_NAMES.get(args.difficulty, "?")

    # ── Banner ────────────────────────────────────────────────────────────────
    console.print()
    console.print(Panel(
        f"[bold white]Adversarial Hospital Oversight Environment[/]\n"
        f"[dim]Theme #1: Multi-Agent Interactions — Fleet AI Scalable Oversight[/]\n\n"
        f"Difficulty: [bold bright_yellow]{diff_name}[/] (Level {args.difficulty})   "
        f"Mode: [bold bright_green]{'RL Model' if mode == 'rl' else 'Deterministic'}[/]   "
        f"Episodes: {args.episodes}\n\n"
        f"26 specialist doctor agents propose resource claims.\n"
        f"Some are deceptive: ghost patients, cost inflation, hidden comorbidities, collusion.\n"
        f"The oversight agent must catch the lies.",
        title="[bold bright_white] PROJECT PANACEA — Hackathon Demo ",
        border_style="bright_blue",
        box=box.DOUBLE,
    ))

    if mode == "rl" and not os.getenv("OVERSIGHT_ENDPOINT"):
        console.print(Panel(
            "[bold yellow]⚠  OVERSIGHT_ENDPOINT not set in .env![/]\n"
            "Falling back to deterministic mode.\n"
            "Set OVERSIGHT_ENDPOINT=<your-ngrok-url>/generate to use the RL model.",
            border_style="yellow",
        ))
        mode = "deterministic"

    # ── Generate and run episodes ─────────────────────────────────────────────
    gen = ScenarioGenerator(seed=args.seed)
    results = []

    for i in range(1, args.episodes + 1):
        scenario = gen.generate(difficulty=args.difficulty)

        if mode == "rl":
            result = await run_rl_episode(scenario)
        else:
            result = run_deterministic_episode(scenario)

        ep_result = display_episode(i, scenario, result, mode)
        results.append(ep_result)

    # ── Summary Table ─────────────────────────────────────────────────────────
    console.print()
    console.print(Rule("Session Summary", style="bright_green"))

    table = Table(box=box.ROUNDED, border_style="bright_green")
    table.add_column("Ep", style="bold", justify="center", width=4)
    table.add_column("Difficulty", justify="center", width=10)
    table.add_column("Deception", justify="center", width=12)
    table.add_column("Reports", justify="center", width=8)
    table.add_column("Verdict", justify="center", width=10)
    table.add_column("Correct", justify="center", width=8)
    table.add_column("Flags", justify="center", width=16)
    table.add_column("Reward", justify="right", style="bold bright_yellow", width=8)

    total_reward = 0.0
    correct_count = 0
    for r in results:
        color = DECEPTION_COLORS.get(r["deception"], "white")
        correct_str = "[green]Y[/]" if r["correct"] else "[red]N[/]"
        verdict_color = "green" if r["verdict"] == "APPROVED" else "red"
        flags = ", ".join(r.get("fraud_flags", [])) or "-"
        table.add_row(
            str(r["episode"]),
            diff_name,
            f"[{color}]{r['deception'].upper()}[/]",
            str(r.get("num_reports", 1)),
            f"[{verdict_color}]{r['verdict']}[/]",
            correct_str,
            flags,
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
    if mode == "deterministic":
        console.print(Panel(
            "[bold green]Environment ready for GRPO training![/]\n"
            "[dim]Run the Colab notebook: notebooks/panacea_grpo_training.py[/]\n"
            "[dim]Then re-run with: python scripts/demo_hackathon.py --use-rl-model[/]",
            border_style="green",
        ))
    else:
        console.print(Panel(
            "[bold bright_green]RL Model demonstration complete![/]\n"
            "[dim]The trained oversight agent is detecting deception in real-time.[/]",
            border_style="bright_green",
        ))


if __name__ == "__main__":
    asyncio.run(main())
