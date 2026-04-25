"""
Real-time terminal display for Panacea.
Shows agent communication live during both training and testing.
Uses Rich for color-coded, structured output.
"""

import sys
import io
from datetime import datetime
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.live import Live
from rich.rule import Rule
from rich import box

# Force UTF-8 on Windows so Rich unicode box chars render correctly
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

console = Console(force_terminal=True, legacy_windows=False)

# ── Color scheme per actor ────────────────────────────────────────────────────
COLORS = {
    "router":       "bold cyan",
    "specialist":   "bold green",
    "patient":      "bold yellow",
    "oversight":    "bold red",
    "mongo":        "dim white",
    "system":       "bold white",
    "synthesis":    "bold magenta",
    "training":     "bold blue",
    "reward":       "bold bright_yellow",
    "fraud":        "bold red",
    "approved":     "bold green",
    "rejected":     "bold red",
    "inter_agent":  "bold bright_cyan",
}

def _ts() -> str:
    return datetime.now().strftime("%H:%M:%S")


class PanaceaDisplay:
    """
    Central display controller.
    All agents, voice pipeline, training loop call methods on this class.
    """

    def session_header(self, patient_id: str, session_id: str, urgency: str, symptoms: list[str]):
        console.print()
        console.print(Panel(
            f"[bold white]Patient:[/] [yellow]{patient_id}[/]   "
            f"[bold white]Session:[/] [dim]{session_id[:12]}...[/]   "
            f"[bold white]Urgency:[/] [{'bold red' if urgency == 'critical' else 'bold yellow'}]{urgency.upper()}[/]\n"
            f"[bold white]Symptoms:[/] {', '.join(symptoms)}",
            title="[bold bright_white] PANACEA — Live Agent Communication [/]",
            border_style="bright_blue",
            box=box.DOUBLE,
        ))

    def router(self, specialists: list[str], urgency: str):
        console.print(
            f"[{_ts()}] [{COLORS['router']}]ROUTER[/]        "
            f"Activated [bold]{len(specialists)}[/] specialist(s): "
            f"[cyan]{', '.join(specialists)}[/]  |  urgency=[yellow]{urgency}[/]"
        )

    def specialist_question(self, specialty: str, role: str, text: str):
        console.print(
            f"[{_ts()}] [{COLORS['specialist']}]{specialty.upper():<18}[/] "
            f"[dim]{role}:[/] {text}"
        )

    def specialist_token(self, token: str):
        """Print a single streaming token without newline."""
        console.print(token, end="", highlight=False)

    def specialist_turn_done(self, specialty: str, turn: int):
        console.print(
            f"\n[{_ts()}] [{COLORS['specialist']}]{specialty.upper():<18}[/] "
            f"[dim]Turn {turn} complete[/]"
        )

    def patient_speech(self, text: str):
        console.print(
            f"[{_ts()}] [{COLORS['patient']}]PATIENT           [/] "
            f"[yellow]\"{text}\"[/]"
        )

    def patient_listening(self):
        console.print(
            f"[{_ts()}] [{COLORS['patient']}]PATIENT           [/] "
            f"[dim yellow]Listening...[/]"
        )

    def inter_agent_message(self, from_spec: str, to_spec: str, msg_type: str, content: str):
        console.print(
            f"[{_ts()}] [{COLORS['inter_agent']}]{from_spec} -> {to_spec:<12}[/] "
            f"[dim]{msg_type}:[/] {content[:90]}"
        )

    def synthesis(self, lead_role: str, specialist_count: int, resource_count: int):
        console.print(Rule(style="dim magenta"))
        console.print(
            f"[{_ts()}] [{COLORS['synthesis']}]SYNTHESIS         [/] "
            f"Lead: [bold]{lead_role}[/] | "
            f"Specialists: [bold]{specialist_count}[/] | "
            f"Resources: [bold]{resource_count}[/]"
        )

    def oversight_check(self, status: str, flags: list[str]):
        color = COLORS["approved"] if status == "APPROVED" else COLORS["rejected"]
        console.print(
            f"[{_ts()}] [{COLORS['oversight']}]OVERSIGHT         [/] "
            f"[{color}]{status}[/]"
            + (f" — flags: {flags}" if flags else " — No fraud detected ✓")
        )

    def decision(self, verdict: str, reward: float):
        color = COLORS["approved"] if verdict == "APPROVED" else COLORS["rejected"]
        console.print(
            f"[{_ts()}] [{COLORS['system']}]DECISION          [/] "
            f"[{color}]{verdict}[/]  |  "
            f"[{COLORS['reward']}]Reward: {reward:+.2f}[/]"
        )

    def mongo_save(self, collection: str, patient_id: str):
        console.print(
            f"[{_ts()}] [{COLORS['mongo']}]MONGO             [/] "
            f"Saved to [dim]{collection}[/] for patient [dim]{patient_id}[/]"
        )

    def diagnosis_summary(self, specialty: str, diagnosis: str, severity: str,
                          medications: list[dict], tests: list[str], follow_up: str):
        table = Table(box=box.SIMPLE, show_header=False, padding=(0, 1))
        table.add_column("Field", style="bold white", width=18)
        table.add_column("Value", style="white")

        sev_color = "red" if severity == "critical" else "yellow" if severity == "high" else "green"
        table.add_row("Specialty",  specialty)
        table.add_row("Diagnosis",  f"[bold]{diagnosis}[/]")
        table.add_row("Severity",   f"[{sev_color}]{severity.upper()}[/]")
        table.add_row("Medications",", ".join(
            f"{m['name']} {m.get('dose','')}" for m in medications[:4]
        ))
        table.add_row("Tests",      ", ".join(tests[:4]))
        table.add_row("Follow-up",  follow_up)

        console.print(Panel(table, title=f"[bold green] {specialty} Assessment [/]",
                            border_style="green", box=box.ROUNDED))

    def final_report(self, patient_name: str, primary_diagnosis: str,
                     all_medications: list[dict], all_tests: list[str], summary: str):
        meds_text = "\n".join(
            f"  • {m['name']} {m.get('dose','')} — {m.get('frequency','')} for {m.get('duration','')}"
            for m in all_medications
        )
        tests_text = "\n".join(f"  • {t}" for t in all_tests)

        console.print()
        console.print(Panel(
            f"[bold white]Patient:[/] {patient_name}\n"
            f"[bold white]Primary Diagnosis:[/] [bold red]{primary_diagnosis}[/]\n\n"
            f"[bold white]Medications:[/]\n{meds_text}\n\n"
            f"[bold white]Tests Ordered:[/]\n{tests_text}\n\n"
            f"[bold white]Summary:[/] [italic]{summary}[/]",
            title="[bold bright_white] FINAL MEDICAL REPORT [/]",
            border_style="bright_green",
            box=box.DOUBLE,
        ))

    # ── Training-specific display ─────────────────────────────────────────────

    def training_header(self, total_episodes: int):
        console.print()
        console.print(Panel(
            "[bold white]Oversight Agent — PPO Training[/]\n"
            f"Episodes: [cyan]{total_episodes}[/]",
            title="[bold bright_white] PANACEA — Training Mode [/]",
            border_style="bright_blue",
            box=box.DOUBLE,
        ))

    def training_episode(self, ep: int, total: int, patient_id: str, strategy: str):
        console.print(Rule(f"Episode {ep}/{total} — Patient {patient_id} — {strategy}", style="dim blue"))

    def sub_agent_claim(self, specialty: str, resource: str, claimed: float, actual: float):
        ratio = claimed / actual if actual else float("inf")
        flag  = " [bold red]INFLATED[/]" if ratio > 1.5 else ""
        console.print(
            f"[{_ts()}] [{COLORS['specialist']}]SUB-AGENT         [/] "
            f"{specialty} claims [bold]{claimed}[/] {resource} "
            f"(actual: {actual}){flag}"
        )

    def oversight_sql(self, sql: str, result: str):
        console.print(
            f"[{_ts()}] [{COLORS['oversight']}]OVERSIGHT SQL     [/] "
            f"[dim]{sql[:60]}[/] → {result[:60]}"
        )

    def training_reward(self, oversight_reward: float, sub_reward: float, loss: float):
        console.print(
            f"[{_ts()}] [{COLORS['reward']}]REWARD            [/] "
            f"Oversight: [green]{oversight_reward:+.2f}[/]  "
            f"Sub-agent: [red]{sub_reward:+.2f}[/]  "
            f"Loss: [dim]{loss:.4f}[/]"
        )

    def training_complete(self, episodes: int, catch_rate: float, avg_reward: float):
        console.print()
        console.print(Panel(
            f"[bold white]Episodes completed:[/] {episodes}\n"
            f"[bold white]Deception catch rate:[/] [bold green]{catch_rate:.1%}[/]\n"
            f"[bold white]Average reward:[/] {avg_reward:+.3f}",
            title="[bold bright_white] Training Complete [/]",
            border_style="bright_green",
            box=box.DOUBLE,
        ))

    def error(self, source: str, msg: str):
        console.print(
            f"[{_ts()}] [bold red]ERROR [{source}][/] {msg}"
        )

    def info(self, msg: str):
        console.print(f"[{_ts()}] [{COLORS['system']}]INFO              [/] {msg}")


# ── Singleton ─────────────────────────────────────────────────────────────────
display = PanaceaDisplay()
