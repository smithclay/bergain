"""Rich progress display for bergain compose pipeline.

ProgressState is a shared mutable dataclass — tools write fields, the
ProgressDisplay daemon thread reads them via rich.live.Live at 500ms intervals.

Thread safety: GIL makes individual reads/writes atomic.  list.append() is
atomic.  Fine for a 500ms display refresh.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any

try:
    from rich.console import Console
    from rich.live import Live
    from rich.panel import Panel
    from rich.text import Text

    HAS_RICH = True
except ImportError:
    HAS_RICH = False


# ---------------------------------------------------------------------------
# ProgressState — shared mutable state, tools write / display reads
# ---------------------------------------------------------------------------


@dataclass
class ProgressState:
    """Mutable state bag that tools update and the display reads."""

    # Config (set once at start)
    brief: str = ""
    live: bool = False
    duration: int = 0
    model: str = ""

    # Phase progression
    phase: str = "setup"  # setup | composing | firing | exporting | analyzing | done

    # Milestones
    browse_done: bool = False
    tracks_done: bool = False
    clips_created: int = 0
    mix_done: bool = False

    # Time
    start_time: float | None = None
    elapsed_min: float = 0.0
    remaining_min: float = 0.0

    # Sections (reference to _live_history — tools mutate, display reads)
    sections: list[dict] = field(default_factory=list)

    # Latest sub-LM interaction
    latest_creative_prompt: str = ""
    latest_sub_lm_response: str = ""
    latest_sub_lm_reasoning: str = ""
    latest_guardrails: list[str] = field(default_factory=list)
    latest_energy_before_clamp: float | None = None
    latest_section: dict = field(default_factory=dict)

    # LLM usage heartbeat
    llm_calls: int = 0
    llm_tokens: int = 0

    # Analysis results
    analysis: dict[str, Any] = field(default_factory=dict)

    # RLM stream log — tools/patches append, TUI reads
    stream: list[dict] = field(default_factory=list)

    # Steering — TUI writes, compose_next reads and clears
    steer_direction: str = ""

    # Control — TUI writes, worker checks
    paused: bool = False
    abort: bool = False


# ---------------------------------------------------------------------------
# ProgressDisplay — background daemon thread
# ---------------------------------------------------------------------------

# Sparkline characters and color thresholds
_SPARK = "▁▂▃▄▅▆▇█"


def _spark_char(energy: float) -> str:
    idx = min(len(_SPARK) - 1, max(0, int(energy * (len(_SPARK) - 1))))
    return _SPARK[idx]


def _spark_color(energy: float) -> str:
    if energy < 0.3:
        return "blue"
    elif energy < 0.6:
        return "yellow"
    return "red"


def _energy_sparkline(sections: list[dict]) -> Text:
    """Build a colored sparkline Text from section energy values."""
    text = Text()
    for s in sections:
        e = s.get("energy", 0)
        text.append(_spark_char(e), style=_spark_color(e))
    return text


class ProgressDisplay:
    """Background daemon thread that renders compose progress via rich.live.Live."""

    def __init__(self, state: ProgressState):
        self.state = state
        self._live: Live | None = None
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()

    def start(self):
        if not HAS_RICH:
            return
        console = Console()
        self._live = Live(
            self._render(),
            console=console,
            refresh_per_second=2,
            transient=False,
        )
        self._live.start()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=2)
        if self._live:
            try:
                self._live.update(self._render())
                self._live.stop()
            except Exception:
                pass

    def _loop(self):
        while not self._stop_event.is_set():
            try:
                if self._live:
                    self._live.update(self._render())
            except Exception:
                pass
            self._stop_event.wait(0.5)

    def _render(self) -> Panel:
        s = self.state
        parts: list[Any] = []

        # Header: mode, elapsed/remaining, brief
        mode = "LIVE" if s.live else "PALETTE"
        time_str = ""
        if s.start_time:
            elapsed = (time.time() - s.start_time) / 60.0
            s.elapsed_min = elapsed
            if s.live and s.duration:
                remaining = max(0, s.duration - elapsed)
                s.remaining_min = remaining
                time_str = f" | {elapsed:.1f}/{s.duration} min"
            else:
                time_str = f" | {elapsed:.1f} min"

        brief_trunc = s.brief[:60] + ("..." if len(s.brief) > 60 else "")
        header = Text()
        header.append(f" {mode}", style="bold cyan")
        header.append(f"{time_str}", style="dim")
        header.append(f"  {brief_trunc}", style="italic")
        parts.append(header)

        # Heartbeat: LLM calls + tokens
        if s.llm_calls > 0:
            hb = Text("  ")
            hb.append(f"LLM calls: {s.llm_calls}", style="dim")
            if s.llm_tokens > 0:
                if s.llm_tokens >= 1_000_000:
                    tok_str = f"{s.llm_tokens / 1_000_000:.1f}M"
                elif s.llm_tokens >= 1_000:
                    tok_str = f"{s.llm_tokens / 1_000:.1f}k"
                else:
                    tok_str = str(s.llm_tokens)
                hb.append(f"  tokens: {tok_str}", style="dim")
            parts.append(hb)

        # Energy arc sparkline
        if s.sections:
            arc_line = Text("  Arc: ")
            arc_line.append_text(_energy_sparkline(s.sections))
            energies = [sec.get("energy", 0) for sec in s.sections]
            arc_line.append(f"  [{min(energies):.2f}–{max(energies):.2f}]", style="dim")
            parts.append(arc_line)

        # Milestones checklist
        checks = []
        checks.append(("browse", s.browse_done))
        checks.append(("tracks", s.tracks_done))
        checks.append((f"clips ({s.clips_created})", s.clips_created > 0))
        checks.append(("mix", s.mix_done))
        mile_text = Text("  ")
        for label, done in checks:
            mark = "✓" if done else "·"
            style = "green" if done else "dim"
            mile_text.append(f"{mark} {label}  ", style=style)
        parts.append(mile_text)

        # Latest section summary
        if s.sections:
            latest = s.sections[-1]
            sec_text = Text("  Now playing: ", style="bold")
            sec_text.append(f"{latest.get('section', '?')}", style="white bold")
            energy = latest.get("energy", 0)
            sec_text.append(f"  E={energy:.2f}", style=_spark_color(energy))

            # Guardrail delta
            if s.latest_energy_before_clamp is not None and s.latest_guardrails:
                delta = energy - s.latest_energy_before_clamp
                if abs(delta) > 0.001:
                    sec_text.append(
                        f" (was {s.latest_energy_before_clamp:.2f})", style="dim"
                    )

            key = latest.get("key", "")
            chords = latest.get("chords", [])
            if key:
                sec_text.append(f"  {key}", style="white")
            if chords:
                sec_text.append(f" {'/'.join(chords[:3])}", style="dim")
            bars = latest.get("bars", "")
            if bars:
                sec_text.append(f"  {bars}bar", style="dim")
            parts.append(sec_text)

            # Instrument breakdown for the section
            sec = s.latest_section or latest
            _STYLE_KEYS = ("drums", "bass", "pad", "stab", "texture")
            active = [
                f"{k}={sec[k]}" for k in _STYLE_KEYS if sec.get(k) and sec[k] != "none"
            ]
            silent = [k for k in _STYLE_KEYS if sec.get(k) == "none"]
            if active:
                inst_text = Text("  Instruments: ", style="dim")
                inst_text.append("  ".join(active), style="cyan")
                if silent:
                    inst_text.append(f"  (off: {', '.join(silent)})", style="dim")
                parts.append(inst_text)

        # DJ thinking — creative direction from conductor RLM
        if s.latest_creative_prompt:
            parts.append(Text(""))  # spacer
            think = Text("  DJ direction: ", style="bold magenta")
            think.append(s.latest_creative_prompt[:120], style="magenta")
            parts.append(think)

        # Sub-LM reasoning — the musical thinking behind the section
        if s.latest_sub_lm_reasoning:
            # Show the reasoning in a wrapped block
            reasoning_lines = s.latest_sub_lm_reasoning[:300].strip()
            reason = Text("  Sub-LM thinking: ", style="bold yellow")
            reason.append(reasoning_lines, style="yellow")
            parts.append(reason)
        elif s.latest_sub_lm_response:
            # Fallback: show raw response if no reasoning_content
            resp = Text("  Sub-LM response: ", style="bold yellow")
            resp.append(s.latest_sub_lm_response[:200], style="yellow")
            parts.append(resp)

        # Guardrails
        if s.latest_guardrails:
            gr = Text("  Guardrails: ", style="bold red")
            gr.append(", ".join(s.latest_guardrails), style="red")
            parts.append(gr)

        # Analysis results
        if s.analysis:
            scores = s.analysis.get("score", {}).get("scores", {})
            if scores:
                score_text = Text("  Scores: ", style="bold green")
                for name, val in scores.items():
                    if isinstance(val, (int, float)):
                        score_text.append(f"{name}={val:.1f} ", style="green")
                parts.append(score_text)

        # Combine into panel
        content = Text("\n").join(parts) if parts else Text("  Initializing...")
        return Panel(
            content,
            title=f"[bold]bergain[/bold] — {s.phase}",
            border_style="blue",
            padding=(0, 1),
        )


class PlainProgress:
    """Fallback progress display that uses plain print statements."""

    def __init__(self, state: ProgressState):
        self.state = state
        self._last_section_count = 0

    def start(self):
        mode = "LIVE" if self.state.live else "PALETTE"
        print(f"=== bergain {mode} ===")
        brief = self.state.brief[:100]
        print(f"  Brief: {brief}")

    def stop(self):
        pass

    def update(self):
        """Call periodically to print new sections."""
        s = self.state
        if len(s.sections) > self._last_section_count:
            for sec in s.sections[self._last_section_count :]:
                name = sec.get("section", "?")
                energy = sec.get("energy", 0)
                key = sec.get("key", "?")
                chords = "/".join(sec.get("chords", [])[:3])
                _STYLE_KEYS = ("drums", "bass", "pad", "stab", "texture")
                active = [
                    f"{k}={sec[k]}"
                    for k in _STYLE_KEYS
                    if sec.get(k) and sec[k] != "none"
                ]
                print(f"  Section: {name}  E={energy:.2f}  key={key}  {chords}")
                if active:
                    print(f"    {', '.join(active)}")
            self._last_section_count = len(s.sections)
        # Show DJ thinking when available
        if s.latest_creative_prompt and len(s.sections) > 0:
            print(f"  DJ: {s.latest_creative_prompt[:120]}")
            if s.latest_sub_lm_reasoning:
                print(f"  Thinking: {s.latest_sub_lm_reasoning[:200]}")
            if s.latest_guardrails:
                print(f"  Guardrails: {', '.join(s.latest_guardrails)}")
