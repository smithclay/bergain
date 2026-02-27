"""Textual TUI for bergain — RLM control dashboard.

Launch with `bergain` (no args). Two screens:
  1. LaunchScreen — brief, mode, config, OSC check
  2. ComposeScreen — streaming RLM output, steer input, pause/abort
"""

import glob as glob_mod
import json
import os
import time as _time

from textual import work
from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical
from textual.screen import Screen
from textual.widgets import (
    Button,
    Input,
    Label,
    ListItem,
    ListView,
    RadioButton,
    RadioSet,
    RichLog,
    Static,
    TextArea,
)

from .progress import ProgressState


# Sparkline helper (re-use logic from progress.py)
_SPARK = "▁▂▃▄▅▆▇█"


def _spark_char(energy: float) -> str:
    idx = min(len(_SPARK) - 1, max(0, int(energy * (len(_SPARK) - 1))))
    return _SPARK[idx]


# ---------------------------------------------------------------------------
# LaunchScreen
# ---------------------------------------------------------------------------


class LaunchScreen(Screen):
    """Brief, mode, config, OSC check — the starting point."""

    DEFAULT_CSS = """
    LaunchScreen {
        align: center middle;
    }

    #launch-container {
        width: 72;
        max-height: 42;
        border: solid $accent;
        padding: 1 2;
    }

    #title {
        text-align: center;
        text-style: bold;
        margin-bottom: 1;
    }

    #brief {
        height: 4;
        margin-bottom: 1;
    }

    .field-row {
        height: 3;
        margin-bottom: 0;
    }

    .field-label {
        width: 14;
        padding-top: 1;
    }

    .field-input {
        width: 1fr;
    }

    #mode-set {
        height: 3;
        margin-bottom: 1;
    }

    #osc-status {
        margin: 1 0;
    }

    #start {
        margin: 1 0;
        width: 100%;
    }

    #recent-label {
        margin-top: 1;
        color: $text-muted;
    }

    #recent {
        height: 6;
        margin-bottom: 1;
    }
    """

    _osc_connected: bool = False

    def compose(self) -> ComposeResult:
        default_model = os.environ.get("BERGAIN_MODEL", "openrouter/openai/gpt-5")

        with Vertical(id="launch-container"):
            yield Static("bergain DJ Console", id="title")

            yield Label("Brief:")
            yield TextArea(id="brief")

            yield Label("Mode:")
            yield RadioSet(
                RadioButton("Palette", value=True, id="mode-palette"),
                RadioButton("Live", id="mode-live"),
                id="mode-set",
            )

            with Horizontal(classes="field-row"):
                yield Label("Model:", classes="field-label")
                yield Input(value=default_model, id="model", classes="field-input")

            with Horizontal(classes="field-row"):
                yield Label("Sub-model:", classes="field-label")
                yield Input(
                    placeholder="(same as model)",
                    id="sub-model",
                    classes="field-input",
                )

            with Horizontal(classes="field-row"):
                yield Label("Duration:", classes="field-label")
                yield Input(value="60", id="duration", classes="field-input")

            with Horizontal(classes="field-row"):
                yield Label("Iterations:", classes="field-label")
                yield Input(value="30", id="iterations", classes="field-input")

            yield Static("Ableton: checking...", id="osc-status")
            yield Button("Start Composing", id="start", variant="primary")

            yield Static("Recent sessions:", id="recent-label")
            yield ListView(id="recent")

    def on_mount(self) -> None:
        self._check_osc()
        self._load_recent()

    @work(thread=True)
    def _check_osc(self) -> None:
        """Check OSC connection to Ableton in a background thread."""
        try:
            from .session import Session

            session = Session()
            s = session.status()
            tempo = s.get("tempo", "?")
            tracks = len(s.get("tracks", []))
            session.close()
            self.app.call_from_thread(
                self._update_osc_status,
                f"Ableton: Connected -- {tempo} BPM, {tracks} tracks",
                True,
            )
        except Exception as e:
            self.app.call_from_thread(
                self._update_osc_status,
                f"Ableton: Not connected -- {e}",
                False,
            )

    def _update_osc_status(self, text: str, connected: bool) -> None:
        widget = self.query_one("#osc-status", Static)
        style = "green" if connected else "red"
        widget.update(f"[{style}]{text}[/{style}]")
        self._osc_connected = connected

    def _load_recent(self) -> None:
        """Load recent session JSON files into the recent list."""
        output_dir = "./output/compose/"
        lv = self.query_one("#recent", ListView)
        try:
            files = sorted(
                glob_mod.glob(os.path.join(output_dir, "*.json")), reverse=True
            )[:10]
            for f in files:
                try:
                    with open(f) as fh:
                        data = json.load(fh)
                    brief = data.get("brief", "?")[:60]
                    name = os.path.basename(f).replace(".json", "")
                    lv.append(ListItem(Label(f"{name} -- {brief}"), name=f))
                except Exception:
                    pass
        except Exception:
            pass

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        """Fill brief from selected recent session."""
        path = event.item.name
        if path:
            try:
                with open(path) as f:
                    data = json.load(f)
                brief_area = self.query_one("#brief", TextArea)
                brief_area.clear()
                brief_area.insert(data.get("brief", ""))
            except Exception:
                pass

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "start":
            brief_area = self.query_one("#brief", TextArea)
            brief = brief_area.text.strip()
            if not brief:
                self.notify("Enter a brief first", severity="warning")
                return

            mode_live = self.query_one("#mode-live", RadioButton).value
            model = self.query_one("#model", Input).value
            sub_model = self.query_one("#sub-model", Input).value or None
            duration = int(self.query_one("#duration", Input).value or "60")
            iterations = int(self.query_one("#iterations", Input).value or "30")

            config = {
                "brief": brief,
                "live": mode_live,
                "model": model,
                "sub_model": sub_model,
                "duration": duration,
                "max_iterations": iterations,
            }
            self.app.push_screen(ComposeScreen(config))


# ---------------------------------------------------------------------------
# ComposeScreen
# ---------------------------------------------------------------------------


class ComposeScreen(Screen):
    """RLM streaming dashboard with steer input."""

    DEFAULT_CSS = """
    ComposeScreen {
        layout: vertical;
    }

    #compose-header {
        height: 1;
        background: $surface;
        padding: 0 1;
    }

    #stream {
        height: 1fr;
        border: solid $primary;
        padding: 0 1;
    }

    #status-bar {
        height: 1;
        background: $surface;
        padding: 0 1;
    }

    #steer {
        height: 3;
        border: solid $secondary;
    }
    """

    BINDINGS = [
        ("space", "toggle_pause", "Pause"),
        ("q", "request_abort", "Abort"),
    ]

    _SPINNER = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"

    def __init__(self, config: dict) -> None:
        super().__init__()
        self.config = config
        self._state: ProgressState | None = None
        self._stream_cursor = 0
        self._spinner_idx = 0
        self._done_rendered = False

    def compose(self) -> ComposeResult:
        mode = "LIVE" if self.config.get("live") else "PALETTE"
        yield Static(
            f"bergain -- {mode} -- starting...",
            id="compose-header",
        )
        yield RichLog(id="stream", auto_scroll=True, highlight=True, markup=True)
        yield Static("", id="status-bar")
        yield Input(
            placeholder="steer> type direction, Enter to send",
            id="steer",
        )

    def on_mount(self) -> None:
        self._run_compose()
        self.set_interval(0.2, self._refresh_display)

    @work(thread=True, exclusive=True)
    def _run_compose(self) -> None:
        """Run the compose pipeline in a background thread."""
        import argparse

        state = ProgressState(
            brief=self.config["brief"],
            live=self.config.get("live", False),
            duration=self.config.get("duration", 60),
            model=self.config.get("model", ""),
        )
        self._state = state

        args = argparse.Namespace(
            brief=self.config["brief"],
            brief_file=None,
            model=self.config.get("model", "openrouter/openai/gpt-5"),
            sub_model=self.config.get("sub_model"),
            max_iterations=self.config.get("max_iterations", 30),
            max_llm_calls=None,
            min_clips=None,
            live=self.config.get("live", False),
            duration=self.config.get("duration", 60),
            dry_run=False,
            skip_export=False,
            analyze=False,
            bars_per_scene=16,
            no_progress=True,
            output_dir="./output/compose/",
            optimized=None,
        )

        try:
            from .cli import cmd_compose

            cmd_compose(args, progress_override=state)
        except KeyboardInterrupt:
            state.stream.append(
                {
                    "type": "step",
                    "content": "Composition interrupted.",
                    "timestamp": _time.time(),
                }
            )
        except Exception as e:
            import traceback

            state.stream.append(
                {
                    "type": "error",
                    "content": f"Compose error: {e}",
                    "timestamp": _time.time(),
                }
            )
            state.stream.append(
                {
                    "type": "error",
                    "content": traceback.format_exc(),
                    "timestamp": _time.time(),
                }
            )
        finally:
            state.phase = "done"

    def _refresh_display(self) -> None:
        """Poll ProgressState and update TUI widgets."""
        if self._state is None:
            return

        s = self._state

        # Update header with spinner
        mode = "LIVE" if s.live else "PALETTE"
        time_str = ""
        if s.start_time:
            elapsed = (_time.time() - s.start_time) / 60.0
            if s.live and s.duration:
                time_str = f" {elapsed:.1f}/{s.duration} min"
            else:
                time_str = f" {elapsed:.1f} min"

        tok_str = ""
        if s.llm_tokens >= 1_000_000:
            tok_str = f"{s.llm_tokens / 1_000_000:.1f}M tok"
        elif s.llm_tokens >= 1_000:
            tok_str = f"{s.llm_tokens / 1_000:.1f}k tok"
        elif s.llm_tokens > 0:
            tok_str = f"{s.llm_tokens} tok"

        stats = " | ".join(
            p
            for p in [
                f"{s.llm_calls} calls" if s.llm_calls else "",
                tok_str,
                time_str.strip(),
            ]
            if p
        )

        phase = s.phase
        if s.paused:
            phase = "PAUSED"
        done = phase == "done"

        if done:
            spinner = "✓"
        else:
            spinner = self._SPINNER[self._spinner_idx % len(self._SPINNER)]
            self._spinner_idx += 1

        header = self.query_one("#compose-header", Static)
        stats_part = f"  {stats}" if stats else ""
        header.update(f" {spinner} {mode} | {phase}{stats_part}")

        # Append new stream entries
        stream_log = self.query_one("#stream", RichLog)
        new_entries = s.stream[self._stream_cursor :]
        for entry in new_entries:
            etype = entry.get("type", "")
            raw = entry.get("content", "")
            # Escape Rich markup characters in content
            content = raw.replace("[", "\\[")

            if etype == "step":
                stream_log.write(f"[bold cyan]{content}[/bold cyan]")
            elif etype == "reasoning":
                for line in content.split("\n"):
                    stream_log.write(f"[yellow]  {line}[/yellow]")
            elif etype == "result":
                stream_log.write(f"[green]  -> {content}[/green]")
            elif etype == "guardrail":
                stream_log.write(f"[red]  {content}[/red]")
            elif etype == "steer":
                stream_log.write(f"[bold magenta]  STEER: {content}[/bold magenta]")
            elif etype == "error":
                stream_log.write(f"[bold red]  ERROR: {content}[/bold red]")
            else:
                stream_log.write(content)

        self._stream_cursor = len(s.stream)

        # Update status bar — sparkline + milestones
        spark = ""
        energy_range = ""
        if s.sections:
            energies = [sec.get("energy", 0) for sec in s.sections]
            spark = "".join(_spark_char(e) for e in energies)
            energy_range = f" [{min(energies):.2f}-{max(energies):.2f}]"

        checks = []
        checks.append(("browse", s.browse_done))
        checks.append(("tracks", s.tracks_done))
        checks.append((f"clips({s.clips_created})", s.clips_created > 0))
        checks.append(("mix", s.mix_done))
        mile = "  ".join(f"{'✓' if done else '·'}{label}" for label, done in checks)

        status = self.query_one("#status-bar", Static)
        status.update(f"{spark}{energy_range}  {mile}")

        # Render done summary once
        if s.phase == "done" and not self._done_rendered:
            self._done_rendered = True
            elapsed_str = ""
            if s.start_time:
                mins = (_time.time() - s.start_time) / 60.0
                elapsed_str = f"{mins:.1f} min"

            tok_str = ""
            if s.llm_tokens >= 1_000:
                tok_str = f"{s.llm_tokens / 1_000:.1f}k"
            elif s.llm_tokens > 0:
                tok_str = str(s.llm_tokens)

            parts = ["", "─" * 40, "  DONE"]
            if elapsed_str:
                parts.append(f"  Time:   {elapsed_str}")
            parts.append(f"  Calls:  {s.llm_calls}")
            if tok_str:
                parts.append(f"  Tokens: {tok_str}")
            parts.append(f"  Clips:  {s.clips_created}")
            parts.append("")
            parts.append("  Press q to exit")
            parts.append("─" * 40)

            for line in parts:
                stream_log.write(f"[bold green]{line}[/bold green]")

    def on_input_submitted(self, event: Input.Submitted) -> None:
        """Handle steer input submission."""
        if event.input.id == "steer" and self._state:
            direction = event.value.strip()
            if direction:
                self._state.steer_direction = direction
                event.input.clear()

    def action_toggle_pause(self) -> None:
        if self._state:
            self._state.paused = not self._state.paused
            label = "PAUSED" if self._state.paused else "RESUMED"
            stream_log = self.query_one("#stream", RichLog)
            stream_log.write(f"[bold white on blue] {label} [/bold white on blue]")

    def action_request_abort(self) -> None:
        if self._state:
            if self._state.phase == "done":
                self.app.exit()
                return
            self._state.abort = True
            stream_log = self.query_one("#stream", RichLog)
            stream_log.write("[bold red] ABORTING... [/bold red]")


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------


class BergainApp(App):
    """bergain TUI — RLM control dashboard."""

    TITLE = "bergain"
    CSS = """
    Screen {
        background: $background;
    }
    """

    def on_mount(self) -> None:
        self.push_screen(LaunchScreen())
