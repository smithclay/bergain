"""TUI tests — Textual app smoke tests."""

import pytest

from bergain.tui import BergainApp, ComposeScreen
from textual.widgets import Input, RichLog


@pytest.fixture
def app():
    return BergainApp()


@pytest.mark.anyio
async def test_app_launches(app):
    """App should mount and show the launch screen."""
    async with app.run_test() as pilot:
        await pilot.pause()
        assert app.screen is not None
        assert app.screen.query_one("#brief") is not None
        assert app.screen.query_one("#start") is not None


@pytest.mark.anyio
async def test_osc_status_shows(app):
    """OSC status widget should exist on launch screen."""
    async with app.run_test() as pilot:
        await pilot.pause()
        osc = app.screen.query_one("#osc-status")
        assert osc is not None


@pytest.mark.anyio
async def test_compose_screen_has_widgets():
    """ComposeScreen should have stream, steer, header, status bar."""
    app = BergainApp()
    config = {
        "brief": "test brief",
        "live": False,
        "model": "test-model",
        "sub_model": None,
        "duration": 60,
        "max_iterations": 30,
    }
    async with app.run_test() as pilot:
        screen = ComposeScreen(config)
        await app.push_screen(screen)
        await pilot.pause()

        assert app.screen.query_one("#stream", RichLog) is not None
        assert app.screen.query_one("#steer", Input) is not None
        assert app.screen.query_one("#compose-header") is not None
        assert app.screen.query_one("#status-bar") is not None
