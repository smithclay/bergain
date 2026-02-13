---
name: use-uv-not-python
enabled: true
event: bash
pattern: (?:^|[;&|]\s*)python3?\s
action: block
---

**Use `uv run` instead of `python` / `python3`.**

This project uses `uv` as its package manager. Replace:
- `python script.py` → `uv run python script.py`
- `python3 -m module` → `uv run python -m module`
- `python3 -c "..."` → `uv run python -c "..."`
