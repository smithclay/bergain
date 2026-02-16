"""Quick browser search utility."""

import sys
from main import LiveAPI

api = LiveAPI()

if len(sys.argv) < 2:
    print("Usage: uv run python browse.py <command> [args]")
    print("  list <category>        — list top-level items in a category")
    print("  browse <cat> <path>    — browse into a subcategory")
    print("  search <query>         — full-text search")
    print(
        "\nCategories: sounds, instruments, drums, audio_effects, midi_effects, packs, samples, clips"
    )
    api.stop()
    sys.exit(0)

cmd = sys.argv[1]

if cmd == "list":
    cat = sys.argv[2] if len(sys.argv) > 2 else "sounds"
    result = api.browser.query("browse", cat)
    for name in result:
        print(f"  {name}")

elif cmd == "browse":
    cat = sys.argv[2]
    path = sys.argv[3]
    result = api.browser.query("browse_path", cat, path)
    for name in result:
        print(f"  {name}")

elif cmd == "search":
    query = " ".join(sys.argv[2:])
    result = api.browser.query("search", query)
    # Results are alternating (category, name) pairs
    pairs = list(result)
    for i in range(0, len(pairs) - 1, 2):
        print(f"  [{pairs[i]:20s}] {pairs[i + 1]}")

api.stop()
