# berg(ai)n

A generative Berlin techno experiment using [Recursive Language Models](https://arxiv.org/abs/2512.24601v1) (RLMs) with [DSPy](https://dspy.ai/).

berg(ai)n builds techno two ways: a static **recipe** that outputs a declarative arrangement (JSON), and a live **DJ mode** where an RLM streams audio in real-time, writing its own loops, selecting samples, and evolving the mix bar-by-bar via it's own ruthless self-critique in German.

## Why RLMs?

Most LLM-driven music tools put a language model inside a while loop: generate one thing, check it, generate the next. The model has no memory of what it just did, no running state, and no ability to course-correct mid-stream.

RLMs flip this. The model gets a persistent REPL — a sandbox where variables, loops, and state survive across iterations. This changes what the model *is*:

- **The REPL is the DJ's brain.** It writes `for bar in range(64)` and that loop *runs*, calling `render_and_play_bar()` 64 times. Each call blocks until the audio buffer has space, so the model is naturally paced at playback speed.
- **Tools are the DJ's hands.** `set_palette()`, `render_and_play_bar()`, `get_history()` are Python closures over shared state — loaded samples, the audio streamer, bar history. The RLM calls them as regular functions inside its own code.
- **State accumulates.** The RLM can define helper functions, store patterns in variables, and build on previous iterations. It doesn't start from scratch each turn — it *evolves*.
- **Backpressure is the clock.** `render_and_play_bar()` blocks when the audio queue is full. No `time.sleep()`, no scheduling logic. The buffer *is* the tempo.

The result: the RLM doesn't just pick samples and hand them off. It writes a DJ set as a program that runs in real-time, with the model reasoning about structure, density, and energy trajectory as it goes.

## Setup

```bash
# Install dependencies
uv sync

# Deno is required for DSPy's sandboxed code interpreter
curl -fsSL https://deno.land/install.sh | sh
# Make sure ~/.deno/bin/ is on your $PATH

# Set your LLM API key (e.g. OpenAI)
export OPENAI_API_KEY=sk-...
```

You'll also need a sample pack in `sample_pack/`. Bergain scans subdirectories and maps folder names to categories (kicks, hats, bass, etc.) via the indexer.

## Usage

### Generate a static arrangement

```bash
uv run bergain generate --seed 42
```

Outputs a JSON arrangement to `output/` describing palette, sections, and bar-level layers. The artifact is declarative — it describes *what* to play, not the audio itself.

### Play an arrangement

```bash
uv run bergain play                    # Play the latest artifact
uv run bergain play output/song.json   # Play a specific arrangement
uv run bergain play output/song.wav    # Play a rendered WAV directly
```

Renders the arrangement to WAV via pydub, then plays with `afplay`.

### DJ mode (live streaming)

```bash
uv run bergain dj --lm "openai/gpt-5-mini"            # Stream to speakers
uv run bergain dj --lm "openai/gpt-5-mini" --verbose   # With RLM debug output
uv run bergain dj --lm "openai/gpt-5-mini" -o set.wav  # Record to file
```

The RLM indexes your sample pack, picks a palette, and starts streaming audio. It renders 48-64 bars per iteration, evolving the arrangement over time. A built-in critique system analyzes density, energy trajectory, and repetition every 16 bars, pushing corrective directives back to the RLM inline.

Press `Ctrl+C` to stop.

## Architecture

```
sample_pack/ --> indexer --> sample index
                              |
               +--------------+--------------+
               |                             |
          [generate]                      [dj]
               |                             |
        recipe.py builds              RLM writes its own
        arrangement dict               DJ loop in REPL
               |                             |
        renderer.render()           render_and_play_bar()
               |                             |
           WAV file                   AudioStreamer
                                          |
                                      speakers
```

The key design decision: **the artifact is a declarative arrangement, not audio.** This keeps the generate/play pipeline inspectable (it's just JSON) and makes the streaming DJ possible — the same `render_bar()` function powers both paths.

## Acknowledgements

- Sample pack provided by [technosupps.gumroad.com](https://technosupps.gumroad.com).
