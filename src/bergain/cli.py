"""
CLI entry point: bergain generate | play | dj
"""

import time
from pathlib import Path

import click


@click.group()
def cli():
    """Bergain — recipe-based techno song builder."""


@cli.command()
@click.option(
    "--sample-dir", default="sample_pack", help="Path to sample pack directory."
)
@click.option(
    "--output", default="output/", help="Output directory for arrangement artifact."
)
@click.option("--seed", type=int, default=None, help="Random seed for reproducibility.")
def generate(sample_dir, output, seed):
    """Generate a song arrangement artifact (JSON)."""
    from bergain.arrangement import save_arrangement
    from bergain.recipe import generate_arrangement

    arrangement = generate_arrangement(sample_dir, seed)

    timestamp = int(time.time())
    out_path = Path(output) / f"song_{timestamp}.json"
    save_arrangement(arrangement, out_path)

    print(f"Arrangement -> {out_path}")
    print(f"  BPM: {arrangement['bpm']}")
    total_bars = sum(s["bars"] for s in arrangement["sections"])
    section_summary = " + ".join(
        f"{s['name']}({s['bars']})" for s in arrangement["sections"]
    )
    print(f"  Structure: {section_summary} = {total_bars} bars")
    print("  Palette:")
    for role, path in arrangement["palette"].items():
        print(f"    {role}: {Path(path).name}")


@cli.command()
@click.argument("file", required=False)
def play(file):
    """Play a song artifact (.json) or WAV file. Defaults to latest artifact."""
    from bergain.arrangement import load_arrangement
    from bergain.renderer import play as play_wav
    from bergain.renderer import render

    if file is None:
        artifacts = sorted(Path("output").glob("song_*.json"))
        if not artifacts:
            raise click.ClickException(
                "No song artifacts found in output/. Run 'bergain generate' first."
            )
        file = str(artifacts[-1])
        print(f"Using latest artifact: {file}")

    path = Path(file)

    if path.suffix == ".wav":
        play_wav(str(path))
    elif path.suffix == ".json":
        arrangement = load_arrangement(path)
        wav_path = path.with_suffix(".wav")
        render(arrangement, str(wav_path))
        play_wav(str(wav_path))
    else:
        raise click.ClickException(f"Unsupported file type: {path.suffix}")


@cli.command()
@click.option(
    "--sample-dir", default="sample_pack", help="Path to sample pack directory."
)
@click.option("--bpm", type=int, default=128, help="Beats per minute.")
@click.option("--lm", required=True, help="LiteLLM model ID, e.g. 'openai/gpt-4o'.")
@click.option("--verbose", is_flag=True, help="Show RLM execution details.")
@click.option(
    "--output",
    "-o",
    default=None,
    help="Write to WAV file instead of streaming to speakers.",
)
@click.option(
    "--bars",
    type=int,
    default=None,
    help="Stop after this many bars (default: run until interrupted).",
)
@click.option(
    "--critic-lm",
    default=None,
    help="Cheap LM for critic feedback (default: same as --lm).",
)
@click.option(
    "--palette",
    default=None,
    help="JSON file with pre-selected palette (role→path map). Skips palette selection.",
)
def dj(sample_dir, bpm, lm, verbose, output, bars, critic_lm, palette):
    """Start a streaming DJ set powered by DSPy RLM."""
    from bergain.dj import run_dj

    run_dj(
        sample_dir=sample_dir,
        bpm=bpm,
        lm=lm,
        verbose=verbose,
        output=output,
        max_bars=bars,
        critic_lm=critic_lm,
        palette_file=palette,
    )
