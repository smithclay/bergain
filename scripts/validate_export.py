"""Validate audio export assumptions against a live Ableton instance.

Run each test independently:
    uv run python scripts/validate_export.py routing
    uv run python scripts/validate_export.py arm_record
    uv run python scripts/validate_export.py record_short
    uv run python scripts/validate_export.py all

Requires: AbletonOSC running, Ableton open with at least 1 track.
"""

import os
import sys
import time

from bergain.osc import LiveAPI
from bergain.session import Session


def test_routing():
    """Test: create audio track, set Resampling input + Sends Only output."""
    print("\n=== Test 1: Track Routing ===")
    api = LiveAPI()

    # Create an audio track at position 0
    print("  Creating audio track...")
    api.song.call("create_audio_track", 0)
    time.sleep(0.5)

    track_idx = 0
    api.track(track_idx).set("name", "_capture_test")
    time.sleep(0.2)

    # List available input routing types
    print("  Available input routing types:")
    result = api.track(track_idx).query("get/available_input_routing_types")
    input_types = list(result) if result else []
    for t in input_types:
        print(f"    - {t}")

    # Set input to Resampling
    if "Resampling" in input_types:
        api.track(track_idx).set("input_routing_type", "Resampling")
        time.sleep(0.3)
        current = api.track(track_idx).query("get/input_routing_type")
        print(f"  Input routing set to: {current}")
    else:
        print("  FAIL: 'Resampling' not in available input types!")
        api.song.call("delete_track", track_idx)
        api.stop()
        return

    # List available output routing types
    print("  Available output routing types:")
    result = api.track(track_idx).query("get/available_output_routing_types")
    output_types = list(result) if result else []
    for t in output_types:
        print(f"    - {t}")

    # Set output to Sends Only
    sends_name = None
    for name in output_types:
        if "sends" in name.lower():
            sends_name = name
            break

    if sends_name:
        api.track(track_idx).set("output_routing_type", sends_name)
        time.sleep(0.3)
        current = api.track(track_idx).query("get/output_routing_type")
        print(f"  Output routing set to: {current}")
    else:
        print("  FAIL: No 'Sends Only' found in output types!")

    # Set monitoring to In (0)
    api.track(track_idx).set("current_monitoring_state", 0)
    time.sleep(0.2)
    mon = api.track(track_idx).get("current_monitoring_state")
    print(f"  Monitoring state: {mon} (0=In, 1=Auto, 2=Off)")

    # Cleanup
    print("  Deleting test track...")
    api.song.call("delete_track", track_idx)
    time.sleep(0.3)

    api.stop()
    print("  PASSED\n")


def test_arm_record():
    """Test: create audio track with Resampling, arm it, check session_record props."""
    print("\n=== Test 2: Arm + Session Record ===")
    api = LiveAPI()

    # Create audio track at end
    num_tracks = api.song.get("num_tracks")
    print(f"  Creating audio track at index {num_tracks}...")
    api.song.call("create_audio_track", num_tracks)
    time.sleep(0.5)
    capture_idx = num_tracks
    api.track(capture_idx).set("name", "_capture_test")
    time.sleep(0.2)

    # Route: Resampling in, Sends Only out
    api.track(capture_idx).set("input_routing_type", "Resampling")
    time.sleep(0.3)
    output_types = list(
        api.track(capture_idx).query("get/available_output_routing_types") or []
    )
    sends_name = next((n for n in output_types if "sends" in n.lower()), None)
    if sends_name:
        api.track(capture_idx).set("output_routing_type", sends_name)
        time.sleep(0.3)
    api.track(capture_idx).set("current_monitoring_state", 0)
    time.sleep(0.2)

    # Arm the track
    print("  Arming track...")
    api.track(capture_idx).set("arm", 1)
    time.sleep(0.3)
    armed = api.track(capture_idx).get("arm")
    print(f"  Armed: {armed}")

    # Check session record properties
    print("  Song session_record properties:")
    for prop in ["session_record", "record_mode", "session_record_status"]:
        try:
            val = api.song.get(prop)
            print(f"    {prop} = {val}")
        except Exception as e:
            print(f"    {prop} = ERROR: {e}")

    # Check available clip slots on capture track
    print("  Checking clip slots on capture track...")
    try:
        has_clip = api.clip_slot(capture_idx, 0).query("get/has_clip")
        print(f"    Slot 0 has_clip: {has_clip}")
    except Exception as e:
        print(f"    Slot 0 query error: {e}")

    # Cleanup
    print("  Disarming and deleting test track...")
    api.track(capture_idx).set("arm", 0)
    time.sleep(0.2)
    api.song.call("delete_track", capture_idx)
    time.sleep(0.3)

    api.stop()
    print("  PASSED\n")


def test_record_short():
    """Test: full e2e using Session.start_recording() / stop_recording()."""
    print("\n=== Test 3: Session Recording API (5s) ===")
    session = Session()

    # Start recording via the new Session API
    session.start_recording()

    print("  Recording for 5 seconds...")
    time.sleep(5.0)

    # Stop and export
    wav = session.stop_recording()

    session.close()
    if wav and os.path.isfile(wav):
        print(f"  PASSED — exported: {wav}\n")
    else:
        print("  FAILED — no export produced\n")


def _analyze_wav(wav):
    """Analyze a WAV file and return (peak_pct, rms) or None on error."""
    import struct
    import wave

    with wave.open(wav, "rb") as wf:
        n_frames = wf.getnframes()
        n_channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        rate = wf.getframerate()
        raw = wf.readframes(n_frames)
        print(
            f"  Format: {n_channels}ch, {rate}Hz, {sampwidth * 8}bit, {n_frames} frames"
        )

        if sampwidth == 2:
            samples = struct.unpack(f"<{n_frames * n_channels}h", raw)
        elif sampwidth == 3:
            samples = []
            for i in range(0, len(raw), 3):
                val = int.from_bytes(raw[i : i + 3], byteorder="little", signed=True)
                samples.append(val)
        else:
            print(f"  Unexpected sample width: {sampwidth}")
            return None

        if not samples:
            return None

        peak = max(abs(s) for s in samples)
        rms = (sum(s * s for s in samples) / len(samples)) ** 0.5
        max_possible = (2 ** (sampwidth * 8 - 1)) - 1
        peak_pct = 100 * peak / max_possible
        print(f"  Peak: {peak}/{max_possible} ({peak_pct:.1f}%)")
        print(f"  RMS: {rms:.1f}")
        return peak_pct, rms


def test_setup_restore():
    """Test: recording survives setup() — simulates the real compose pipeline flow.

    Flow: start_recording → setup(tracks) → create clips → fire → wait → stop_recording.
    This is the exact sequence that happens in the --export compose pipeline.
    """
    from bergain.session import Track

    print("\n=== Test 4: Recording + Setup Restore ===")
    session = Session()

    # Step 1: Start recording
    print("  Step 1: Starting recording...")
    session.start_recording()
    time.sleep(1.0)

    # Step 2: Call setup() — destroys everything, restores capture track
    print("\n  Step 2: Calling setup() (destroys + restores capture track)...")
    tracks = [
        Track(name="TestBass", instrument="Operator"),
    ]
    count = session.setup(tracks)
    print(f"  Created {count} tracks")

    # Step 3: Verify the restored capture track
    print("\n  Step 3: Verifying restored capture track...")
    capture_idx = session._capture_track_idx
    if capture_idx is None:
        print("  FAILED — capture track not restored")
        session.close()
        return

    print(f"  Capture track: {capture_idx}")
    print(f"  Armed: {session.api.track(capture_idx).get('arm')}")
    input_routing = session.api.track(capture_idx).query("get/input_routing_type")
    print(f"  Input routing: {input_routing}")
    try:
        status = session.api.song.get("session_record_status")
        print(f"  Session record status: {status} (want 2)")
    except TimeoutError:
        print("  Session record status: (timeout)")

    # Step 4: Create a clip with notes and play it to generate audio
    print("\n  Step 4: Creating clip with notes and playing...")
    # Sustained C3 notes (pitch 48) at max velocity — loud enough to measure
    notes = [(48, beat * 2, 1.5, 127) for beat in range(8)]
    session.clip("TestBass", 0, 16.0, notes, name="Test Bass")
    session.api.track(0).set("volume", 1.0)  # max volume
    time.sleep(0.3)
    session.fire(0)
    time.sleep(0.5)

    # Let it play for 5 seconds
    print("  Playing for 5 seconds...")
    time.sleep(5.0)

    # Step 5: Stop recording and export
    print("\n  Step 5: Stopping recording...")
    wav = session.stop_recording()
    session.close()

    if not wav or not os.path.isfile(wav):
        print("  FAILED — no WAV produced")
        return

    size_kb = os.path.getsize(wav) / 1024
    print(f"  WAV: {wav} ({size_kb:.1f} KB)")

    result = _analyze_wav(wav)
    if result is None:
        print("  FAILED — could not analyze WAV")
    elif result[0] < 0.01:
        print("  FAILED — audio is ZERO")
    else:
        print(
            f"  PASSED — audio has content (peak={result[0]:.2f}%, RMS={result[1]:.1f})"
        )


def test_record_with_audio():
    """Test: baseline recording with Operator playing.

    Starts playback FIRST, then triggers recording — avoids timing issues
    with trigger_session_record starting transport prematurely.
    """
    print("\n=== Test 5: Baseline Recording with Audio ===")
    api = LiveAPI()

    # Ensure at least 1 MIDI track with Operator
    num_tracks = api.song.get("num_tracks")
    print(f"  Existing tracks: {num_tracks}")
    if num_tracks == 0:
        api.song.call("create_midi_track", 0)
        time.sleep(0.5)
        num_tracks = 1

    # Load Operator on track 0
    api.view.set("selected_track", 0)
    time.sleep(0.1)
    result = api.browser.query("load_instrument", "Operator")
    loaded = result[0] if result else None
    print(f"  Loaded: {loaded}")
    api.track(0).set("volume", 1.0)
    time.sleep(0.5)

    # Create clip with loud sustained notes
    cs = api.clip_slot(0, 0)
    if cs.get("has_clip"):
        cs.call("delete_clip")
        time.sleep(0.1)
    cs.call("create_clip", 16.0)
    time.sleep(0.2)

    # Add notes: sustained C3 notes at max velocity
    from bergain.session import _notes_to_wire

    notes = [(48, beat * 2, 1.5, 127) for beat in range(8)]
    wire = _notes_to_wire(notes)
    api.clip(0, 0).send_batched("add/notes", wire)
    api.clip(0, 0).set("name", "Test Bass")
    time.sleep(0.2)

    # Create capture track at end
    capture_idx = api.song.get("num_tracks")
    api.song.call("create_audio_track", capture_idx)
    time.sleep(0.5)
    api.track(capture_idx).set("name", "_capture")
    api.track(capture_idx).set("input_routing_type", "Resampling")
    time.sleep(0.3)
    output_types = list(
        api.track(capture_idx).query("get/available_output_routing_types") or []
    )
    sends = next((n for n in output_types if "sends" in n.lower()), None)
    if sends:
        api.track(capture_idx).set("output_routing_type", sends)
        time.sleep(0.3)
    api.track(capture_idx).set("current_monitoring_state", 0)
    time.sleep(0.2)
    api.track(capture_idx).set("arm", 1)
    time.sleep(0.3)

    # Step 1: Start playback and fire clip
    print("  Starting playback...")
    api.scene(0).call("fire")
    time.sleep(1.0)

    # Verify playback
    playing = api.song.get("is_playing")
    print(f"  Playing: {playing}")
    slot = api.track(0).get("playing_slot_index")
    print(f"  Track 0 playing slot: {slot}")

    # Step 2: NOW trigger session recording (transport already running)
    print("  Triggering session recording...")
    api.song.call("trigger_session_record")
    time.sleep(1.0)

    try:
        status = api.song.get("session_record_status")
        print(f"  Session record status: {status} (want 2)")
    except TimeoutError:
        print("  Status check timed out")

    # Step 3: Wait for recording
    print("  Recording for 5 seconds...")
    time.sleep(5.0)

    # Step 4: Stop everything
    api.song.call("stop_playing")
    time.sleep(1.0)

    # Step 5: Find recorded clip on capture track
    print(f"  Scanning capture track {capture_idx} for clips...")
    clip_path = None
    num_scenes = api.song.get("num_scenes")
    for slot_idx in range(num_scenes):
        try:
            has = api.clip_slot(capture_idx, slot_idx).query("get/has_clip")
            if has and has[0]:
                fp = api.clip(capture_idx, slot_idx).query("get/file_path")
                path = fp[0] if fp else None
                print(f"    Slot {slot_idx}: has clip, file_path={path}")
                if path and os.path.isfile(path):
                    clip_path = path
                    break
                elif path:
                    print("    (file not found on disk)")
        except Exception as e:
            print(f"    Slot {slot_idx}: error: {e}")

    # Cleanup
    api.track(capture_idx).set("arm", 0)
    time.sleep(0.1)
    api.song.call("delete_track", capture_idx)
    time.sleep(0.3)
    api.stop()

    if not clip_path:
        print("  FAILED — no recorded clip found")
        return

    size_kb = os.path.getsize(clip_path) / 1024
    print(f"  WAV: {clip_path} ({size_kb:.1f} KB)")
    result = _analyze_wav(clip_path)
    if result is None:
        print("  FAILED — could not analyze")
    elif result[0] < 0.01:
        print("  FAILED — audio is ZERO")
    else:
        print(f"  PASSED — peak={result[0]:.2f}%, RMS={result[1]:.1f}")


def test_multi_scene():
    """Test: recording survives multiple scene fires — simulates real compose flow.

    Flow: start_recording → setup(2 tracks) → create clips in 3 slots →
          fire scene 0 → wait → fire scene 1 → wait → fire scene 2 → wait → stop.
    Verifies the exported WAV has audio content across the full duration (~15s).
    """
    from bergain.session import Track

    print("\n=== Test 6: Multi-Scene Recording ===")
    session = Session()

    # Step 1: Start recording
    print("  Step 1: Starting recording...")
    session.start_recording()
    time.sleep(1.0)

    # Step 2: Setup tracks (destroys + restores capture)
    print("  Step 2: Setting up tracks...")
    tracks = [
        Track(name="Bass", instrument="Operator", volume=1.0),
        Track(name="Pad", instrument="Operator", volume=1.0),
    ]
    session.setup(tracks)
    print(f"  Capture track idx: {session._capture_track_idx}")

    # Step 3: Create clips in 3 slots with different notes
    print("  Step 3: Creating clips in 3 slots...")
    for slot in range(3):
        base_pitch = 36 + slot * 12  # C2, C3, C4
        notes = [(base_pitch + i, i * 2, 1.5, 127) for i in range(4)]
        session.clip("Bass", slot, 8.0, notes, name=f"Bass S{slot}")
        pad_notes = [(base_pitch + 7, 0, 7.5, 100)]  # sustained fifth
        session.clip("Pad", slot, 8.0, pad_notes, name=f"Pad S{slot}")

    # Step 4: Fire scenes sequentially with waits
    for slot in range(3):
        print(f"  Step 4.{slot}: Firing scene {slot}, waiting 5s...")
        session.fire(slot)
        time.sleep(5.0)

    # Step 5: Stop and export
    print("  Step 5: Stopping recording...")
    wav = session.stop_recording()
    session.close()

    if not wav or not os.path.isfile(wav):
        print("  FAILED — no WAV produced")
        return

    size_kb = os.path.getsize(wav) / 1024
    print(f"  WAV: {wav} ({size_kb:.1f} KB)")

    result = _analyze_wav(wav)
    if result is None:
        print("  FAILED — could not analyze WAV")
    elif result[0] < 1.0:
        print(f"  FAILED — audio too quiet (peak={result[0]:.2f}%)")
    else:
        print(f"  PASSED — peak={result[0]:.2f}%, RMS={result[1]:.1f}")

    # Check for audio across the full duration (not just at the end)
    import wave as wave_mod
    import struct as struct_mod

    with wave_mod.open(wav, "rb") as wf:
        n_frames = wf.getnframes()
        n_ch = wf.getnchannels()
        sw = wf.getsampwidth()
        rate = wf.getframerate()
        raw = wf.readframes(n_frames)
        max_val = (2 ** (sw * 8 - 1)) - 1
        if sw == 3:
            samples = [
                int.from_bytes(raw[i : i + 3], "little", signed=True)
                for i in range(0, len(raw), 3)
            ]
        else:
            samples = list(struct_mod.unpack(f"<{n_frames * n_ch}h", raw))

        chunk_sec = 5
        chunk_size = chunk_sec * rate * n_ch
        has_audio_chunks = 0
        total_chunks = 0
        for i in range(0, len(samples), chunk_size):
            chunk = samples[i : i + chunk_size]
            peak = max(abs(s) for s in chunk) if chunk else 0
            pct = peak / max_val * 100
            t0 = (i // n_ch) / rate
            label = "AUDIO" if pct > 1.0 else "silent"
            print(f"    {t0:.0f}-{t0 + chunk_sec:.0f}s: peak={pct:.1f}% [{label}]")
            total_chunks += 1
            if pct > 1.0:
                has_audio_chunks += 1

        coverage = has_audio_chunks / total_chunks * 100 if total_chunks else 0
        print(
            f"  Audio coverage: {has_audio_chunks}/{total_chunks} chunks ({coverage:.0f}%)"
        )
        if coverage < 50:
            print("  FAILED — audio present in less than 50% of duration")
        else:
            print("  PASSED — good audio coverage")


TESTS = {
    "routing": test_routing,
    "arm_record": test_arm_record,
    "record_short": test_record_short,
    "setup_restore": test_setup_restore,
    "record_with_audio": test_record_with_audio,
    "multi_scene": test_multi_scene,
}


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in {*TESTS, "all"}:
        print("Usage: uv run python scripts/validate_export.py <test|all>")
        print(f"  Tests: {', '.join(TESTS)} | all")
        sys.exit(1)

    if sys.argv[1] == "all":
        for name, fn in TESTS.items():
            try:
                fn()
            except Exception as e:
                import traceback

                print(f"  ERROR in {name}: {e}")
                traceback.print_exc()
    else:
        TESTS[sys.argv[1]]()


if __name__ == "__main__":
    main()
