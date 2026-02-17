"""Populated AbletonOSC API spec with all ~500 endpoints across 10 domains.

Built by extracting every add_handler call from the AbletonOSC handler source files.
"""

from .spec import ParamSpec, EndpointSpec, DomainSpec, AbletonOSCSpec


# =============================================================================
# Helpers
# =============================================================================


def _p(
    name: str, type: str = "any", description: str = "", optional: bool = False
) -> ParamSpec:
    return ParamSpec(name=name, type=type, description=description, optional=optional)


# Common index params
_track = _p("track_id", "int", "Track index")
_clip = _p("clip_id", "int", "Clip slot index")
_device = _p("device_id", "int", "Device index")
_scene = _p("scene_id", "int", "Scene index")


def _prop_eps(
    domain: str,
    prop: str,
    access: str,
    ptype: str = "any",
    idx: list[ParamSpec] | None = None,
    wildcard: bool = False,
    desc: str = "",
) -> list[EndpointSpec]:
    """Generate get/set/listen endpoints for a standard property."""
    ix = list(idx) if idx else []
    eps = [
        EndpointSpec(
            address=f"/live/{domain}/get/{prop}",
            domain=domain,
            kind="get",
            params=ix,
            returns=ix + [_p(prop, ptype)],
            description=desc or f"Get {prop}",
            wildcard=wildcard,
        ),
        EndpointSpec(
            address=f"/live/{domain}/start_listen/{prop}",
            domain=domain,
            kind="listen_start",
            params=ix,
            description=f"Listen to {prop} changes",
            wildcard=wildcard,
        ),
        EndpointSpec(
            address=f"/live/{domain}/stop_listen/{prop}",
            domain=domain,
            kind="listen_stop",
            params=ix,
            description=f"Stop listening to {prop}",
            wildcard=wildcard,
        ),
    ]
    if access == "rw":
        eps.append(
            EndpointSpec(
                address=f"/live/{domain}/set/{prop}",
                domain=domain,
                kind="set",
                params=ix + [_p(prop, ptype)],
                description=f"Set {prop}",
                wildcard=wildcard,
            )
        )
    return eps


def _method_ep(
    domain: str,
    method: str,
    params: list[ParamSpec] | None = None,
    returns: list[ParamSpec] | None = None,
    desc: str = "",
    wildcard: bool = False,
) -> EndpointSpec:
    """Generate endpoint for a standard method call."""
    return EndpointSpec(
        address=f"/live/{domain}/{method}",
        domain=domain,
        kind="method",
        params=params or [],
        returns=returns or [],
        description=desc or method.replace("_", " ").capitalize(),
        wildcard=wildcard,
    )


def _custom(
    address: str,
    domain: str,
    kind: str = "custom",
    params: list[ParamSpec] | None = None,
    returns: list[ParamSpec] | None = None,
    desc: str = "",
    wildcard: bool = False,
) -> EndpointSpec:
    return EndpointSpec(
        address=address,
        domain=domain,
        kind=kind,
        params=params or [],
        returns=returns or [],
        description=desc,
        wildcard=wildcard,
    )


def _bulk_props(
    domain: str,
    props_ro: list[tuple[str, str]],
    props_rw: list[tuple[str, str]],
    idx: list[ParamSpec] | None = None,
    wildcard: bool = False,
) -> list[EndpointSpec]:
    """Generate all property endpoints for lists of (name, type) tuples."""
    eps: list[EndpointSpec] = []
    for prop, ptype in props_ro:
        eps.extend(_prop_eps(domain, prop, "r", ptype, idx=idx, wildcard=wildcard))
    for prop, ptype in props_rw:
        eps.extend(_prop_eps(domain, prop, "rw", ptype, idx=idx, wildcard=wildcard))
    return eps


def _bulk_methods(
    domain: str,
    methods: list[str],
    idx: list[ParamSpec] | None = None,
    wildcard: bool = False,
) -> list[EndpointSpec]:
    """Generate method endpoints for a list of method names."""
    return [
        _method_ep(domain, m, params=list(idx) if idx else None, wildcard=wildcard)
        for m in methods
    ]


# =============================================================================
# Song domain  (source: abletonosc/song.py)
# =============================================================================

_song_methods = [
    "capture_and_insert_scene",
    "capture_midi",
    "continue_playing",
    "create_audio_track",
    "create_midi_track",
    "create_return_track",
    "create_scene",
    "delete_return_track",
    "delete_scene",
    "delete_track",
    "duplicate_scene",
    "duplicate_track",
    "force_link_beat_time",
    "jump_by",
    "jump_to_prev_cue",
    "jump_to_next_cue",
    "redo",
    "re_enable_automation",
    "set_or_delete_cue",
    "start_playing",
    "stop_all_clips",
    "stop_playing",
    "tap_tempo",
    "trigger_session_record",
    "undo",
]

_song_props_rw: list[tuple[str, str]] = [
    ("arrangement_overdub", "bool"),
    ("back_to_arranger", "bool"),
    ("clip_trigger_quantization", "int"),
    ("current_song_time", "float"),
    ("groove_amount", "float"),
    ("is_ableton_link_enabled", "bool"),
    ("loop", "bool"),
    ("loop_length", "float"),
    ("loop_start", "float"),
    ("metronome", "bool"),
    ("midi_recording_quantization", "int"),
    ("nudge_down", "bool"),
    ("nudge_up", "bool"),
    ("punch_in", "bool"),
    ("punch_out", "bool"),
    ("record_mode", "bool"),
    ("root_note", "int"),
    ("scale_name", "str"),
    ("session_record", "bool"),
    ("signature_denominator", "int"),
    ("signature_numerator", "int"),
    ("tempo", "float"),
]

_song_props_ro: list[tuple[str, str]] = [
    ("can_redo", "bool"),
    ("can_undo", "bool"),
    ("is_playing", "bool"),
    ("song_length", "float"),
    ("session_record_status", "int"),
]

_song_custom: list[EndpointSpec] = [
    _custom(
        "/live/song/get/num_tracks",
        "song",
        kind="get",
        returns=[_p("num_tracks", "int")],
        desc="Get number of tracks",
    ),
    _custom(
        "/live/song/get/track_names",
        "song",
        kind="get",
        params=[_p("start", "int", optional=True), _p("end", "int", optional=True)],
        returns=[_p("names", "str", "Track names")],
        desc="Get track names, optionally for a range",
    ),
    _custom(
        "/live/song/get/track_data",
        "song",
        kind="get",
        params=[
            _p("start", "int"),
            _p("end", "int"),
            _p("properties", "str", "e.g. track.name clip.name clip.length"),
        ],
        returns=[_p("values", "any", "Flattened property values")],
        desc="Retrieve properties of a block of tracks and their clips",
    ),
    _custom(
        "/live/song/export/structure",
        "song",
        kind="custom",
        returns=[_p("success", "int")],
        desc="Export song structure as JSON to temp directory",
    ),
    _custom(
        "/live/song/get/num_scenes",
        "song",
        kind="get",
        returns=[_p("num_scenes", "int")],
        desc="Get number of scenes",
    ),
    _custom(
        "/live/song/get/scenes/name",
        "song",
        kind="get",
        params=[_p("start", "int", optional=True), _p("end", "int", optional=True)],
        returns=[_p("names", "str", "Scene names")],
        desc="Get scene names, optionally for a range",
    ),
    _custom(
        "/live/song/get/cue_points",
        "song",
        kind="get",
        returns=[_p("cue_data", "any", "Alternating name, time pairs")],
        desc="Get all cue points as name/time pairs",
    ),
    _custom(
        "/live/song/cue_point/jump",
        "song",
        kind="method",
        params=[_p("cue_point", "any", "Index (int) or name (str)")],
        desc="Jump to a cue point by index or name",
    ),
    _custom(
        "/live/song/cue_point/add_or_delete",
        "song",
        kind="method",
        desc="Add or delete a cue point at current position",
    ),
    _custom(
        "/live/song/cue_point/set/name",
        "song",
        kind="set",
        params=[_p("cue_index", "int"), _p("name", "str")],
        desc="Set the name of a cue point",
    ),
    _custom(
        "/live/song/start_listen/beat",
        "song",
        kind="listen_start",
        desc="Start beat listener (sends /live/song/get/beat on each beat)",
    ),
    _custom(
        "/live/song/stop_listen/beat",
        "song",
        kind="listen_stop",
        desc="Stop beat listener",
    ),
    _custom(
        "/live/song/get/beat",
        "song",
        kind="get",
        returns=[_p("beat", "int", "Current beat position")],
        desc="Beat notification (outbound, sent by beat listener on each beat)",
    ),
]

song_domain = DomainSpec(
    name="song",
    description="Global song properties, transport, cue points, and structure queries",
    base_address="/live/song",
    index_params=[],
    endpoints=(
        _bulk_methods("song", _song_methods)
        + _bulk_props("song", _song_props_ro, _song_props_rw)
        + _song_custom
    ),
)


# =============================================================================
# Track domain  (source: abletonosc/track.py)
# =============================================================================

_track_idx = [_track]

_track_methods = ["delete_device", "stop_all_clips"]

_track_props_ro: list[tuple[str, str]] = [
    ("can_be_armed", "bool"),
    ("fired_slot_index", "int"),
    ("has_audio_input", "bool"),
    ("has_audio_output", "bool"),
    ("has_midi_input", "bool"),
    ("has_midi_output", "bool"),
    ("is_foldable", "bool"),
    ("is_grouped", "bool"),
    ("is_visible", "bool"),
    ("output_meter_level", "float"),
    ("output_meter_left", "float"),
    ("output_meter_right", "float"),
    ("playing_slot_index", "int"),
]

_track_props_rw: list[tuple[str, str]] = [
    ("arm", "bool"),
    ("color", "int"),
    ("color_index", "int"),
    ("current_monitoring_state", "int"),
    ("fold_state", "int"),
    ("mute", "bool"),
    ("solo", "bool"),
    ("name", "str"),
]

# Mixer properties have the same get/set/listen pattern but go through mixer_device
_track_mixer_rw: list[tuple[str, str]] = [
    ("volume", "float"),
    ("panning", "float"),
]

_track_custom: list[EndpointSpec] = [
    # Send get/set
    _custom(
        "/live/track/get/send",
        "track",
        kind="get",
        params=[_track, _p("send_id", "int")],
        returns=[_track, _p("send_id", "int"), _p("value", "float")],
        desc="Get send level",
        wildcard=True,
    ),
    _custom(
        "/live/track/set/send",
        "track",
        kind="set",
        params=[_track, _p("send_id", "int"), _p("value", "float")],
        desc="Set send level",
        wildcard=True,
    ),
    # Delete clip
    _custom(
        "/live/track/delete_clip",
        "track",
        kind="method",
        params=[_track, _p("clip_index", "int")],
        desc="Delete clip at slot index",
        wildcard=True,
    ),
    # Batch clip queries
    _custom(
        "/live/track/get/clips/name",
        "track",
        kind="get",
        params=[_track],
        returns=[_track, _p("names", "str")],
        desc="Get names of all clips in track",
        wildcard=True,
    ),
    _custom(
        "/live/track/get/clips/length",
        "track",
        kind="get",
        params=[_track],
        returns=[_track, _p("lengths", "float")],
        desc="Get lengths of all clips in track",
        wildcard=True,
    ),
    _custom(
        "/live/track/get/clips/color",
        "track",
        kind="get",
        params=[_track],
        returns=[_track, _p("colors", "int")],
        desc="Get colors of all clips in track",
        wildcard=True,
    ),
    # Arrangement clips
    _custom(
        "/live/track/get/arrangement_clips/name",
        "track",
        kind="get",
        params=[_track],
        returns=[_track, _p("names", "str")],
        desc="Get arrangement clip names",
        wildcard=True,
    ),
    _custom(
        "/live/track/get/arrangement_clips/length",
        "track",
        kind="get",
        params=[_track],
        returns=[_track, _p("lengths", "float")],
        desc="Get arrangement clip lengths",
        wildcard=True,
    ),
    _custom(
        "/live/track/get/arrangement_clips/start_time",
        "track",
        kind="get",
        params=[_track],
        returns=[_track, _p("start_times", "float")],
        desc="Get arrangement clip start times",
        wildcard=True,
    ),
    # Device queries
    _custom(
        "/live/track/get/num_devices",
        "track",
        kind="get",
        params=[_track],
        returns=[_track, _p("num_devices", "int")],
        desc="Get number of devices on track",
        wildcard=True,
    ),
    _custom(
        "/live/track/get/devices/name",
        "track",
        kind="get",
        params=[_track],
        returns=[_track, _p("names", "str")],
        desc="Get device names",
        wildcard=True,
    ),
    _custom(
        "/live/track/get/devices/type",
        "track",
        kind="get",
        params=[_track],
        returns=[_track, _p("types", "int")],
        desc="Get device types (0=audio_effect, 1=instrument, 2=midi_effect)",
        wildcard=True,
    ),
    _custom(
        "/live/track/get/devices/class_name",
        "track",
        kind="get",
        params=[_track],
        returns=[_track, _p("class_names", "str")],
        desc="Get device class names (e.g. Operator, Reverb)",
        wildcard=True,
    ),
    _custom(
        "/live/track/get/devices/can_have_chains",
        "track",
        kind="get",
        params=[_track],
        returns=[_track, _p("can_have_chains", "bool")],
        desc="Get whether each device can have chains",
        wildcard=True,
    ),
    # Output routing
    _custom(
        "/live/track/get/available_output_routing_types",
        "track",
        kind="get",
        params=[_track],
        returns=[_track, _p("types", "str")],
        desc="Get available output routing type names",
        wildcard=True,
    ),
    _custom(
        "/live/track/get/available_output_routing_channels",
        "track",
        kind="get",
        params=[_track],
        returns=[_track, _p("channels", "str")],
        desc="Get available output routing channel names",
        wildcard=True,
    ),
    _custom(
        "/live/track/get/output_routing_type",
        "track",
        kind="get",
        params=[_track],
        returns=[_track, _p("type", "str")],
        desc="Get current output routing type",
        wildcard=True,
    ),
    _custom(
        "/live/track/set/output_routing_type",
        "track",
        kind="set",
        params=[_track, _p("type_name", "str")],
        desc="Set output routing type by name",
        wildcard=True,
    ),
    _custom(
        "/live/track/get/output_routing_channel",
        "track",
        kind="get",
        params=[_track],
        returns=[_track, _p("channel", "str")],
        desc="Get current output routing channel",
        wildcard=True,
    ),
    _custom(
        "/live/track/set/output_routing_channel",
        "track",
        kind="set",
        params=[_track, _p("channel_name", "str")],
        desc="Set output routing channel by name",
        wildcard=True,
    ),
    # Input routing
    _custom(
        "/live/track/get/available_input_routing_types",
        "track",
        kind="get",
        params=[_track],
        returns=[_track, _p("types", "str")],
        desc="Get available input routing type names",
        wildcard=True,
    ),
    _custom(
        "/live/track/get/available_input_routing_channels",
        "track",
        kind="get",
        params=[_track],
        returns=[_track, _p("channels", "str")],
        desc="Get available input routing channel names",
        wildcard=True,
    ),
    _custom(
        "/live/track/get/input_routing_type",
        "track",
        kind="get",
        params=[_track],
        returns=[_track, _p("type", "str")],
        desc="Get current input routing type",
        wildcard=True,
    ),
    _custom(
        "/live/track/set/input_routing_type",
        "track",
        kind="set",
        params=[_track, _p("type_name", "str")],
        desc="Set input routing type by name",
        wildcard=True,
    ),
    _custom(
        "/live/track/get/input_routing_channel",
        "track",
        kind="get",
        params=[_track],
        returns=[_track, _p("channel", "str")],
        desc="Get current input routing channel",
        wildcard=True,
    ),
    _custom(
        "/live/track/set/input_routing_channel",
        "track",
        kind="set",
        params=[_track, _p("channel_name", "str")],
        desc="Set input routing channel by name",
        wildcard=True,
    ),
]

track_domain = DomainSpec(
    name="track",
    description="Track properties, mixer controls, clip/device queries, and routing",
    base_address="/live/track",
    index_params=[_track],
    endpoints=(
        _bulk_methods("track", _track_methods, idx=_track_idx, wildcard=True)
        + _bulk_props(
            "track", _track_props_ro, _track_props_rw, idx=_track_idx, wildcard=True
        )
        + _bulk_props("track", [], _track_mixer_rw, idx=_track_idx, wildcard=True)
        + _track_custom
        + [
            # Arrangement clip track-level operations (from arrangement_clip.py)
            _custom(
                "/live/track/create_arrangement_clip",
                "track",
                kind="method",
                params=[_track, _p("start_time", "float"), _p("length", "float")],
                returns=[_track, _p("clip_id", "int")],
                desc="Create MIDI clip in arrangement (Live 12+)",
                wildcard=True,
            ),
            _custom(
                "/live/track/delete_arrangement_clip",
                "track",
                kind="method",
                params=[_track, _p("clip_id", "int")],
                desc="Delete arrangement clip",
                wildcard=True,
            ),
            _custom(
                "/live/track/duplicate_to_arrangement",
                "track",
                kind="method",
                params=[_track, _p("clip_slot_id", "int"), _p("dest_time", "float")],
                returns=[_track, _p("clip_id", "int")],
                desc="Copy session clip to arrangement at dest_time",
                wildcard=True,
            ),
            _custom(
                "/live/track/split_arrangement_clip",
                "track",
                kind="method",
                params=[_track, _p("clip_id", "int"), _p("split_time", "float")],
                returns=[
                    _track,
                    _p("original_clip_id", "int"),
                    _p("new_clip_id", "int"),
                ],
                desc="Split arrangement clip at split_time",
                wildcard=True,
            ),
            _custom(
                "/live/track/move_arrangement_clip",
                "track",
                kind="method",
                params=[_track, _p("clip_id", "int"), _p("new_start", "float")],
                returns=[_track, _p("new_clip_id", "int")],
                desc="Move arrangement clip to new_start",
                wildcard=True,
            ),
            _custom(
                "/live/track/duplicate_arrangement_clip",
                "track",
                kind="method",
                params=[_track, _p("clip_id", "int"), _p("dest_time", "float")],
                returns=[_track, _p("new_clip_id", "int")],
                desc="Duplicate arrangement clip to dest_time",
                wildcard=True,
            ),
        ]
    ),
)


# =============================================================================
# Clip domain  (source: abletonosc/clip.py)
# =============================================================================

_clip_idx = [_track, _clip]

_clip_methods = ["fire", "stop", "duplicate_loop", "remove_notes_by_id"]

_clip_props_ro: list[tuple[str, str]] = [
    ("end_time", "float"),
    ("file_path", "str"),
    ("gain_display_string", "str"),
    ("has_groove", "bool"),
    ("is_midi_clip", "bool"),
    ("is_audio_clip", "bool"),
    ("is_overdubbing", "bool"),
    ("is_playing", "bool"),
    ("is_recording", "bool"),
    ("is_triggered", "bool"),
    ("length", "float"),
    ("playing_position", "float"),
    ("sample_length", "float"),
    ("start_time", "float"),
    ("will_record_on_start", "bool"),
]

_clip_props_rw: list[tuple[str, str]] = [
    ("color", "int"),
    ("color_index", "int"),
    ("end_marker", "float"),
    ("gain", "float"),
    ("launch_mode", "int"),
    ("launch_quantization", "int"),
    ("legato", "bool"),
    ("loop_end", "float"),
    ("loop_start", "float"),
    ("looping", "bool"),
    ("muted", "bool"),
    ("name", "str"),
    ("pitch_coarse", "int"),
    ("pitch_fine", "int"),
    ("position", "float"),
    ("ram_mode", "bool"),
    ("start_marker", "float"),
    ("velocity_amount", "float"),
    ("warp_mode", "int"),
    ("warping", "bool"),
]

_clip_custom: list[EndpointSpec] = [
    _custom(
        "/live/clip/get/notes",
        "clip",
        kind="get",
        params=[
            _track,
            _clip,
            _p("pitch_start", "int", optional=True),
            _p("pitch_span", "int", optional=True),
            _p("time_start", "float", optional=True),
            _p("time_span", "float", optional=True),
        ],
        returns=[
            _track,
            _clip,
            _p(
                "note_data",
                "any",
                "Flattened: pitch, start_time, duration, velocity, mute per note",
            ),
        ],
        desc="Get MIDI notes (all or filtered by pitch/time range)",
    ),
    _custom(
        "/live/clip/add/notes",
        "clip",
        kind="custom",
        params=[
            _track,
            _clip,
            _p(
                "note_data",
                "any",
                "Repeating groups of: pitch, start_time, duration, velocity, mute",
            ),
        ],
        desc="Add MIDI notes to clip",
    ),
    _custom(
        "/live/clip/remove/notes",
        "clip",
        kind="custom",
        params=[
            _track,
            _clip,
            _p("pitch_start", "int", optional=True),
            _p("pitch_span", "int", optional=True),
            _p("time_start", "float", optional=True),
            _p("time_span", "float", optional=True),
        ],
        desc="Remove MIDI notes (all or filtered by pitch/time range)",
    ),
    _custom(
        "/live/clips/filter",
        "clip",
        kind="custom",
        params=[_p("note_names", "str", "Note names like C, D#, Bb")],
        desc="Mute clips whose notes don't match the given note names",
    ),
    _custom(
        "/live/clips/unfilter",
        "clip",
        kind="custom",
        params=[
            _p("track_start", "int", optional=True),
            _p("track_end", "int", optional=True),
        ],
        desc="Unmute all clips (optionally within a track range)",
    ),
]

clip_domain = DomainSpec(
    name="clip",
    description="Clip properties, MIDI note manipulation, and clip filtering",
    base_address="/live/clip",
    index_params=[_track, _clip],
    endpoints=(
        _bulk_methods("clip", _clip_methods, idx=_clip_idx)
        + _bulk_props("clip", _clip_props_ro, _clip_props_rw, idx=_clip_idx)
        + _clip_custom
    ),
)


# =============================================================================
# Arrangement Clip domain  (source: abletonosc/arrangement_clip.py)
# =============================================================================

_ac_idx = [_track, _p("clip_id", "int", "Arrangement clip index")]

_ac_props_ro: list[tuple[str, str]] = [
    ("start_time", "float"),
    ("end_time", "float"),
    ("length", "float"),
    ("is_midi_clip", "bool"),
    ("color", "int"),
]

_ac_props_rw: list[tuple[str, str]] = [
    ("name", "str"),
    ("loop_start", "float"),
    ("loop_end", "float"),
    ("start_marker", "float"),
    ("end_marker", "float"),
    ("looping", "bool"),
]

_ac_custom: list[EndpointSpec] = [
    _custom(
        "/live/arrangement_clip/get/notes",
        "arrangement_clip",
        kind="get",
        params=[
            _track,
            _p("clip_id", "int"),
            _p("pitch_start", "int", optional=True),
            _p("pitch_span", "int", optional=True),
            _p("time_start", "float", optional=True),
            _p("time_span", "float", optional=True),
        ],
        returns=[
            _track,
            _p("clip_id", "int"),
            _p(
                "note_data",
                "any",
                "Flattened: pitch, start_time, duration, velocity, mute per note",
            ),
        ],
        desc="Get MIDI notes from arrangement clip",
    ),
    _custom(
        "/live/arrangement_clip/add/notes",
        "arrangement_clip",
        kind="custom",
        params=[
            _track,
            _p("clip_id", "int"),
            _p(
                "note_data",
                "any",
                "Repeating groups of: pitch, start_time, duration, velocity, mute",
            ),
        ],
        desc="Add MIDI notes to arrangement clip",
    ),
    _custom(
        "/live/arrangement_clip/remove/notes",
        "arrangement_clip",
        kind="custom",
        params=[
            _track,
            _p("clip_id", "int"),
            _p("pitch_start", "int", optional=True),
            _p("pitch_span", "int", optional=True),
            _p("time_start", "float", optional=True),
            _p("time_span", "float", optional=True),
        ],
        desc="Remove MIDI notes from arrangement clip",
    ),
]

arrangement_clip_domain = DomainSpec(
    name="arrangement_clip",
    description="Arrangement clip properties, MIDI note manipulation",
    base_address="/live/arrangement_clip",
    index_params=[_track, _p("clip_id", "int", "Arrangement clip index")],
    endpoints=(
        _bulk_props("arrangement_clip", _ac_props_ro, _ac_props_rw, idx=_ac_idx)
        + _ac_custom
    ),
)


# =============================================================================
# Clip Slot domain  (source: abletonosc/clip_slot.py)
# =============================================================================

_cs_idx = [_track, _clip]

_cs_methods = ["fire", "stop", "create_clip", "delete_clip"]

_cs_props_ro: list[tuple[str, str]] = [
    ("has_clip", "bool"),
    ("controls_other_clips", "bool"),
    ("is_group_slot", "bool"),
    ("is_playing", "bool"),
    ("is_triggered", "bool"),
    ("playing_status", "int"),
    ("will_record_on_start", "bool"),
]

_cs_props_rw: list[tuple[str, str]] = [
    ("has_stop_button", "bool"),
]

_cs_custom: list[EndpointSpec] = [
    _custom(
        "/live/clip_slot/duplicate_clip_to",
        "clip_slot",
        kind="method",
        params=[_track, _clip, _p("target_track", "int"), _p("target_clip", "int")],
        desc="Duplicate clip to another slot",
    ),
]

clip_slot_domain = DomainSpec(
    name="clip_slot",
    description="Clip slot state, firing, and clip management",
    base_address="/live/clip_slot",
    index_params=[_track, _clip],
    endpoints=(
        _bulk_methods("clip_slot", _cs_methods, idx=_cs_idx)
        + _bulk_props("clip_slot", _cs_props_ro, _cs_props_rw, idx=_cs_idx)
        + _cs_custom
    ),
)


# =============================================================================
# Device domain  (source: abletonosc/device.py)
# =============================================================================

_dev_idx = [_track, _device]

_dev_props_ro: list[tuple[str, str]] = [
    ("class_name", "str"),
    ("name", "str"),
    ("type", "int"),
]

_dev_custom: list[EndpointSpec] = [
    # Batch parameter queries
    _custom(
        "/live/device/get/num_parameters",
        "device",
        kind="get",
        params=_dev_idx,
        returns=[_track, _device, _p("count", "int")],
        desc="Get number of parameters",
    ),
    _custom(
        "/live/device/get/parameters/name",
        "device",
        kind="get",
        params=_dev_idx,
        returns=[_track, _device, _p("names", "str")],
        desc="Get all parameter names",
    ),
    _custom(
        "/live/device/get/parameters/value",
        "device",
        kind="get",
        params=_dev_idx,
        returns=[_track, _device, _p("values", "float")],
        desc="Get all parameter values",
    ),
    _custom(
        "/live/device/get/parameters/min",
        "device",
        kind="get",
        params=_dev_idx,
        returns=[_track, _device, _p("mins", "float")],
        desc="Get all parameter minimums",
    ),
    _custom(
        "/live/device/get/parameters/max",
        "device",
        kind="get",
        params=_dev_idx,
        returns=[_track, _device, _p("maxs", "float")],
        desc="Get all parameter maximums",
    ),
    _custom(
        "/live/device/get/parameters/is_quantized",
        "device",
        kind="get",
        params=_dev_idx,
        returns=[_track, _device, _p("quantized", "bool")],
        desc="Get whether each parameter is quantized",
    ),
    _custom(
        "/live/device/set/parameters/value",
        "device",
        kind="set",
        params=[_track, _device, _p("values", "float", "Values for all parameters")],
        desc="Set all parameter values at once",
    ),
    # Individual parameter access
    _custom(
        "/live/device/get/parameter/value",
        "device",
        kind="get",
        params=[_track, _device, _p("param_index", "int")],
        returns=[_track, _device, _p("param_index", "int"), _p("value", "float")],
        desc="Get a single parameter value",
    ),
    _custom(
        "/live/device/get/parameter/value_string",
        "device",
        kind="get",
        params=[_track, _device, _p("param_index", "int")],
        returns=[_track, _device, _p("param_index", "int"), _p("value_string", "str")],
        desc="Get parameter value as display string (e.g. '2500 Hz')",
    ),
    _custom(
        "/live/device/set/parameter/value",
        "device",
        kind="set",
        params=[_track, _device, _p("param_index", "int"), _p("value", "float")],
        desc="Set a single parameter value",
    ),
    _custom(
        "/live/device/get/parameter/name",
        "device",
        kind="get",
        params=[_track, _device, _p("param_index", "int")],
        returns=[_track, _device, _p("param_index", "int"), _p("name", "str")],
        desc="Get a single parameter name",
    ),
    # Parameter value listener
    _custom(
        "/live/device/start_listen/parameter/value",
        "device",
        kind="listen_start",
        params=[_track, _device, _p("param_index", "int")],
        desc="Listen for parameter value changes (sends value + value_string)",
    ),
    _custom(
        "/live/device/stop_listen/parameter/value",
        "device",
        kind="listen_stop",
        params=[_track, _device, _p("param_index", "int")],
        desc="Stop listening for parameter value changes",
    ),
    # Chain operations (rack devices)
    _custom(
        "/live/device/get/num_chains",
        "device",
        kind="get",
        params=_dev_idx,
        returns=[_track, _device, _p("count", "int")],
        desc="Get number of chains (rack devices only)",
    ),
    _custom(
        "/live/device/get/chains/name",
        "device",
        kind="get",
        params=_dev_idx,
        returns=[_track, _device, _p("names", "str")],
        desc="Get all chain names",
    ),
    _custom(
        "/live/device/get/chains/mute",
        "device",
        kind="get",
        params=_dev_idx,
        returns=[_track, _device, _p("mutes", "bool")],
        desc="Get all chain mute states",
    ),
    _custom(
        "/live/device/get/chains/volume",
        "device",
        kind="get",
        params=_dev_idx,
        returns=[_track, _device, _p("volumes", "float")],
        desc="Get all chain volumes",
    ),
    _custom(
        "/live/device/set/chain/name",
        "device",
        kind="set",
        params=[_track, _device, _p("chain_id", "int"), _p("name", "str")],
        desc="Set chain name",
    ),
    _custom(
        "/live/device/set/chain/volume",
        "device",
        kind="set",
        params=[_track, _device, _p("chain_id", "int"), _p("value", "float")],
        desc="Set chain volume",
    ),
    _custom(
        "/live/device/set/chain/mute",
        "device",
        kind="set",
        params=[_track, _device, _p("chain_id", "int"), _p("muted", "bool")],
        desc="Set chain mute state",
    ),
]

device_domain = DomainSpec(
    name="device",
    description="Device properties, parameter access, parameter listeners, and rack chain operations",
    base_address="/live/device",
    index_params=[_track, _device],
    endpoints=(_bulk_props("device", _dev_props_ro, [], idx=_dev_idx) + _dev_custom),
)


# =============================================================================
# Scene domain  (source: abletonosc/scene.py)
# =============================================================================

_scene_idx = [_scene]

_scene_methods = ["fire", "fire_as_selected"]

_scene_props_ro: list[tuple[str, str]] = [
    ("is_empty", "bool"),
    ("is_triggered", "bool"),
]

_scene_props_rw: list[tuple[str, str]] = [
    ("color", "int"),
    ("color_index", "int"),
    ("name", "str"),
    ("tempo", "float"),
    ("tempo_enabled", "bool"),
    ("time_signature_numerator", "int"),
    ("time_signature_denominator", "int"),
    ("time_signature_enabled", "bool"),
]

_scene_custom: list[EndpointSpec] = [
    _custom(
        "/live/scene/fire_selected",
        "scene",
        kind="method",
        desc="Fire the currently selected scene",
    ),
]

scene_domain = DomainSpec(
    name="scene",
    description="Scene properties, tempo, time signature, and firing",
    base_address="/live/scene",
    index_params=[_scene],
    endpoints=(
        _bulk_methods("scene", _scene_methods, idx=_scene_idx)
        + _bulk_props("scene", _scene_props_ro, _scene_props_rw, idx=_scene_idx)
        + _scene_custom
    ),
)


# =============================================================================
# View domain  (source: abletonosc/view.py)
# =============================================================================

view_domain = DomainSpec(
    name="view",
    description="Selection state for scene, track, clip, and device",
    base_address="/live/view",
    index_params=[],
    endpoints=[
        _custom(
            "/live/view/get/selected_scene",
            "view",
            kind="get",
            returns=[_p("scene_index", "int")],
            desc="Get index of selected scene",
        ),
        _custom(
            "/live/view/get/selected_track",
            "view",
            kind="get",
            returns=[_p("track_index", "int")],
            desc="Get index of selected track",
        ),
        _custom(
            "/live/view/get/selected_clip",
            "view",
            kind="get",
            returns=[_p("track_index", "int"), _p("scene_index", "int")],
            desc="Get selected clip position (track, scene)",
        ),
        _custom(
            "/live/view/get/selected_device",
            "view",
            kind="get",
            returns=[_p("track_index", "int"), _p("device_index", "int")],
            desc="Get selected device position (track, device)",
        ),
        _custom(
            "/live/view/set/selected_scene",
            "view",
            kind="set",
            params=[_p("scene_index", "int")],
            desc="Set selected scene by index",
        ),
        _custom(
            "/live/view/set/selected_track",
            "view",
            kind="set",
            params=[_p("track_index", "int")],
            desc="Set selected track by index",
        ),
        _custom(
            "/live/view/set/selected_clip",
            "view",
            kind="set",
            params=[_p("track_index", "int"), _p("scene_index", "int")],
            desc="Set selected clip by track and scene index",
        ),
        _custom(
            "/live/view/set/selected_device",
            "view",
            kind="set",
            params=[_p("track_index", "int"), _p("device_index", "int")],
            returns=[_p("track_index", "int"), _p("device_index", "int")],
            desc="Set selected device by track and device index",
        ),
        _custom(
            "/live/view/start_listen/selected_scene",
            "view",
            kind="listen_start",
            desc="Listen for selected scene changes",
        ),
        _custom(
            "/live/view/start_listen/selected_track",
            "view",
            kind="listen_start",
            desc="Listen for selected track changes",
        ),
        _custom(
            "/live/view/stop_listen/selected_scene",
            "view",
            kind="listen_stop",
            desc="Stop listening for selected scene changes",
        ),
        _custom(
            "/live/view/stop_listen/selected_track",
            "view",
            kind="listen_stop",
            desc="Stop listening for selected track changes",
        ),
    ],
)


# =============================================================================
# Application domain  (source: abletonosc/application.py)
# =============================================================================

application_domain = DomainSpec(
    name="application",
    description="Application version and system usage",
    base_address="/live/application",
    index_params=[],
    endpoints=[
        _custom(
            "/live/application/get/version",
            "application",
            kind="get",
            returns=[_p("major", "int"), _p("minor", "int")],
            desc="Get Ableton Live version (major, minor)",
        ),
        _custom(
            "/live/application/get/average_process_usage",
            "application",
            kind="get",
            returns=[_p("usage", "float")],
            desc="Get average CPU process usage",
        ),
    ],
)


# =============================================================================
# MIDI Map domain  (source: abletonosc/midimap.py)
# =============================================================================

midimap_domain = DomainSpec(
    name="midimap",
    description="MIDI CC mapping to device parameters",
    base_address="/live/midimap",
    index_params=[],
    endpoints=[
        _custom(
            "/live/midimap/map_cc",
            "midimap",
            kind="method",
            params=[
                _track,
                _p("device_index", "int"),
                _p("parameter_index", "int"),
                _p("channel", "int"),
                _p("cc", "int"),
            ],
            desc="Map a MIDI CC to a device parameter",
        ),
    ],
)


# =============================================================================
# Browser domain  (source: abletonosc/browser.py — remix-mcp fork)
# =============================================================================

browser_domain = DomainSpec(
    name="browser",
    description="Browser operations: load instruments/effects/samples, search, navigate, hotswap, preview",
    base_address="/live/browser",
    index_params=[],
    endpoints=[
        # Instruments
        _custom(
            "/live/browser/load_instrument",
            "browser",
            kind="method",
            params=[_p("name", "str", "Instrument name to search for")],
            returns=[_p("loaded_name", "str")],
            desc="Load an instrument by name onto the selected track",
        ),
        _custom(
            "/live/browser/load_drum_kit",
            "browser",
            kind="method",
            params=[_p("name", "str", "Drum kit name", optional=True)],
            returns=[_p("loaded_name", "str")],
            desc="Load a drum kit onto the selected track",
        ),
        _custom(
            "/live/browser/load_default_instrument",
            "browser",
            kind="method",
            returns=[_p("loaded_name", "str")],
            desc="Load a default instrument (prefers synths: Drift, Analog, Wavetable)",
        ),
        # Audio & MIDI effects
        _custom(
            "/live/browser/load_audio_effect",
            "browser",
            kind="method",
            params=[_p("name", "str", "Effect name")],
            returns=[_p("loaded_name", "str")],
            desc="Load an audio effect by name onto the selected track",
        ),
        _custom(
            "/live/browser/load_midi_effect",
            "browser",
            kind="method",
            params=[_p("name", "str", "Effect name")],
            returns=[_p("loaded_name", "str")],
            desc="Load a MIDI effect by name onto the selected track",
        ),
        _custom(
            "/live/browser/load_default_audio_effect",
            "browser",
            kind="method",
            returns=[_p("loaded_name", "str")],
            desc="Load a default audio effect (prefers Reverb, Delay, EQ Eight)",
        ),
        _custom(
            "/live/browser/load_default_midi_effect",
            "browser",
            kind="method",
            returns=[_p("loaded_name", "str")],
            desc="Load a default MIDI effect (prefers Arpeggiator, Chord, Scale)",
        ),
        _custom(
            "/live/browser/list_audio_effects",
            "browser",
            kind="get",
            returns=[_p("names", "str")],
            desc="List available audio effect categories",
        ),
        _custom(
            "/live/browser/list_midi_effects",
            "browser",
            kind="get",
            returns=[_p("names", "str")],
            desc="List available MIDI effect categories",
        ),
        # Sounds & Presets
        _custom(
            "/live/browser/load_sound",
            "browser",
            kind="method",
            params=[_p("name", "str", "Sound preset name")],
            returns=[_p("loaded_name", "str")],
            desc="Load a sound preset by name",
        ),
        _custom(
            "/live/browser/list_sounds",
            "browser",
            kind="get",
            returns=[_p("names", "str")],
            desc="List available sound preset categories",
        ),
        # Samples & Clips
        _custom(
            "/live/browser/load_sample",
            "browser",
            kind="method",
            params=[_p("name", "str", "Sample name")],
            returns=[_p("loaded_name", "str")],
            desc="Load a sample by name onto the selected track",
        ),
        _custom(
            "/live/browser/load_clip",
            "browser",
            kind="method",
            params=[_p("name", "str", "Clip name")],
            returns=[_p("loaded_name", "str")],
            desc="Load a clip by name",
        ),
        _custom(
            "/live/browser/list_samples",
            "browser",
            kind="get",
            params=[_p("category", "str", optional=True)],
            returns=[_p("names", "str")],
            desc="List available samples, optionally within a category",
        ),
        _custom(
            "/live/browser/list_clips",
            "browser",
            kind="get",
            params=[_p("category", "str", optional=True)],
            returns=[_p("names", "str")],
            desc="List available clips, optionally within a category",
        ),
        # Plugins & Max4Live
        _custom(
            "/live/browser/load_plugin",
            "browser",
            kind="method",
            params=[_p("name", "str", "Plugin name")],
            returns=[_p("loaded_name", "str")],
            desc="Load a VST/AU plugin by name",
        ),
        _custom(
            "/live/browser/load_max_device",
            "browser",
            kind="method",
            params=[_p("name", "str", "Max for Live device name")],
            returns=[_p("loaded_name", "str")],
            desc="Load a Max for Live device by name",
        ),
        _custom(
            "/live/browser/list_plugins",
            "browser",
            kind="get",
            returns=[_p("names", "str")],
            desc="List available VST/AU plugins",
        ),
        _custom(
            "/live/browser/list_max_devices",
            "browser",
            kind="get",
            returns=[_p("names", "str")],
            desc="List available Max for Live devices",
        ),
        # Browser navigation
        _custom(
            "/live/browser/browse",
            "browser",
            kind="get",
            params=[
                _p(
                    "category",
                    "str",
                    "instruments, drums, sounds, audio_effects, midi_effects, max_for_live, plugins, clips, samples, packs, user_library",
                )
            ],
            returns=[_p("names", "str")],
            desc="Browse a top-level browser category",
        ),
        _custom(
            "/live/browser/browse_path",
            "browser",
            kind="get",
            params=[_p("category", "str"), _p("path", "str")],
            returns=[_p("names", "str")],
            desc="Browse a specific path within a category",
        ),
        _custom(
            "/live/browser/search",
            "browser",
            kind="get",
            params=[_p("query", "str")],
            returns=[_p("results", "str", "Alternating category, name pairs")],
            desc="Search for items across all browser categories",
        ),
        _custom(
            "/live/browser/get_item_info",
            "browser",
            kind="get",
            params=[_p("category", "str"), _p("name", "str")],
            returns=[
                _p("name", "str"),
                _p("is_loadable", "bool"),
                _p("is_device", "bool"),
                _p("has_children", "bool"),
                _p("child_count", "int"),
            ],
            desc="Get detailed info about a browser item",
        ),
        # User library
        _custom(
            "/live/browser/list_user_presets",
            "browser",
            kind="get",
            params=[_p("category", "str", optional=True)],
            returns=[_p("names", "str")],
            desc="List presets in the user library",
        ),
        _custom(
            "/live/browser/load_user_preset",
            "browser",
            kind="method",
            params=[_p("name", "str", "Preset name or path")],
            returns=[_p("loaded_name", "str")],
            desc="Load a preset from the user library",
        ),
        # Hotswap & Preview
        _custom(
            "/live/browser/hotswap_start",
            "browser",
            kind="method",
            params=[_p("track_index", "int"), _p("device_index", "int")],
            returns=[_p("device_name", "str")],
            desc="Enter hotswap mode for a device",
        ),
        _custom(
            "/live/browser/hotswap_load",
            "browser",
            kind="method",
            params=[_p("name", "str")],
            returns=[_p("loaded_name", "str")],
            desc="Load an item via hotswap (must call hotswap_start first)",
        ),
        _custom(
            "/live/browser/preview_sample",
            "browser",
            kind="method",
            params=[_p("name", "str", "Sample name")],
            returns=[_p("name", "str")],
            desc="Preview a sample before loading",
        ),
        _custom(
            "/live/browser/stop_preview",
            "browser",
            kind="method",
            returns=[_p("success", "bool")],
            desc="Stop sample preview playback",
        ),
    ],
)


# =============================================================================
# Internal/API domain  (endpoints from osc_server, not in handler files)
# =============================================================================

internal_domain = DomainSpec(
    name="internal",
    description="Internal AbletonOSC control and lifecycle endpoints",
    base_address="/live",
    index_params=[],
    endpoints=[
        _custom("/live/test", "internal", kind="method", desc="Test connectivity"),
        _custom(
            "/live/api/reload",
            "internal",
            kind="method",
            desc="Reload the OSC API handlers",
        ),
        _custom(
            "/live/api/get/log_level",
            "internal",
            kind="get",
            returns=[_p("level", "int")],
            desc="Get current log level",
        ),
        _custom(
            "/live/api/set/log_level",
            "internal",
            kind="set",
            params=[_p("level", "int")],
            desc="Set log level",
        ),
        _custom(
            "/live/startup",
            "internal",
            kind="custom",
            desc="Sent on API startup (outbound notification)",
        ),
        _custom(
            "/live/error",
            "internal",
            kind="custom",
            desc="Error notification (outbound)",
        ),
    ],
)


# =============================================================================
# Assembled spec
# =============================================================================

spec = AbletonOSCSpec(
    version="1.0",
    source="AbletonOSC (smithclay/AbletonOSC remix-mcp fork)",
    domains=[
        song_domain,
        track_domain,
        clip_domain,
        arrangement_clip_domain,
        clip_slot_domain,
        device_domain,
        scene_domain,
        view_domain,
        application_domain,
        midimap_domain,
        browser_domain,
        internal_domain,
    ],
)


if __name__ == "__main__":
    print(
        f"{len(spec.domains)} domains, {sum(len(d.endpoints) for d in spec.domains)} endpoints"
    )
    for d in spec.domains:
        print(f"  {d.name}: {len(d.endpoints)} endpoints")
