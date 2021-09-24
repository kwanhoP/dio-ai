import copy
import json
import re
from fractions import Fraction
from typing import Dict

import miditoolkit
import mido
import numpy as np

from dioai.preprocessor.encoder.remi import chord_recognition, constant
from dioai.preprocessor.encoder.remi.exceptions import InvalidMidiError, InvalidMidiErrorMessage

# parameters for input
DEFAULT_VELOCITY_BINS = np.linspace(0, 128, 32 + 1, dtype=np.int)
DEFAULT_TEMPO_INTERVALS = [range(30, 90), range(90, 150), range(150, 210)]

# parameters for output
DEFAULT_RESOLUTION = 480


def extract_events(
    input_path,
    position_resolution,
    duration_bins,
    tick_per_bar=None,
    tick_per_beat=None,
    chord_progression=None,
    audio_key=None,
    use_backoffice_chord=True,
):
    note_items, tempo_items = read_items(input_path, tick_per_beat)
    max_time = note_items[-1].end
    if not use_backoffice_chord:
        chord_items = extract_chords(note_items)
        items = chord_items + tempo_items + note_items
    else:
        items = tempo_items + note_items
    groups = group_items(items, max_time, ticks_per_bar=tick_per_bar)
    events = item2event(groups, position_resolution, duration_bins)
    if use_backoffice_chord:
        audio_key = audio_key.replace("minor", "").replace("major", "")
        if "raw" in input_path:
            new_chords = chord_progression[0]
        else:
            try:
                aug_key = (
                    input_path.split("/")[-1]
                    .split("_")[1]
                    .replace("minor", "")
                    .replace("major", "")
                )
            except IndexError:
                print(f"path index error_{input_path}")
            try:
                new_chords = sync_key_augment(chord_progression[0], aug_key, audio_key)
            except ValueError:
                print(f"sync value error_{input_path}")
        events = insert_chord_on_event(events, new_chords)

    return events


def extract_decimal_chord(event_idx, event_time, dec_idx, events):
    "코드 위치가 정수(마디) 단위가 아닌 한 마디 안에서 바뀌는 경우 event index와 time을 구하는 함수"

    for dec in dec_idx:
        intergers = int(dec)
        decimals = dec - intergers

        n = Fraction(decimals).numerator
        d = Fraction(decimals).denominator
        resolution = 32
        scale_factor = resolution / d
        n = n * scale_factor

        bar_flags = -1
        for idx, e in enumerate(events):
            if e.name == "Bar":
                bar_flags += 1
                if bar_flags == intergers:
                    bar_position_time = events[idx + 1].time
                    bar_time = bar_position_time if bar_position_time is not None else 0
                    tick_per_32notes = 60
                    chord_time = bar_time + tick_per_32notes * n
            try:
                if e.name == "Position" and e.time == chord_time:
                    event_idx.append(idx + 1)
                    event_time.append(int(chord_time))
            except UnboundLocalError:
                continue
    return sorted(list(set(event_idx))), sorted(list(set(event_time)))


def detect_chord(chord_progression):
    "백오피스의 코드 진행 정보로 코드 위치와 정보를 구하는 함수"

    chord = []
    chord_idx = []
    for idx, c in enumerate(chord_progression):
        if not chord:
            chord.append(c)
            chord_idx.append(idx)
        else:
            if c != chord[-1]:
                chord.append(c)
                chord_idx.append(idx)
    return list(np.array(chord_idx) / 8), chord


def insert_chord_on_event(events, chord_progression):
    "백오피스 코드 정보를 remi event에 수동으로 할당 하는 함수"

    bar_idx, bar_time = extract_bar_position(events)
    chord_idx_lst, chords = detect_chord(chord_progression)

    int_idx = list(filter(lambda x: round(x) == x, chord_idx_lst))  # 마디 단위 코드 진행(정수 인덱스)
    dec_idx = list(filter(lambda x: round(x) != x, chord_idx_lst))  # 마디 사이에 코드 바뀜(소수 인덱스)

    event_idx = [c for idx, c in enumerate(bar_idx) if idx in int_idx]
    event_time = [c for idx, c in enumerate(bar_time) if idx in int_idx]

    if dec_idx:  # 한 마디 내에 코드가 바뀌는 경우
        event_idx, event_time = extract_decimal_chord(event_idx, event_time, dec_idx, events)

    list_flags = 0
    for idx, time, chord in zip(event_idx, event_time, chords):
        events.insert(idx + list_flags, Event("Chord", time, chord, chord))
        list_flags += 1
    return events


def extract_bar_position(events):
    "remi bar event 위치 구하는 함수"

    bar_idx = []
    bar_time = []
    for idx, e in enumerate(events):
        if e.name == "Bar":
            bar_idx.append(idx + 2)
            bar_time.append(events[idx + 1].time)
    return bar_idx, bar_time


def sync_key_augment(chords, aug_key, origin_key):
    "key augment 된 샘플의 코드 진행을 맞춰주는 함수"

    chord_lst = [
        "a",
        "a#",
        "b",
        "c",
        "c#",
        "d",
        "d#",
        "e",
        "f",
        "f#",
        "g",
        "g#",
        "ab",
        "bb",
        "db",
        "eb",
        "gb",
    ]
    chord2symbol = {k: v for k, v in zip(chord_lst, range(12))}
    chord2symbol["ab"] = 11
    chord2symbol["bb"] = 1
    chord2symbol["db"] = 4
    chord2symbol["eb"] = 6
    chord2symbol["gb"] = 9
    symbol2chord = {v: k for k, v in chord2symbol.items()}

    basic_chord = []
    for c in chords:
        match = re.findall(r"([A-Z,#,b]+)", c)
        basic_chord.append(match[0])

    symbol_lst = [chord2symbol[c.lower()] for c in basic_chord]

    origin_key_symbol = chord_lst.index(origin_key)

    augment_key_symbol = chord_lst.index(aug_key)

    key_diff = origin_key_symbol - augment_key_symbol
    key_chage = abs(key_diff)
    if key_diff < 0:
        new_symbol_lst = []
        for s in symbol_lst:
            new_s = s + key_chage
            if new_s >= 12:
                new_s = new_s - 12
            new_symbol_lst.append(new_s)
    else:
        new_symbol_lst = []
        for s in symbol_lst:
            new_s = s - key_chage
            if new_s < 0:
                new_s = new_s + 12
            new_symbol_lst.append(new_s)

    new_chord_lst = [symbol2chord[s] for s in new_symbol_lst]
    return new_chord_lst


# define "Item" for general storage
class Item(object):
    def __init__(self, name, start, end, velocity, pitch):
        self.name = name
        self.start = start
        self.end = end
        self.velocity = velocity
        self.pitch = pitch

    def __repr__(self):
        return "Item(name={}, start={}, end={}, velocity={}, pitch={})".format(
            self.name, self.start, self.end, self.velocity, self.pitch
        )


# read notes and tempo changes from midi (assume there is only one track)
def read_items(file_path, tick_per_beat):
    midi_obj = miditoolkit.midi.parser.MidiFile(file_path)
    # note
    note_items = []
    notes = midi_obj.instruments[0].notes
    notes.sort(key=lambda x: (x.start, x.pitch))
    for note in notes:
        note_items.append(
            Item(
                name="Note",
                start=note.start,
                end=note.end,
                velocity=note.velocity,
                pitch=note.pitch,
            )
        )
    note_items.sort(key=lambda x: x.start)
    # tempo
    tempo_items = []
    for tempo in midi_obj.tempo_changes:
        tempo_items.append(
            Item(name="Tempo", start=tempo.time, end=None, velocity=None, pitch=int(tempo.tempo))
        )
    tempo_items.sort(key=lambda x: x.start)
    # expand to all beat
    max_tick = tempo_items[-1].start
    existing_ticks = {item.start: item.pitch for item in tempo_items}
    wanted_ticks = np.arange(0, max_tick + 1, tick_per_beat)
    output = []
    for tick in wanted_ticks:
        if tick in existing_ticks:
            output.append(
                Item(name="Tempo", start=tick, end=None, velocity=None, pitch=existing_ticks[tick])
            )
        else:
            output.append(
                Item(name="Tempo", start=tick, end=None, velocity=None, pitch=output[-1].pitch)
            )
    tempo_items = output
    return note_items, tempo_items


# quantize items
def quantize_items(items, ticks=120):
    # grid
    grids = np.arange(0, items[-1].start, ticks, dtype=int)
    # process
    for item in items:
        index = np.argmin(abs(grids - item.start))
        shift = grids[index] - item.start
        item.start += shift
        item.end += shift
    return items


# extract chord
def extract_chords(items):
    method = chord_recognition.MIDIChord()
    chords = method.extract(notes=items)
    output = []
    for chord in chords:
        # chord: List [start tick, end tick, chord info]
        # chord info: str 'root note/base note'
        # root note: str 'f#:m'
        pitch = chord[2].split("/")[0]
        chord_pitch = pitch.split(":")[0]
        chord_tone = pitch.split(":")[1]
        if chord_tone == "M":
            pitch = chord_pitch
        else:
            pitch = chord_pitch + chord_tone
        output.append(
            Item(
                name="Chord",
                start=chord[0],
                end=chord[1],
                velocity=None,
                pitch=pitch,
            )
        )
    return output


# group items
def group_items(items, max_time, ticks_per_bar=DEFAULT_RESOLUTION * 4):
    items.sort(key=lambda x: x.start)
    downbeats = np.arange(0, max_time + ticks_per_bar, ticks_per_bar)
    groups = []
    for db1, db2 in zip(downbeats[:-1], downbeats[1:]):
        insiders = []
        for item in items:
            if (item.start >= db1) and (item.start < db2):
                insiders.append(item)
        overall = [db1] + insiders + [db2]
        groups.append(overall)
    return groups


# define "Event" for event storage
class Event(object):
    def __init__(self, name, time, value, text):
        self.name = name
        self.time = time
        self.value = value
        self.text = text

    def __repr__(self):
        return "Event(name={}, time={}, value={}, text={})".format(
            self.name, self.time, self.value, self.text
        )


# item to event
def item2event(groups, DEFAULT_FRACTION, DEFAULT_DURATION_BINS):
    events = []
    n_downbeat = 0
    for i in range(len(groups)):
        if "Note" not in [item.name for item in groups[i][1:-1]]:
            continue
        bar_st, bar_et = groups[i][0], groups[i][-1]
        n_downbeat += 1
        events.append(Event(name="Bar", time=None, value=None, text="{}".format(n_downbeat)))
        for item in groups[i][1:-1]:
            # position
            flags = np.linspace(bar_st, bar_et, DEFAULT_FRACTION, endpoint=False)
            index = np.argmin(abs(flags - item.start))
            events.append(
                Event(
                    name="Position",
                    time=item.start,
                    value="{}/{}".format(index + 1, DEFAULT_FRACTION),
                    text="{}".format(item.start),
                )
            )
            if item.name == "Note":
                # velocity
                velocity_index = (
                    np.searchsorted(DEFAULT_VELOCITY_BINS, item.velocity, side="right") - 1
                )
                events.append(
                    Event(
                        name="Note Velocity",
                        time=item.start,
                        value=velocity_index,
                        text="{}/{}".format(item.velocity, DEFAULT_VELOCITY_BINS[velocity_index]),
                    )
                )
                # pitch
                events.append(
                    Event(
                        name="Note On",
                        time=item.start,
                        value=item.pitch,
                        text="{}".format(item.pitch),
                    )
                )
                # duration
                duration = item.end - item.start
                index = np.argmin(abs(DEFAULT_DURATION_BINS - duration))
                events.append(
                    Event(
                        name="Note Duration",
                        time=item.start,
                        value=index,
                        text="{}/{}".format(duration, DEFAULT_DURATION_BINS[index]),
                    )
                )
            elif item.name == "Chord":
                events.append(
                    Event(
                        name="Chord",
                        time=item.start,
                        value=item.pitch,
                        text="{}".format(item.pitch),
                    )
                )
            elif item.name == "Tempo":
                tempo = item.pitch
                if tempo in DEFAULT_TEMPO_INTERVALS[0]:
                    tempo_style = Event("Tempo Class", item.start, "slow", None)
                    tempo_value = Event(
                        "Tempo Value", item.start, tempo - DEFAULT_TEMPO_INTERVALS[0].start, None
                    )
                elif tempo in DEFAULT_TEMPO_INTERVALS[1]:
                    tempo_style = Event("Tempo Class", item.start, "mid", None)
                    tempo_value = Event(
                        "Tempo Value", item.start, tempo - DEFAULT_TEMPO_INTERVALS[1].start, None
                    )
                elif tempo in DEFAULT_TEMPO_INTERVALS[2]:
                    tempo_style = Event("Tempo Class", item.start, "fast", None)
                    tempo_value = Event(
                        "Tempo Value", item.start, tempo - DEFAULT_TEMPO_INTERVALS[2].start, None
                    )
                elif tempo < DEFAULT_TEMPO_INTERVALS[0].start:
                    tempo_style = Event("Tempo Class", item.start, "slow", None)
                    tempo_value = Event("Tempo Value", item.start, 0, None)
                elif tempo > DEFAULT_TEMPO_INTERVALS[2].stop:
                    tempo_style = Event("Tempo Class", item.start, "fast", None)
                    tempo_value = Event("Tempo Value", item.start, 59, None)
                events.append(tempo_style)
                events.append(tempo_value)
    return events


#############################################################################################
# WRITE MIDI
#############################################################################################
def word_to_event(words, word2event):
    events = []
    for word in words:
        try:
            event_name, event_value = word2event[word].split("_")
        except KeyError:
            continue
        events.append(Event(event_name, None, event_value, None))
    return events


def write_midi(
    words,
    word2event,
    output_path,
    prompt_path=None,
    DEFAULT_FRACTION=32,
    DEFAULT_DURATION_BINS=np.arange(30, 3841, 30, dtype=int),
    beat_per_bar=4,
):
    events = word_to_event(words, word2event)
    # get downbeat and note (no time)
    temp_notes = []
    temp_chords = []
    temp_tempos = []
    for i in range(len(events) - 3):
        if events[i].name == "Bar" and i > 0:
            temp_notes.append("Bar")
            temp_chords.append("Bar")
            temp_tempos.append("Bar")
        elif (
            events[i].name == "Position"
            and events[i + 1].name == "Note Velocity"
            and events[i + 2].name == "Note On"
            and events[i + 3].name == "Note Duration"
        ):
            # start time and end time from position
            position = int(events[i].value.split("/")[0]) - 1
            # velocity
            index = int(events[i + 1].value)
            velocity = int(DEFAULT_VELOCITY_BINS[index])
            # pitch
            pitch = int(events[i + 2].value)
            # duration
            index = int(events[i + 3].value)
            duration = DEFAULT_DURATION_BINS[index]
            # adding
            temp_notes.append([position, velocity, pitch, duration])
        elif events[i].name == "Position" and events[i + 1].name == "Chord":
            position = int(events[i].value.split("/")[0]) - 1
            temp_chords.append([position, events[i + 1].value])
        elif (
            events[i].name == "Position"
            and events[i + 1].name == "Tempo Class"
            and events[i + 2].name == "Tempo Value"
        ):
            position = int(events[i].value.split("/")[0]) - 1
            if events[i + 1].value == "slow":
                tempo = DEFAULT_TEMPO_INTERVALS[0].start + int(events[i + 2].value)
            elif events[i + 1].value == "mid":
                tempo = DEFAULT_TEMPO_INTERVALS[1].start + int(events[i + 2].value)
            elif events[i + 1].value == "fast":
                tempo = DEFAULT_TEMPO_INTERVALS[2].start + int(events[i + 2].value)
            temp_tempos.append([position, tempo])
    # get specific time for notes
    ticks_per_beat = DEFAULT_RESOLUTION
    ticks_per_bar = ticks_per_beat * beat_per_bar  # assume 4/4
    notes = []
    current_bar = 0
    for note in temp_notes:
        if note == "Bar":
            current_bar += 1
        else:
            position, velocity, pitch, duration = note
            # position (start time)
            current_bar_st = current_bar * ticks_per_bar
            current_bar_et = (current_bar + 1) * ticks_per_bar
            flags = np.linspace(
                int(current_bar_st),
                int(current_bar_et),
                int(DEFAULT_FRACTION),
                endpoint=False,
                dtype=int,
            )
            st = flags[position]
            # duration (end time)
            et = st + duration
            notes.append(miditoolkit.Note(velocity, pitch, st, et))
    # get specific time for chords
    if len(temp_chords) > 0:
        chords = []
        current_bar = 0
        for chord in temp_chords:
            if chord == "Bar":
                current_bar += 1
            else:
                position, value = chord
                # position (start time)
                current_bar_st = current_bar * ticks_per_bar
                current_bar_et = (current_bar + 1) * ticks_per_bar
                flags = np.linspace(
                    current_bar_st, current_bar_et, DEFAULT_FRACTION, endpoint=False, dtype=int
                )
                st = flags[position]
                chords.append([st, value])
    # get specific time for tempos
    tempos = []
    current_bar = 0
    for tempo in temp_tempos:
        if tempo == "Bar":
            current_bar += 1
        else:
            position, value = tempo
            # position (start time)
            current_bar_st = current_bar * ticks_per_bar
            current_bar_et = (current_bar + 1) * ticks_per_bar
            flags = np.linspace(
                current_bar_st, current_bar_et, DEFAULT_FRACTION, endpoint=False, dtype=int
            )
            st = flags[position]
            tempos.append([int(st), value])
    # write
    if prompt_path:
        midi = miditoolkit.midi.parser.MidiFile(prompt_path)
        #
        last_time = DEFAULT_RESOLUTION * 4 * 4
        # note shift
        for note in notes:
            note.start += last_time
            note.end += last_time
        midi.instruments[0].notes.extend(notes)
        # tempo changes
        temp_tempos = []
        for tempo in midi.tempo_changes:
            if tempo.time < DEFAULT_RESOLUTION * 4 * 4:
                temp_tempos.append(tempo)
            else:
                break
        for st, bpm in tempos:
            st += last_time
            temp_tempos.append(miditoolkit.midi.containers.TempoChange(bpm, st))
        midi.tempo_changes = temp_tempos
        # write chord into marker
        if len(temp_chords) > 0:
            for c in chords:
                midi.markers.append(
                    miditoolkit.midi.containers.Marker(text=c[1], time=c[0] + last_time)
                )
    else:
        midi = miditoolkit.midi.parser.MidiFile()
        midi.ticks_per_beat = DEFAULT_RESOLUTION
        # write instrument
        inst = miditoolkit.midi.containers.Instrument(0, is_drum=False)
        inst.notes = notes
        midi.instruments.append(inst)
        # write tempo
        tempo_changes = []
        for st, bpm in tempos:
            tempo_changes.append(miditoolkit.midi.containers.TempoChange(bpm, st))
        midi.tempo_changes = tempo_changes
        # write chord into marker
        if len(temp_chords) > 0:
            for c in chords:
                midi.markers.append(miditoolkit.midi.containers.Marker(text=c[1], time=c[0]))
    # write
    midi.dump(output_path)


def mk_remi_map(resolution):
    event = copy.deepcopy(constant.base_event)
    # resolution에 따른 duration과 position event 추가
    for i in range(resolution * 4):
        event.append(f"Note Duration_{i}")
    for i in range(1, resolution + 1):
        event.append(f"Position_{i}/{resolution}")

    event2word = {k: v for k, v in zip(event, range(2, len(event) + 2))}
    word2event = {v: k for k, v in zip(event, range(2, len(event) + 2))}

    return event2word, word2event


def add_flat_chord2map(event2word: Dict):
    """
    플랫 코드를 # 코드 인덱스에 할당
    ex. Ab -> G#
    """

    flat_chord = ["Chord_ab:", "Chord_bb:", "Chord_db:", "Chord_eb:", "Chord_gb:"]
    scale = ["", "maj", "maj7", "7", "dim", "dim7", "aug", "m", "m7"]

    flat_chords = []
    for c in flat_chord:
        for s in scale:
            flat_chords.append(c + s)

    for c in flat_chords:
        scale = c.split(":")[1]
        key = c.split(":")[0].split("_")[1][0]
        c = c.replace(":", "")
        if c.startswith("Chord_ab"):
            if scale == "" or scale == "maj":
                event2word[c] = event2word["Chord_g#"]
            elif scale == "maj7" or scale == "7":
                event2word[c] = event2word["Chord_g#7"]
            elif scale == "dim" or scale == "dim7":
                event2word[c] = event2word["Chord_g#dim"]
            elif scale == "aug":
                event2word[c] = event2word["Chord_g#aug"]
            elif scale == "m" or scale == "m7":
                event2word[c] = event2word["Chord_g#m"]
        else:
            if scale == "" or scale == "maj":
                new_key = chr(ord(key) - 1)
                word = "Chord_" + new_key + "#"
                event2word[c] = event2word[word]
            elif scale == "maj7" or scale == "7":
                new_key = chr(ord(key) - 1)
                word = "Chord_" + new_key + "#7"
                event2word[c] = event2word[word]
            elif scale == "dim" or scale == "dim7":
                new_key = chr(ord(key) - 1)
                word = "Chord_" + new_key + "#dim"
                event2word[c] = event2word[word]
            elif scale == "aug":
                new_key = chr(ord(key) - 1)
                word = "Chord_" + new_key + "#aug"
                event2word[c] = event2word[word]
            elif scale == "m" or scale == "m7":
                new_key = chr(ord(key) - 1)
                word = "Chord_" + new_key + "#m"
                event2word[c] = event2word[word]

    return event2word


def get_meta_message(meta_track: mido.MidiTrack, event_type: str) -> mido.MetaMessage:
    def _check_unique_meta_message(_meta_messages):
        unique_encoded_messages = set(json.dumps(vars(_m)).encode() for _m in _meta_messages)
        if len(unique_encoded_messages) != 1:
            return False
        return True

    def _get_unique_tempos(_meta_messages):
        """pretty_midi.pretty_midi L#122 _load_tempo_changes() 참조"""
        _result = []
        for _message in _meta_messages:
            if not _result:
                _result.append(_message)
            else:
                if _result[-1].tempo != _message.tempo:
                    _result.append(_message)
        return _result

    messages = [event for event in meta_track if event.type == event_type]
    if event_type == "set_tempo":
        messages = _get_unique_tempos(messages)

    if not _check_unique_meta_message(messages):
        raise InvalidMidiError(
            InvalidMidiErrorMessage.duplicate_meta.value.format(
                event_type=constant.MIDI_EVENT_LABEL[event_type]
            )
        )

    return messages.pop()


def get_time_signature(meta_message: mido.MetaMessage):
    attrs = ("numerator", "denominator")
    return [getattr(meta_message, attr) for attr in attrs]
