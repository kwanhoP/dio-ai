import collections
import datetime
from pathlib import Path
from typing import Any, List, Optional, Union

import mido
import note_seq

# https://github.com/magenta/magenta/blob/master/magenta/models/score2perf/score2perf.py#L39-L42
import numpy as np
import pretty_midi
from magenta.models.score2perf.music_encoders import (
    MidiPerformanceEncoder as MagentaMidiPerformanceEncoder,
)
from tensor2tensor.data_generators import text_encoder

from dioai.preprocessor.encoder.remi import utils
from dioai.preprocessor.encoder.remi.exceptions import InvalidMidiError
from dioai.preprocessor.utils import get_inst_from_info, get_ts_from_info
from dioai.preprocessor.utils.constants import (
    BPM_INTERVAL,
    MAX_PITCH,
    MIN_PITCH,
    NUM_VELOCITY_BINS,
    STEPS_PER_SECOND,
)
from dioai.preprocessor.utils.container import MidiInfo

pretty_midi.pretty_midi.MAX_TICK = 1e10

# The offset used to change the mode of a key from major to minor when
# generating a PrettyMIDI KeySignature.
_PRETTY_MIDI_MAJOR_TO_MINOR_OFFSET = 12
_START_OFFSET = 0


def encode_midi(filename: str) -> List[int]:
    encoder = MidiPerformanceEncoderWithInstrument()
    encode_seq = encoder.encode(filename)

    return encode_seq


def decode_midi(
    output_path, midi_info: MidiInfo, filename: Optional[str] = None, decoder_name="remi"
):

    if decoder_name == "remi":
        decoder = RemiEncoder(32)
    else:
        decoder = MidiPerformanceEncoderWithInstrument()
        output_path = Path(output_path)

    decoder.decode(
        output_path=output_path,
        midi_info=midi_info,
        origin_name=filename,
    )


def note_sequence_to_midi_file(midi_info: MidiInfo, sequence: note_seq.NoteSequence, output_file):
    """Convert NoteSequence to a MIDI file on disk.
    Time is stored in the NoteSequence in absolute values (seconds) as opposed to
    relative values (MIDI ticks). When the NoteSequence is translated back to
    MIDI the absolute time is retained. The tempo map is also recreated.

    Args:
        sequence: A NoteSequence.
        output_file: String path to MIDI file that will be written.
    """
    pretty_midi_object = note_sequence_to_pretty_midi(midi_info, sequence)
    pretty_midi_object.write(open(output_file, "wb"))


def note_sequence_to_pretty_midi(midi_info: MidiInfo, sequence: note_seq.NoteSequence):
    """Convert NoteSequence to a PrettyMIDI.

    Time is stored in the NoteSequence in absolute values (seconds) as opposed to
    relative values (MIDI ticks). When the NoteSequence is translated back to
    PrettyMIDI the absolute time is retained. The tempo map is also recreated.

    TODO: absolute한 시간을 바탕으로 저장되기 때문에 4마디, 8마디에 정확히 맞춰주는 PostProcessor 필요
    -> 데이터 생성시에 BPM_INTERVAL단위로 고정하여 임시 해결
    마디에 정확히 맞게 나오게 하기 위해선 정확한 bpm 정보를 넣어야함.
    예를 들면 64bpm에서는 4마디에 딱 맞지만 60bpm에서는 3마디임.
    현재 인코딩 방식이 60~64를 같은 BPM으로 보기 때문에 마디가 맞지 않는 문제가 발생함.


    Args:
        sequence: A NoteSequence.
    Returns:
        A pretty_midi.PrettyMIDI object or None if sequence could not be decoded.
    """

    # Try to find a tempo at time zero. The list is not guaranteed to be in order.
    tempo = midi_info.bpm * BPM_INTERVAL

    pm = pretty_midi.PrettyMIDI(initial_tempo=tempo)

    # Create an empty instrument to contain time and key signatures.
    program = get_inst_from_info(midi_info.inst)
    inst = pretty_midi.Instrument(program)
    pm.instruments.append(inst)

    numerator, denominator = get_ts_from_info(midi_info.time_signature)
    time_signature = pretty_midi.containers.TimeSignature(numerator, denominator, _START_OFFSET)
    pm.time_signature_changes.append(time_signature)

    key_signature = pretty_midi.containers.KeySignature(midi_info.audio_key, _START_OFFSET)
    pm.key_signature_changes.append(key_signature)

    # Populate instrument events by first gathering notes and other event types
    # in lists then write them sorted to the PrettyMidi object.
    instrument_events = collections.defaultdict(lambda: collections.defaultdict(list))
    for seq_note in sequence.notes:
        instrument_events[(seq_note.instrument, program, seq_note.is_drum)]["notes"].append(
            pretty_midi.Note(
                seq_note.velocity, seq_note.pitch, seq_note.start_time, seq_note.end_time
            )
        )
    for seq_bend in sequence.pitch_bends:
        instrument_events[(seq_bend.instrument, program, seq_bend.is_drum)]["bends"].append(
            pretty_midi.PitchBend(seq_bend.bend, seq_bend.time)
        )

    for seq_cc in sequence.control_changes:
        instrument_events[(seq_cc.instrument, program, seq_cc.is_drum)]["controls"].append(
            pretty_midi.ControlChange(seq_cc.control_number, seq_cc.control_value, seq_cc.time)
        )

    for (inst_id, prog_id, is_drum) in sorted(instrument_events.keys()):
        # For instr_id 0 append to the instrument created above.
        if inst_id > 0:
            instrument = pretty_midi.Instrument(prog_id, is_drum)
            pm.instruments.append(instrument)
        else:
            inst.is_drum = is_drum
        # propagate instrument name to the midi file
        inst.program = prog_id

        inst.notes = instrument_events[(inst_id, prog_id, is_drum)]["notes"]
        inst.pitch_bends = instrument_events[(inst_id, prog_id, is_drum)]["bends"]
        inst.control_changes = instrument_events[(inst_id, prog_id, is_drum)]["controls"]

    return pm


class MidiPerformanceEncoder(MagentaMidiPerformanceEncoder):
    def __init__(
        self,
        steps_per_second: int = STEPS_PER_SECOND,
        num_velocity_bins: int = NUM_VELOCITY_BINS,
        min_pitch: int = MIN_PITCH,
        max_pitch: int = MAX_PITCH,
        add_eos: bool = True,
        ngrams: Optional[List[Any]] = None,
    ):
        super().__init__(
            steps_per_second=steps_per_second,
            num_velocity_bins=num_velocity_bins,
            min_pitch=min_pitch,
            max_pitch=max_pitch,
            add_eos=add_eos,
            ngrams=ngrams,
        )

    def decode_batch(
        self,
        output_dir: Union[str, Path],
        batch_ids: List[List[int]],
        strip_extraneous: bool = False,
    ) -> List[str]:
        date = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = output_dir.joinpath(date)
        output_dir.mkdir(exist_ok=True, parents=True)

        result = []
        for i, ids in enumerate(batch_ids):
            output_path = self.decode(
                output_path=output_dir.joinpath(f"{i:03d}_decoded.mid"),
                midi_info=ids,
                strip_extraneous=strip_extraneous,
            )
            result.append(output_path)
        return result

    def decode(
        self,
        output_path: Union[str, Path],
        midi_info: MidiInfo,
        origin_name: str,
        strip_extraneous: bool = False,
    ) -> str:

        ids = midi_info.note_seq

        if strip_extraneous:
            ids = text_encoder.strip_ids(ids, list(range(self.num_reserved_ids)))

        # Decode indices corresponding to event n-grams back into the n-grams.
        event_ids = []
        for i in ids:
            if i >= self.unigram_vocab_size:
                event_ids += self._ngrams[i - self.unigram_vocab_size]
            else:
                event_ids.append(i)

        performance = note_seq.Performance(
            quantized_sequence=None,
            steps_per_second=self._steps_per_second,
            num_velocity_bins=self._num_velocity_bins,
        )
        for i in event_ids:
            performance.append(self._encoding.decode_event(i - self.num_reserved_ids))

        return self.note_sequence_to_midi_file(
            output_path, midi_info, performance.to_sequence(), origin_name
        )

    @staticmethod
    def note_sequence_to_midi_file(
        output_path: Union[str, Path],
        midi_info: MidiInfo,
        ns: note_seq.NoteSequence,
        origin_name: Optional[str] = None,
    ) -> str:
        output_path = output_path.expanduser()
        if ".mid" not in output_path.name:
            output_path.mkdir(exist_ok=True, parents=True)
            output_path = output_path.joinpath(f"{origin_name}.mid")

        output_path = str(output_path)
        note_sequence_to_midi_file(midi_info, ns, output_path)
        return output_path


# https://github.com/magenta/magenta/blob/master/magenta/models/score2perf/music_encoders.py#L33
class MidiPerformanceEncoderWithInstrument(MidiPerformanceEncoder):
    def __init__(
        self,
        steps_per_second: int = STEPS_PER_SECOND,
        num_velocity_bins: int = NUM_VELOCITY_BINS,
        min_pitch: int = MIN_PITCH,
        max_pitch: int = MAX_PITCH,
        add_eos: bool = True,
        ngrams: Optional[List[Any]] = None,
        use_midi_instrument: bool = False,
        # midi instrument program number: 0 ~ 127
        num_midi_instrument: int = 128,
    ):
        self._num_special_token = text_encoder.NUM_RESERVED_TOKENS
        self._num_reserved_ids = text_encoder.NUM_RESERVED_TOKENS
        self.use_midi_instrument = use_midi_instrument
        if self.use_midi_instrument:
            self._num_reserved_ids += num_midi_instrument
        super().__init__(
            steps_per_second=steps_per_second,
            num_velocity_bins=num_velocity_bins,
            min_pitch=min_pitch,
            max_pitch=max_pitch,
            add_eos=add_eos,
            ngrams=ngrams,
        )

    def encode(self, s: Optional[str] = None) -> List[int]:
        if s is not None:
            ns = note_seq.midi_file_to_sequence_proto(s)
        else:
            ns = note_seq.NoteSequence()

        return self.encode_note_sequence(ns)

    def decode(
        self,
        output_path: Union[str, Path],
        midi_info: MidiInfo,
        origin_name: str,
        strip_extraneous=False,
    ) -> str:
        ids = midi_info.note_seq

        if strip_extraneous:
            ids = text_encoder.strip_ids(ids, list(range(self.num_reserved_ids)))

        # Decode indices corresponding to event n-grams back into the n-grams.
        event_ids = []
        for i in ids:
            if i >= self.unigram_vocab_size:
                try:
                    event_ids += self._ngrams[i - self.unigram_vocab_size]
                except IndexError:
                    continue
            else:
                event_ids.append(i)

        performance = note_seq.Performance(
            quantized_sequence=None,
            steps_per_second=self._steps_per_second,
            num_velocity_bins=self._num_velocity_bins,
        )
        for i in event_ids:
            if i == 1 or i == 0:  # EOS and PAD
                continue
            performance.append(self._encoding.decode_event(i - self.num_reserved_ids))

        new_performance = note_seq.Performance(
            quantized_sequence=None,
            steps_per_second=self._steps_per_second,
            num_velocity_bins=self._num_velocity_bins,
        )
        note_found = False
        for event in performance:
            if not note_found and event.event_type == note_seq.PerformanceEvent.TIME_SHIFT:
                continue

            if event.event_type != note_seq.PerformanceEvent.TIME_SHIFT:
                note_found = True
            new_performance.append(event)

        return self.note_sequence_to_midi_file(
            output_path, midi_info, new_performance.to_sequence(), origin_name
        )

    @property
    def num_reserved_ids(self) -> int:
        return self._num_reserved_ids


REMI_EOS_TOKEN = 1


class RemiEncoder:
    name = "remi"

    def __init__(self, resolution):
        self.event2word, self.word2event = utils.mk_remi_map(resolution)
        self.event2word = utils.add_flat_chord2map(self.event2word)
        self.position_resolution = resolution
        default_tick_per_bar = 3840
        self.duration_bins = np.arange(
            int(default_tick_per_bar / 4 / resolution),
            3841,
            int(default_tick_per_bar / 4 / resolution),
            dtype=int,
        )

    def encode(self, midi_paths, sample_info=None):
        mido_file = mido.MidiFile(midi_paths)
        mido_meta = mido_file.tracks[0]
        tick_per_beat = mido_file.ticks_per_beat
        if sample_info is not None:
            chord_progression = sample_info["chord_progressions"]
            audio_key = sample_info["audio_key"]
            numerator = int(sample_info["time_signature"].split("/")[0])
            denominator = int(sample_info["time_signature"].split("/")[1])
        else:
            numerator, denominator = utils.get_time_signature(
                utils.get_meta_message(mido_meta, "time_signature")
            )
            try:
                audio_key = getattr(utils.get_meta_message(mido_meta, "key_signature"), "key")
            except InvalidMidiError:  # key 정보가 없는 경우
                audio_key = "C"
        if denominator == 8:
            numerator = numerator / 2
        tick_per_bar = tick_per_beat * numerator
        events = utils.extract_events(
            midi_paths,
            self.position_resolution,
            self.duration_bins,
            tick_per_bar=tick_per_bar,
            tick_per_beat=tick_per_beat,
            chord_progression=None if sample_info is None else chord_progression,
            audio_key=audio_key,
            use_backoffice_chord=False if sample_info is None else True,
        )

        words = []
        for event in events:
            e = "{}_{}".format(event.name, event.value)
            if e in self.event2word:
                words.append(self.event2word[e])
            else:
                # OOV
                if event.name == "Note Velocity":
                    # replace with max velocity based on our training data
                    words.append(self.event2word["Note Velocity_31"])
                else:
                    # something is wrong
                    # you should handle it for your own purpose
                    print("OOV {}".format(e))
        words.append(REMI_EOS_TOKEN)  # eos token
        return np.array(words)

    def decode(self, output_path, midi_info, origin_name=None):
        utils.write_midi(
            midi_info.note_seq,
            self.word2event,
            output_path,
            DEFAULT_FRACTION=self.position_resolution,
            DEFAULT_DURATION_BINS=self.duration_bins,
        )
