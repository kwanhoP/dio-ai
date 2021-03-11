import datetime
import uuid
from pathlib import Path
from typing import Any, List, Optional, Union

import note_seq
import pretty_midi
from magenta.models.score2perf.music_encoders import (
    MidiPerformanceEncoder as MagentaMidiPerformanceEncoder,
)
from tensor2tensor.data_generators import text_encoder

# https://github.com/magenta/magenta/blob/master/magenta/models/score2perf/score2perf.py#L39-L42
NUM_VELOCITY_BINS = 32
STEPS_PER_SECOND = 100
MIN_PITCH = 21
MAX_PITCH = 108


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
                ids=ids,
                strip_extraneous=strip_extraneous,
            )
            result.append(output_path)
        return result

    def decode(
        self, output_path: Union[str, Path], ids: List[int], strip_extraneous: bool = False
    ) -> str:
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

        return self.note_sequence_to_midi_file(output_path, performance.to_sequence())

    @staticmethod
    def note_sequence_to_midi_file(output_path: Union[str, Path], ns: note_seq.NoteSequence) -> str:
        output_path = output_path.expanduser()
        if ".mid" not in output_path.name:
            output_path.mkdir(exist_ok=True, parents=True)
            output_path = output_path.joinpath(f"{uuid.uuid4().hex}.mid")

        output_path = str(output_path)
        note_seq.sequence_proto_to_midi_file(ns, output_path)
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

        midi_data = pretty_midi.PrettyMIDI(s)
        # There should be only 1 instrument at preprocessed midifile
        instrument_program = midi_data.instruments[0].program
        # Should prepend token for instrument program number at the front
        return [self.num_reserved_ids + instrument_program] + self.encode_note_sequence(ns)

    def decode(self, output_path: Union[str, Path], ids: List[int], strip_extraneous=False) -> str:
        program = None
        if strip_extraneous:
            ids = text_encoder.strip_ids(ids, list(range(self.num_reserved_ids)))

        if self.use_midi_instrument and ids[0] in range(
            self._num_special_token, self.num_reserved_ids
        ):
            program = ids.pop(0) - self._num_special_token
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
            program=program,
        )
        for i in event_ids:
            performance.append(self._encoding.decode_event(i - self.num_reserved_ids))
        return self.note_sequence_to_midi_file(output_path, performance.to_sequence())

    @property
    def num_reserved_ids(self) -> int:
        return self._num_reserved_ids
