import collections
import enum
import math
from dataclasses import astuple, dataclass
from pathlib import Path
from typing import Any, List, Optional, Union

import attr
import note_seq
from absl import logging
from magenta.models.score2perf.music_encoders import (
    MidiPerformanceEncoder as _MidiPerformanceEncoder,
)
from note_seq import Performance as _Performance
from note_seq import PerformanceEvent as _PerformanceEvent
from note_seq import PerformanceOneHotEncoding as _PerformanceOneHotEncoding
from note_seq.performance_lib import (
    _program_and_is_drum_from_sequence,
    velocity_bin_to_velocity,
    velocity_to_bin,
)
from note_seq.protobuf import music_pb2
from note_seq.sequences_lib import assert_is_absolute_quantized_sequence
from tensor2tensor.data_generators import text_encoder

from dioai.preprocessor.encoder.encoder import note_sequence_to_midi_file
from dioai.preprocessor.utils.constants import (
    CONTROL_CHANGE_DICT,
    DEFAULT_INSTRUMENT,
    DEFAULT_PROGRAM,
    MAX_PITCH,
    MIN_PITCH,
    NUM_EXPRESSION_BINS,
    NUM_MODULATION_BINS,
    NUM_SUSTAIN_BINS,
    NUM_VELOCITY_BINS,
    STANDARD_PPQ,
    STEPS_PER_SECOND,
    SUSTAIN_OFF,
    SUSTAIN_ON,
    SUSTAIN_THRESHOLD,
)
from dioai.preprocessor.utils.container import MidiInfo

MIN_CONTROL_CHANGE = 1
MAX_CONTROL_CHANGE = 127


def _control_change_bin_size(num_control_change_bins: int):
    return int(math.ceil((MAX_CONTROL_CHANGE - MIN_CONTROL_CHANGE + 1) / num_control_change_bins))


def control_change_to_bin(control_value: int, num_control_change_bins: int):
    return int(
        abs(control_value - MIN_CONTROL_CHANGE) // _control_change_bin_size(num_control_change_bins)
        + 1
    )


def control_change_bin_to_control_change(control_change_bin: int, num_control_change_bins: int):
    return int(
        MIN_CONTROL_CHANGE
        + (control_change_bin - 1) * _control_change_bin_size(num_control_change_bins)
    )


@dataclass
class EventSortingOrder:
    step: int
    event_index: int
    event_priority: int
    event: object


@attr.s(frozen=True)
class DioPerformanceEvent(_PerformanceEvent):
    """Class for storing events in a performance."""

    event_type = attr.ib()
    event_value = attr.ib()

    MODULATION = 6
    EXPRESSION = 7
    SUSTAIN = 8

    @event_type.validator
    def _check_event(self, attribute, value):
        try:
            super()._check_event(attribute, value)
        except ValueError:
            if self.event_type == DioPerformanceEvent.MODULATION:
                if not 1 <= self.event_value <= NUM_MODULATION_BINS:
                    raise ValueError("Invalid modulation value: %s" % self.event_value)
            elif self.event_type == DioPerformanceEvent.EXPRESSION:
                if not 1 <= self.event_value <= NUM_EXPRESSION_BINS:
                    raise ValueError("Invalid expression value: %s" % self.event_value)
            elif self.event_type == DioPerformanceEvent.SUSTAIN:
                if not 1 <= self.event_value <= NUM_SUSTAIN_BINS:
                    raise ValueError("Invalid sustain value: %s" % self.event_value)
            else:
                raise ValueError("Invalid event type: %s" % self.event_type)


@enum.unique
class NoteEvents(int, enum.Enum):
    NOTE_ON = DioPerformanceEvent.NOTE_ON
    NOTE_OFF = DioPerformanceEvent.NOTE_OFF

    @classmethod
    def get_types(cls):
        return (cls.NOTE_ON.value, cls.NOTE_OFF.value)


@enum.unique
class ControlChangeEvents(int, enum.Enum):
    MODULATION = DioPerformanceEvent.MODULATION
    EXPRESSION = DioPerformanceEvent.EXPRESSION
    SUSTAIN = DioPerformanceEvent.SUSTAIN

    @classmethod
    def get_types(cls):
        return (cls.MODULATION.value, cls.EXPRESSION.value, cls.SUSTAIN.value)


@dataclass
class EventPriorityChecker:
    MODULATION: int = DioPerformanceEvent.MODULATION
    EXPRESSION: int = DioPerformanceEvent.EXPRESSION
    SUSTAIN: int = DioPerformanceEvent.SUSTAIN
    NOTE_ON: int = DioPerformanceEvent.NOTE_ON
    NOTE_OFF: int = DioPerformanceEvent.NOTE_OFF

    def on_type(self, event_type):
        return astuple(self).index(event_type)

    def get_type_from_priority(self, priority):
        return astuple(self)[priority]


class DioPerformance(_Performance):
    def __init__(
        self,
        quantized_sequence,
        steps_per_second: Optional[int] = None,
        start_step: int = 0,
        num_velocity_bins: int = NUM_VELOCITY_BINS,
        max_shift_steps: int = STEPS_PER_SECOND,
        instrument: Optional[int] = None,
        program: Optional[int] = None,
        is_drum: Optional[bool] = None,
        num_modulation_bins: int = NUM_MODULATION_BINS,
        num_expression_bins: int = NUM_EXPRESSION_BINS,
        num_sustain_bins: int = NUM_SUSTAIN_BINS,
        use_control_changes: bool = True,
    ):
        if (quantized_sequence, steps_per_second).count(None) != 1:
            raise ValueError("Must specify exactly one of quantized_sequence or steps_per_second")

        if quantized_sequence:
            assert_is_absolute_quantized_sequence(quantized_sequence)
            self._steps_per_second = quantized_sequence.quantization_info.steps_per_second
            self._events = self._from_quantized_sequence(
                quantized_sequence=quantized_sequence,
                start_step=start_step,
                max_shift_steps=max_shift_steps,
                num_velocity_bins=num_velocity_bins,
                instrument=instrument,
                num_modulation_bins=num_modulation_bins,
                num_expression_bins=num_expression_bins,
                num_sustain_bins=num_sustain_bins,
                use_control_changes=use_control_changes,
            )
            program, is_drum = _program_and_is_drum_from_sequence(quantized_sequence, instrument)
        else:
            self._steps_per_second = steps_per_second
            self._events = []

        self._start_step = start_step
        self._num_velocity_bins = num_velocity_bins
        self._max_shift_steps = max_shift_steps
        self._program = program
        self._is_drum = is_drum
        self._num_modulation_bins = num_modulation_bins
        self._num_expression_bins = num_expression_bins
        self._num_sustain_bins = num_sustain_bins
        self._use_control_changes = use_control_changes

    def __str__(self):
        strs = []
        for event in self:
            if event.event_type == DioPerformanceEvent.NOTE_ON:
                strs.append("(%s, ON)" % event.event_value)
            elif event.event_type == DioPerformanceEvent.NOTE_OFF:
                strs.append("(%s, OFF)" % event.event_value)
            elif event.event_type == DioPerformanceEvent.TIME_SHIFT:
                strs.append("(%s, SHIFT)" % event.event_value)
            elif event.event_type == DioPerformanceEvent.VELOCITY:
                strs.append("(%s, VELOCITY)" % event.event_value)
            elif event.event_type == DioPerformanceEvent.MODULATION:
                strs.append("(%s, MODULATION)" % event.event_value)
            elif event.event_type == DioPerformanceEvent.EXPRESSION:
                strs.append("(%s, EXPRESSION)" % event.event_value)
            elif event.event_type == DioPerformanceEvent.SUSTAIN:
                strs.append("(%s, SUSTAIN)" % event.event_value)
            else:
                raise ValueError("Unknown event type: %s" % event.event_type)
        return "\n".join(strs)

    @staticmethod
    def _from_quantized_sequence(
        quantized_sequence,
        start_step,
        num_velocity_bins,
        max_shift_steps,
        num_modulation_bins,
        num_expression_bins,
        num_sustain_bins,
        instrument=None,
        use_control_changes=True,
    ):
        """사용된 악기가 1개 이하인 경우만 고려합니다."""

        def _get_sorting_order(event: object, event_type: int, event_idx: int = 0):
            if event_type == DioPerformanceEvent.NOTE_ON:
                step = event.quantized_start_step
            elif event_type == DioPerformanceEvent.NOTE_OFF:
                step = event.quantized_end_step
            elif event_type in ControlChangeEvents.get_types():
                step = event.quantized_step
            else:
                raise ValueError("Unknown event type: %s" % event_type)
            event_priority = EventPriorityChecker().on_type(event_type=event_type)
            return EventSortingOrder(step, event_idx, event_priority, event)

        notes = [
            note
            for note in quantized_sequence.notes
            if note.quantized_start_step >= start_step
            and (instrument is None or note.instrument == instrument)
        ]
        sorted_notes = sorted(notes, key=lambda note: (note.start_time, note.pitch))
        events = []  # 모든 NoteSequence.notes 이벤트와 NoteSequence.control_changes 이벤트를 저장합니다.
        for idx, note in enumerate(sorted_notes):
            events.append(
                _get_sorting_order(note, event_type=DioPerformanceEvent.NOTE_ON, event_idx=idx)
            )
            events.append(
                _get_sorting_order(note, event_type=DioPerformanceEvent.NOTE_OFF, event_idx=idx)
            )
        if use_control_changes:
            for control_name, event_type in ControlChangeEvents.__members__.items():
                control_name = str(control_name).lower()
                control_number = CONTROL_CHANGE_DICT[control_name]
                ccs = [
                    cc
                    for cc in quantized_sequence.control_changes
                    if cc.control_number == control_number and cc.quantized_step >= start_step
                ]
                sorted_ccs = sorted(ccs, key=lambda cc: cc.time)
                events += [_get_sorting_order(cc, event_type=event_type) for cc in sorted_ccs]
        events = sorted([astuple(event) for event in events], key=lambda tup: tup[:-1])

        current_step = start_step
        current_velocity_bin = 0
        performance_events = []
        priority_checker = EventPriorityChecker()
        for step, _, event_priority, event in events:
            event_type = priority_checker.get_type_from_priority(priority=event_priority)
            if step > current_step:
                # Shift time forward from the current step to this event.
                while step > current_step + max_shift_steps:
                    # We need to move further than the maximum shift size.
                    performance_events.append(
                        DioPerformanceEvent(
                            event_type=DioPerformanceEvent.TIME_SHIFT, event_value=max_shift_steps
                        )
                    )
                    current_step += max_shift_steps
                performance_events.append(
                    DioPerformanceEvent(
                        event_type=DioPerformanceEvent.TIME_SHIFT,
                        event_value=int(step - current_step),
                    )
                )
                current_step = step

            # control change 이벤트가 있다면 DioPerformanceEvent 형식으로 저장합니다.
            if event_type in ControlChangeEvents.get_types():
                if num_modulation_bins and event_type == DioPerformanceEvent.MODULATION:
                    num_control_change_bins = num_modulation_bins
                elif num_expression_bins and event_type == DioPerformanceEvent.EXPRESSION:
                    num_control_change_bins = num_expression_bins
                elif num_sustain_bins and event_type == DioPerformanceEvent.SUSTAIN:
                    num_control_change_bins = num_sustain_bins
                else:
                    raise ValueError("Unknown control change event type: %s" % event.event_type)
                control_change_bin = control_change_to_bin(
                    event.control_value, num_control_change_bins
                )
                performance_events.append(
                    DioPerformanceEvent(event_type=event_type, event_value=control_change_bin)
                )

            # If we're using velocity and this note's velocity is different from the
            # current velocity, change the current velocity.
            if num_velocity_bins and event_type == DioPerformanceEvent.NOTE_ON:
                velocity_bin = velocity_to_bin(event.velocity, num_velocity_bins)
                if velocity_bin != current_velocity_bin:
                    current_velocity_bin = velocity_bin
                    performance_events.append(
                        DioPerformanceEvent(
                            event_type=DioPerformanceEvent.VELOCITY,
                            event_value=current_velocity_bin,
                        )
                    )

            # Add a performance event for this note on/off.
            if event_type in NoteEvents.get_types():
                performance_events.append(
                    DioPerformanceEvent(event_type=event_type, event_value=event.pitch)
                )
        return performance_events

    def _to_sequence(
        self,
        velocity: Optional[int] = None,
        seconds_per_step: float = 1.0 / STEPS_PER_SECOND,
        instrument: Optional[int] = DEFAULT_INSTRUMENT,
        program: Optional[int] = None,
        max_note_duration: Optional[float] = None,
    ):
        sequence_start_time = self._start_step * seconds_per_step
        sequence = music_pb2.NoteSequence()
        sequence.ticks_per_quarter = STANDARD_PPQ

        if program is None:
            program = self._program if self._program is not None else DEFAULT_PROGRAM
            is_drum = self._is_drum if self._is_drum is not None else False

        step = 0
        # Map pitch to list because one pitch may be active multiple times.
        pitch_start_steps_and_velocities = collections.defaultdict(list)
        for i, event in enumerate(self):
            if event.event_type == DioPerformanceEvent.NOTE_ON:
                pitch_start_steps_and_velocities[event.event_value].append((step, velocity))
            elif event.event_type == DioPerformanceEvent.NOTE_OFF:
                if not pitch_start_steps_and_velocities[event.event_value]:
                    logging.debug("Ignoring NOTE_OFF at position %d with no previous NOTE_ON", i)
                else:
                    # Create a note for the pitch that is now ending.
                    pitch_start_step, pitch_velocity = pitch_start_steps_and_velocities[
                        event.event_value
                    ][0]
                    pitch_start_steps_and_velocities[
                        event.event_value
                    ] = pitch_start_steps_and_velocities[event.event_value][1:]
                    if step == pitch_start_step:
                        logging.debug("Ignoring note with zero duration at step %d", step)
                        continue
                    note = sequence.notes.add()
                    note.start_time = pitch_start_step * seconds_per_step + sequence_start_time
                    note.end_time = step * seconds_per_step + sequence_start_time
                    if max_note_duration and note.end_time - note.start_time > max_note_duration:
                        note.end_time = note.start_time + max_note_duration
                    note.pitch = event.event_value
                    note.velocity = pitch_velocity
                    note.instrument = instrument
                    note.program = program
                    note.is_drum = is_drum
                    if note.end_time > sequence.total_time:
                        sequence.total_time = note.end_time
            elif event.event_type == DioPerformanceEvent.TIME_SHIFT:
                step += event.event_value
            elif event.event_type == DioPerformanceEvent.VELOCITY:
                assert self._num_velocity_bins
                velocity = velocity_bin_to_velocity(
                    event.event_value, num_velocity_bins=self._num_velocity_bins
                )
            # control change 이벤트가 있다면 NoteSequence 에 추가합니다.
            elif event.event_type in ControlChangeEvents.get_types():
                assert self._use_control_changes
                cc = sequence.control_changes.add()
                cc.time = step * seconds_per_step + sequence_start_time
                if event.event_type == DioPerformanceEvent.MODULATION:
                    cc.control_number = CONTROL_CHANGE_DICT["modulation"]
                    cc.control_value = control_change_bin_to_control_change(
                        event.event_value, num_control_change_bins=self._num_modulation_bins
                    )
                elif event.event_type == DioPerformanceEvent.EXPRESSION:
                    cc.control_number = CONTROL_CHANGE_DICT["expression"]
                    cc.control_value = control_change_bin_to_control_change(
                        event.event_value, num_control_change_bins=self._num_expression_bins
                    )
                elif event.event_type == DioPerformanceEvent.SUSTAIN:
                    cc.control_number = CONTROL_CHANGE_DICT["sustain"]
                    _sustain = control_change_bin_to_control_change(
                        event.event_value, num_control_change_bins=self._num_sustain_bins
                    )
                    cc.control_value = SUSTAIN_OFF if _sustain < SUSTAIN_THRESHOLD else SUSTAIN_ON
                else:
                    raise ValueError("Unknown control change event type: %s" % event.event_type)
            else:
                raise ValueError("Unknown event type: %s" % event.event_type)

        # There could be remaining pitches that were never ended. End them now and create notes.
        for pitch in pitch_start_steps_and_velocities:
            for pitch_start_step, pitch_velocity in pitch_start_steps_and_velocities[pitch]:
                if step == pitch_start_step:
                    logging.debug("Ignoring note with zero duration at step %d", step)
                    continue
                note = sequence.notes.add()
                note.start_time = pitch_start_step * seconds_per_step + sequence_start_time
                note.end_time = step * seconds_per_step + sequence_start_time
                if max_note_duration and note.end_time - note.start_time > max_note_duration:
                    note.end_time = note.start_time + max_note_duration
                note.pitch = pitch
                note.velocity = pitch_velocity
                note.instrument = instrument
                note.program = program
                note.is_drum = is_drum
                if note.end_time > sequence.total_time:
                    sequence.total_time = note.end_time

        return sequence


class DioPerformanceOneHotEncoding(_PerformanceOneHotEncoding):
    def __init__(
        self,
        num_velocity_bins: int = NUM_VELOCITY_BINS,
        max_shift_steps: int = STEPS_PER_SECOND,
        min_pitch: int = MIN_PITCH,
        max_pitch: int = MAX_PITCH,
        num_modulation_bins: int = NUM_MODULATION_BINS,
        num_expression_bins: int = NUM_EXPRESSION_BINS,
        num_sustain_bins: int = NUM_SUSTAIN_BINS,
        use_control_changes: bool = True,
    ):
        super(DioPerformanceOneHotEncoding, self).__init__(
            num_velocity_bins=num_velocity_bins,
            max_shift_steps=max_shift_steps,
            min_pitch=min_pitch,
            max_pitch=max_pitch,
        )
        if use_control_changes:
            _control_changes_ranges = [
                (DioPerformanceEvent.MODULATION, 1, num_modulation_bins),
                (DioPerformanceEvent.EXPRESSION, 1, num_expression_bins),
                (DioPerformanceEvent.SUSTAIN, 1, num_sustain_bins),
            ]
            self._event_ranges += _control_changes_ranges

    def encode_event(self, event) -> int:
        offset = 0
        for event_type, min_value, max_value in self._event_ranges:
            if event.event_type == event_type:
                return offset + event.event_value - min_value
            offset += max_value - min_value + 1
        raise ValueError("Unknown event type: %s" % event.event_type)

    def decode_event(self, index) -> DioPerformanceEvent:
        offset = 0
        for event_type, min_value, max_value in self._event_ranges:
            if offset <= index <= offset + max_value - min_value:
                return DioPerformanceEvent(
                    event_type=event_type, event_value=min_value + index - offset
                )
            offset += max_value - min_value + 1
        raise ValueError("Unknown event index: %s" % index)


class DioMidiPerformanceEncoder(_MidiPerformanceEncoder):
    def __init__(
        self,
        steps_per_second: int = STEPS_PER_SECOND,
        num_velocity_bins: int = NUM_VELOCITY_BINS,
        min_pitch: int = MIN_PITCH,
        max_pitch: int = MAX_PITCH,
        add_eos: bool = True,
        ngrams: Optional[List[Any]] = None,
        use_control_changes: bool = True,
    ):
        super(DioMidiPerformanceEncoder, self).__init__(
            steps_per_second=steps_per_second,
            num_velocity_bins=num_velocity_bins,
            min_pitch=min_pitch,
            max_pitch=max_pitch,
            add_eos=add_eos,
            ngrams=ngrams,
        )
        self._encoding = DioPerformanceOneHotEncoding(use_control_changes=use_control_changes)
        self._use_control_changes = use_control_changes

    def encode_note_sequence(self, ns):
        performance = DioPerformance(
            note_seq.quantize_note_sequence_absolute(ns, self._steps_per_second),
            use_control_changes=self._use_control_changes,
        )
        event_ids = [
            self._encoding.encode_event(event) + self.num_reserved_ids for event in performance
        ]
        if self._add_eos:
            event_ids.append(text_encoder.EOS_ID)

        return event_ids

    def decode_to_note_sequence(
        self,
        ids: List[int],
        strip_extraneous: bool = True,
        remove_front_time_shift: bool = True,
    ):
        if strip_extraneous:
            ids = text_encoder.strip_ids(ids, list(range(self.num_reserved_ids)))

        # Decode indices corresponding to event n-grams back into the n-grams.
        event_ids = []
        for i in ids:
            if i >= self.unigram_vocab_size:
                event_ids += self._ngrams[i - self.unigram_vocab_size]
            else:
                event_ids.append(i)

        performance = DioPerformance(
            quantized_sequence=None,
            steps_per_second=self._steps_per_second,
            use_control_changes=self._use_control_changes,
        )
        # dioai.preprocessor.encoder.encoder.MidiPerformanceEncoderWithInstrument
        note_found = False
        for i in event_ids:
            event = self._encoding.decode_event(i - self.num_reserved_ids)

            if remove_front_time_shift:
                if not note_found and event.event_type == DioPerformanceEvent.TIME_SHIFT:
                    continue
                if event.event_type != DioPerformanceEvent.TIME_SHIFT:
                    note_found = True
            performance.append(event)

        ns = performance._to_sequence()
        return ns

    def encode(self, s: Union[str, Path]) -> List[int]:
        if s:
            ns = note_seq.midi_file_to_sequence_proto(str(s))
        else:
            ns = note_seq.NoteSequence()
        return self.encode_note_sequence(ns)

    def decode(
        self,
        ids: List[int],
        midi_info: MidiInfo,
        output_file: Union[str, Path],
        strip_extraneous: bool = True,
        remove_front_time_shift: bool = True,
    ) -> None:
        ns = self.decode_to_note_sequence(
            ids,
            strip_extraneous=strip_extraneous,
            remove_front_time_shift=remove_front_time_shift,
        )
        note_sequence_to_midi_file(midi_info=midi_info, sequence=ns, output_file=str(output_file))
