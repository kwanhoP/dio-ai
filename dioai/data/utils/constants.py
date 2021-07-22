from dataclasses import dataclass

META_OFFSET = 421
NOTE_SEQ_OFFSET = 2

NOTE_SEQ_COMPONENTS = {"note_on": 128, "note_off": 128, "time_shift": 100, "velocity": 64}


@dataclass
class RagVocab:
    """
    RAG, DPR, BERT에 대한 meta, note 통합 vocab
    """

    pad_id = 0
    eos_id = 1
    note_start_id = 2
    meta_end_id = 637
    sos_id = 638
    mask_id = 639
    sep_id = 640


@dataclass
class DPRVocab:
    pad_id = 0
    eos_id = 1
    sos_id = 2
    note_shift = 1
    meta_shift = 420


@dataclass
class Meta:
    bpm = 1
    key = 2
    time_sig = 3
    pitch_range = 4
    num_measure = 5
    inst = 6
    genre = 7
