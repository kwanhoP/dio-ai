from dataclasses import dataclass

META_OFFSET = 421
NOTE_SEQ_OFFSET = 2

NOTE_SEQ_COMPONENTS = {"note_on": 128, "note_off": 128, "time_shift": 100, "velocity": 64}


@dataclass
class BertVocab:
    """
    meta, note에 대한 Bert Vocab
    meta vocab size: 219
    note vocab size: 424
    DPR 구축에 활용, meta(query), note(context)
    """

    NOTE_VOCAB = {"pad_id": 0, "eos_id": 1, "sos_id": 2, "mask_id": 3, "note_seq_shift": 2}

    META_VOCAB = {"pad_id": 0, "sos_id": 1, "mask_id": 2, "meta_seq_shift": 419}


@dataclass
class DPRVocab:
    pad_id = 0
    eos_id = 1
    sos_id = 2
    note_shift = 1
    meta_shift = 420
