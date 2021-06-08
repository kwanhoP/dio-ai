import enum
import multiprocessing
from typing import Optional


class DefaultValue(enum.Enum):
    BPM = 120
    AUDIO_KEY = "cmajor"
    TIME_SIGNATURE = "4/4"


class KeySwitchVelocity(int, enum.Enum):
    DEFAULT = 1

    @classmethod
    def get_value(cls, key: Optional[str]) -> int:
        key = key or "DEFAULT"
        if hasattr(cls, key):
            return getattr(cls, key).value
        return cls.DEFAULT.value


NUM_KEY_AUGMENT = 6
NUM_BPM_AUGMENT = 2

DRUM_CHANNEL = 9
CHORD_CHANNEL = 5

MIDI_EXTENSIONS = (".mid", ".MID", ".midi", ".MIDI")
DEFAULT_NUM_BEATS = 4
CHORD_TRACK_NAME = "chord"
PITCH_RANGE_CUT = {
    "very_low": 36,
    "low": 48,
    "mid_low": 60,
    "mid": 72,
    "mid_high": 84,
    "high": 96,
    "very_high": 128,
}
DEFAULT_PITCH_RANGE = "mid"
CHORD_TYPE_IDX = -1
MINOR_KEY = "m"
NO_META_MESSAGE = "no_info"
UNKNOWN = "unknown"

NUM_VELOCITY_BINS = 64
STEPS_PER_SECOND = 100
MIN_PITCH = 0
MAX_PITCH = 127
BPM_INTERVAL = 5
VELOCITY_INTERVAL = 5
MAX_BPM = 200

DEFAULT_BPM = 24
DEFAULT_KEY = 0
DEFAULT_TS = 4

BPM_START_POINT = 423
KEY_START_POINT = 464
TS_START_POINT = 489
PITCH_RANGE_START_POINT = 507
MEASURES_4 = 514
MEASURES_8 = 515
INST_START_POINT = 517
GENRE_START_POINT = 526
VELOCITY_START_POINT = 540
TRACK_CATEGORY_START_POINT = 568

BPM_UNKHOWN = 422
KEY_UNKHOWN = 463
TS_UNKHOWN = 488
PITCH_RANGE_UNKHOWN = 506
INST_UNKHOWN = 516
GENRE_UNKNOWN = 525
VELOCITY_UNKNOWN = 539
TRACK_CATEGORY_UNKNOWN = 567

NUM_CORES = multiprocessing.cpu_count() - 10  # core 전부 다 쓰면 병목현상 보임

PRETTY_MAJOR_KEY = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
PRETTY_MINOR_KEY = [12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]
MINOR_KEY_OFFSET = 12

TIME_SIG_MAP = {
    "3/2": 0,
    "6/4": 0,
    "4/2": 1,
    "8/4": 1,
    "6/3": 1,
    "1/2": 2,
    "4/8": 2,
    "2/4": 2,
    "3/4": 3,
    "4/4": 4,
    "8/8": 4,
    "2/2": 4,
    "5/4": 5,
    "1/8": 6,
    "7/4": 7,
    "9/4": 8,
    "12/4": 9,
    "3/8": 10,
    "6/8": 11,
    "7/8": 12,
    "9/8": 13,
    "12/8": 14,
    "2/8": 15,
    "1/4": 15,
    "15/8": 16,
}

SIG_TIME_MAP = {
    0: "3/2",
    1: "4/2",
    2: "2/4",
    3: "3/4",
    4: "4/4",
    5: "5/4",
    6: "6/4",
    7: "7/4",
    8: "9/4",
    9: "12/4",
    10: "3/8",
    11: "6/8",
    12: "7/8",
    13: "9/8",
    14: "12/8",
    15: "1/4",
    16: "15/8",
}

KEY_MAP = {
    "cmajor": 0,
    "c#major": 1,
    "dbmajor": 1,
    "dmajor": 2,
    "d#major": 3,
    "ebmajor": 3,
    "emajor": 4,
    "fmajor": 5,
    "f#major": 6,
    "gbmajor": 6,
    "gmajor": 7,
    "g#major": 8,
    "abmajor": 8,
    "amajor": 9,
    "a#major": 10,
    "bbmajor": 10,
    "bmajor": 11,
    "cminor": 12,
    "c#minor": 13,
    "dbminor": 13,
    "dminor": 14,
    "d#minor": 15,
    "ebminor": 15,
    "eminor": 16,
    "fminor": 17,
    "f#minor": 18,
    "gbminor": 18,
    "gminor": 19,
    "g#minor": 20,
    "abminor": 20,
    "aminor": 21,
    "a#minor": 22,
    "bbminor": 22,
    "bminor": 23,
}

""" 0: 건반악기, 1: 리드악기(신스리드 포함), 2: 체명악기, 3: 발현악기, 4: 현악기(신스패드 포함)
    5: 부는악기(목관악기, 금관악기 포함), 6: 타악기, 7: 기타"""

INST_PROGRAM_MAP = {
    0: 0,  # 건반악기: Acoustic Piano
    1: 80,  # 리드악기: Synth Lead (Square)
    2: 8,  # 채명악기: Celesta
    3: 24,  # 발현악기: Acoustic Guitar
    4: 48,  # 현악기: String Ensemble
    5: 56,  # 부는악기: Trumpet
    6: 57,  # 타악기: Timpani
    7: 96,  # 기타: FX 1 (rain)
}

PROGRAM_INST_MAP = {
    "0": 0,
    "1": 0,
    "2": 0,
    "3": 0,
    "4": 0,
    "5": 0,
    "6": 0,
    "7": 0,
    "8": 2,
    "9": 2,
    "10": 2,
    "11": 2,
    "12": 2,
    "13": 2,
    "14": 2,
    "15": 2,
    "16": 0,
    "17": 0,
    "18": 0,
    "19": 0,
    "20": 0,
    "21": 1,
    "22": 1,
    "23": 1,
    "24": 3,
    "25": 3,
    "26": 3,
    "27": 3,
    "28": 3,
    "29": 3,
    "30": 3,
    "31": 3,
    "32": 3,
    "33": 3,
    "34": 3,
    "35": 3,
    "36": 3,
    "37": 3,
    "38": 3,
    "39": 3,
    "40": 4,
    "41": 4,
    "42": 4,
    "43": 4,
    "44": 4,
    "45": 4,
    "46": 3,
    "47": 6,  # 팀파니라 타악기
    "48": 4,
    "49": 4,
    "50": 4,
    "51": 4,
    "52": 7,
    "53": 7,
    "54": 7,
    "55": 6,
    "56": 5,
    "57": 5,
    "58": 5,
    "59": 5,
    "60": 5,
    "61": 5,
    "62": 5,
    "63": 5,
    "64": 5,
    "65": 5,
    "66": 5,
    "67": 5,
    "68": 5,
    "69": 5,
    "70": 5,
    "71": 5,
    "72": 5,
    "73": 5,
    "74": 5,
    "75": 5,
    "76": 5,
    "77": 5,
    "78": 5,
    "79": 5,
    "80": 1,
    "81": 1,
    "82": 1,
    "83": 1,
    "84": 1,
    "85": 1,
    "86": 1,
    "87": 1,
    "88": 4,
    "89": 4,
    "90": 4,
    "91": 4,
    "92": 4,
    "93": 4,
    "94": 4,
    "95": 4,
    "96": 7,
    "97": 7,
    "98": 7,
    "99": 7,
    "100": 7,
    "101": 7,
    "102": 7,
    "103": 7,
    "104": 3,
    "105": 3,
    "106": 3,
    "107": 3,
    "108": 1,
    "109": 5,
    "110": 4,
    "111": 5,
    "112": 6,
    "113": 6,
    "114": 6,
    "115": 6,
    "116": 6,
    "117": 6,
    "118": 6,
    "119": 6,
    "120": 7,
    "121": 7,
    "122": 7,
    "123": 7,
    "124": 7,
    "125": 7,
    "126": 7,
    "127": 7,
}

PITCH_RANGE_MAP = {
    "very_low": 0,
    "low": 1,
    "mid_low": 2,
    "mid": 3,
    "mid_high": 4,
    "high": 5,
    "very_high": 6,
}

# 로직에서 표기되는 채널에서 1을 빼야함 (로직과 달리 `mido`에서는 0부터 시작하므로)
# 순서대로 sfx, basic_beat, additional_beat, electric_beat
CHANNEL_NOT_FOR_MELODY = {
    "sfx": 6,
    "basic_beat": 9,
    "additional_beat": 10,
    "electric_beat": 11,
}
INSTRUMENT_NOT_FOR_MELODY = [
    55,
    96,
    97,
    98,
    99,
    100,
    101,
    102,
    103,
    113,
    114,
    115,
    116,
    117,
    118,
    119,
    120,
    121,
    122,
    123,
    124,
    125,
    126,
    127,
]

POZA_INST_MAP = {
    "accordion": 1,
    "acoustic_bass": 3,
    "acoustic_guitar": 3,
    "acoustic_piano": 0,
    "banjo": 3,
    "bassoon": 5,
    "bell": 6,
    "brass_ensemble": 5,
    "celesta": 2,
    "choir": 4,
    "clarinet": 5,
    "drums_full": 6,
    "drums_tops": 6,
    "electric_bass": 3,
    "electric_guitar_clean": 3,
    "electric_guitar_distortion": 3,
    "electric_piano": 0,
    "fiddle": 4,
    "flute": 5,
    "glockenspiel": 2,
    "harp": 3,
    "harpsichord": 0,
    "horn": 5,
    "keyboard": 0,
    "mandolin": 3,
    "marimba": 2,
    "nylon_guitar": 3,
    "oboe": 5,
    "organ": 0,
    "oud": 0,
    "pad_synth": 4,
    "percussion": 6,
    "recorder": 5,
    "sitar": 3,
    "string_cello": 4,
    "string_double_bass": 4,
    "string_ensemble": 4,
    "string_viola": 4,
    "string_violin": 4,
    "synth_bass": 3,
    "synth_bass_808": 3,
    "synth_bass_wobble": 3,
    "synth_bell": 6,
    "synth_lead": 1,
    "synth_pad": 4,
    "synth_pluck": 7,
    "synth_voice": 7,
    "timpani": 6,
    "trombone": 5,
    "trumpet": 5,
    "tuba": 5,
    "ukulele": 3,
    "vibraphone": 2,
    "whistle": 5,
    "xylophone": 2,
    "zither": 3,
    "orgel": 2,
}

DEFAULT_GENRE = "cinematic"
GENRE_MAP = {
    "newage": 0,
    "cinematic": 1,
    "children": 2,
    "jazz": 3,
    "funk": 4,
    "edm": 5,
    "acoustic": 6,
    # country -> acoustic
    "country": 6,
    "hiphop": 7,
    "rock": 8,
    "ambient": 9,
    "latin": 10,
    "reggae": 11,
    "traditional": 12,
}

CHANNEL_FOR_MELODY = {
    0: "main_melody",
    1: "sub_melody",
    2: "accompaniment",
    3: "bass",
    4: "pad",
    7: "riff",
}

TRACK_CATEGORY_MAP = {
    "main_melody": 0,
    "sub_melody": 1,
    "accompaniment": 2,
    "bass": 3,
    "pad": 4,
    "riff": 5,
}
NON_KEY_TRACK_CATEGORIES = ("drums", "percussion")

CHORD_NOTE = {
    "": [0, 4, 7],
    "maj7": [0, 4, 7, 11],
    "maj9": [0, 2, 4, 7, 11],
    "7": [0, 4, 7, 10],
    "6": [0, 4, 7, 9],  # 9인 경우, 8 인경우 3화음 취급
    "m": [0, 3, 7],
    "m6": [0, 3, 7, 9],
    "m7": [0, 3, 7, 10],
    "dim": [0, 3, 6],
    "dim7": [0, 3, 6, 9],
    "m7b5": [0, 3, 6, 10],
    "aug": [0, 4, 8],
    "sus4": [0, 5, 7],
    "sus2": [0, 2, 7],
    "add2": [0, 2, 4, 7],
    "madd2": [0, 2, 3, 7],
    "add4": [0, 4, 5, 7],
    "madd4": [0, 3, 5, 7],
    "7sus4": [0, 5, 7, 10],
    "madd9": [0, 2, 3, 7],
}

CHORD_NAME_FORM = {"-5": "b5", "M": "maj"}

VAL_NOTE_DICT = {
    0: ["C"],
    1: ["Db", "C#"],
    2: ["D"],
    3: ["Eb", "D#"],
    4: ["E"],
    5: ["F"],
    6: ["F#", "Gb"],
    7: ["G"],
    8: ["Ab", "G#"],
    9: ["A"],
    10: ["Bb", "A#"],
    11: ["B", "Cb"],
}

DEGREE = len(VAL_NOTE_DICT)
EIGHTH_NOTE_BEATS: float = 0.5
UNIT_MEASURES = 2

RHYTHM_MAP = {
    "standard": 0,
    "triplet": 1,
    "swing": 1,
}

CONTROL_CHANGE_DICT = {
    "modulation": 1,
    "expression": 11,
    "sustain": 64,
}

MODULATION_INTERVAL = 5
EXPRESSION_INTERVAL = 5
SUSTAIN_INTERVAL = 127

NUM_MODULATION_BINS = 64
NUM_EXPRESSION_BINS = 64
NUM_SUSTAIN_BINS = 2

# Standard pulses per quarter.
# https://en.wikipedia.org/wiki/Pulses_per_quarter_note
STANDARD_PPQ = 220
DEFAULT_PROGRAM = 0
DEFAULT_INSTRUMENT = 0

SUSTAIN_OFF = 0
SUSTAIN_ON = 127
SUSTAIN_THRESHOLD = 64
