import multiprocessing

DEFAULT_NUM_BEATS = 4
CHORD_TRACK_NAME = "chord"
PITCH_RANGE_CUT = {
    "very_low": 36,
    "low": 48,
    "mid_low": 60,
    "mid": 72,
    "mid_high": 84,
    "high": 96,
    "very_high": 108,
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
MAX_BPM = 200

DEFAULT_BPM = 24
DEFAULT_KEY = 0
DEFAULT_TS = 4

BPM_START_POINT = 1
KEY_START_POINT = 42
TS_START_POINT = 67
PITCH_RANGE_START_POINT = 85
MEASURES_4 = 92
MEASURES_8 = 93
INST_START_POINT = 95
META_LEN = 103

BPM_UNKHOWN = 0
KEY_UNKHOWN = 41
TS_UNKHOWN = 66
PITCH_RANGE_UNKHOWN = 84
INST_UNKHOWN = 94

NUM_CORES = multiprocessing.cpu_count()

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
    15: "2/2",
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
