from pathlib import Path
from typing import Dict, Union

import pandas as pd

from dioai.preprocessor.utils import constants


class TableReader:
    """CSV table reader.
    https://github.com/POZAlabs/melydlf/blob/master/melydlf/preprocessor/parser/v2/csv/_reader.py
    """

    STATUS_COL = "상태"
    VALID_STATUS = "c"
    FIRST_COL_NAME = "고유번호"
    LAST_COL_NAME = "재검수 일시 (통과일시)"
    GENRE = "장르"

    def __init__(self, file_path: Union[str, Path]):
        self.file_path = file_path

    def get_meta_dict(self) -> Dict:
        return self._get_meta_dict(self.file_path)

    def _get_meta_dict(self, file_path: Union[str, Path]) -> Dict:
        """Get filename-genre mapper dictionary."""
        valid_table = self._get_valid_table(file_path)
        meta_dict = dict()
        for record in valid_table.to_dict("records"):
            midi_name = record[self.FIRST_COL_NAME]
            genre = record[self.GENRE].lower()
            meta_dict[midi_name] = genre
        return meta_dict

    def _get_valid_table(self, file_path: Union[str, Path]) -> pd.DataFrame:
        table = self._read_meta_csv(file_path)

        valid_table = table.fillna(value={self.GENRE: constants.UNKNOWN})
        valid_table[self.GENRE] = valid_table[self.GENRE].apply(lambda x: x.lower())
        valid_table = (
            valid_table.loc[valid_table[self.GENRE].isin(constants.GENRE_MAP.keys())]
            .dropna(subset=[self.FIRST_COL_NAME])
            .reset_index(drop=True)
            .loc[:, [self.FIRST_COL_NAME, self.GENRE]]
        )
        return valid_table

    def _read_meta_csv(self, file_path: Union[str, Path]) -> pd.DataFrame:
        raw_df = pd.read_csv(file_path)
        try:
            df = self._set_valid_col(self._set_valid_header(raw_df))
        # When cannot find column names (in `self._set_valid_header`).
        except TypeError:
            df = raw_df
        return df

    def _set_valid_header(self, df: pd.DataFrame) -> pd.DataFrame:
        header_idx = None
        for idx, row in df.iterrows():
            _, values = zip(*row.items())
            if self.STATUS_COL in values:
                header_idx = idx
                break

        header = df.iloc[header_idx]
        df = df[(header_idx + 1) :]
        df.columns = header
        return df

    def _set_valid_col(self, df: pd.DataFrame) -> pd.DataFrame:
        start_idx = df.columns.get_loc(self.FIRST_COL_NAME)
        end_idx = df.columns.get_loc(self.LAST_COL_NAME)
        return df.iloc[:, start_idx : (end_idx + 1)]
