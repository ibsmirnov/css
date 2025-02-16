import os.path
from typing import Optional, List

import pandas as pd


def load_dataset(csv_path: Optional[str] = None, header: Optional[List[str]] = None) -> pd.DataFrame:
    if header is None:
        header = ["Type", "Speaker", "Topic", "Year"]
        header = ["Speaker", "Topic"]
    if csv_path is None:
        # to save space, we provide a pre-filtered dataset
        csv_path = os.path.abspath("css.csv")
    df = pd.read_csv(csv_path)
    df = df[header]
    df = df.dropna()
    df['combined'] = df.apply(
        lambda row: '; '.join([f'{h}: {row.get(h, "__noname__")}' for h in header]),
        axis=1
    )
    return df
