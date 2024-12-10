import json
import pickle
from pathlib import Path
from typing import Any, Dict, List

import chardet
import jsonlines
import pandas as pd


def safe_pd_read_csv(path: str | Path, **kwargs: Dict[str, Any]) -> pd.DataFrame:
    """
    Reads a CSV file with `pd.read_csv` and automatically detects the encoding of the file.
    Parameters:
        path: Path to the CSV file to read.
        encoding: Encoding of the CSV file. If `None`, the encoding will be detected automatically.
        kwargs: Keyword arguments to be passed to `pd.read_csv`.
    Returns:
        A pandas DataFrame.
    """
    try:
        return pd.read_csv(path, **kwargs)
    except UnicodeDecodeError:
        encoding = get_file_encoding(path)
        return pd.read_csv(path, encoding=encoding, **kwargs)


def get_file_encoding(file_path: str | Path) -> str | None:
    with open(file_path, 'rb') as f:
        result = chardet.detect(f.read())
    return result['encoding']


def load_jsonlines(path: str | Path) -> List[Dict]:
    data_list = []
    with jsonlines.open(str(path)) as reader:
        for data in reader:
            data_list.append(data)
    return data_list


def save_as_jsonlines(
    data: List[Dict],
    path: str | Path,
    parents: bool = True,
    exist_ok: bool = True,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=parents, exist_ok=exist_ok)
    with jsonlines.open(str(path), mode="w") as writer:
        for datum in data:
            writer.write(datum)
    return


def load_json(path: str | Path) -> Dict:
    with open(str(path)) as f:
        data = json.load(f)
    return data  # type: ignore


def save_formatted_json(
    data: Dict | List | str,
    path: str | Path,
    parents: bool = True,
    exist_ok: bool = True,
) -> None:
    """
    Saves a dictionary or a list which is JSON serializable to a formatted JSON
    (UTF-8, 4 space indent).
    Paramters:
        data: A dictionary or a list which is JSON serializable. JSON data as string
            can also be input.
        path: Path to save the input `data` to.
        parents: Determines whether to make parent directories of the output file.
            Will be input to `pathlib.Path.mkdir` method.
        exist_ok: Determines whether to make parent directory if it exists already.
            Will be input to `pathlib.Path.mkdir` method.
    """
    path = Path(path)
    path.parent.mkdir(parents=parents, exist_ok=exist_ok)
    if isinstance(data, str):
        data = json.loads(data)
    with path.open(mode="w", encoding="utf-8") as fout:
        json.dump(data, fout, ensure_ascii=False, indent=4, separators=(",", ": "))
    return None


def save_pickle(
    data: Dict | List | str,
    path: str | Path,
    parents: bool = True,
    exist_ok: bool = True,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=parents, exist_ok=exist_ok)
    with path.open(mode="wb") as fout:
        pickle.dump(data, fout)
    return None


def load_pickle(
    path: str | Path,
) -> Dict | List | str:
    path = Path(path)
    with path.open(mode="rb") as fin:
        data = pickle.load(fin)
    return data  # type: ignore
