import pickle as p
from pathlib import Path


def dump_data(array, out_dir, array_name):
    out_path = Path(out_dir) / array_name

    with open(out_path.with_suffix(".p"), "wb") as out:
        p.dump(array, out, protocol=4)


def load_data(file_dir, file_name):
    path = Path(file_dir) / file_name

    with open(path, "rb") as file:
        contents = p.load(file)

    return contents
