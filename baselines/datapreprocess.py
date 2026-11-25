"""Prepare the local data layout used by the four baseline entry points."""

import argparse
from contextlib import contextmanager
from filecmp import cmp
from os.path import relpath
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
from scipy.io import loadmat


DATASETS = ("Beeradvocate", "HotelRec", "OpenRiceEmoji", "OpenRiceText", "Yelp")
BASELINES = ("AMCF", "BPER-J", "PITF", "SULM")
BASELINE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASELINE_DIR.parent / "datasets"


@contextmanager
def output_file(path):
    """Replace a generated file only after it has been written successfully."""
    tmp = NamedTemporaryFile(mode="w", encoding="ascii", dir=path.parent, delete=False)
    temp_path = Path(tmp.name)
    try:
        with tmp:
            yield tmp
        temp_path.replace(path)
    finally:
        temp_path.unlink(missing_ok=True)


def link_mat(source, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.resolve() == source:
        return
    if destination.is_file() and cmp(source, destination, shallow=False):
        return
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"Move the existing file before creating the link: {destination}")
    destination.symlink_to(relpath(source, destination.parent))


def write_amcf(directory, name, users, items, ratings, aspects, train, regenerate_averages):
    train_users, train_items = users[train], items[train]
    # Match the transformation used when AMCF writes .ratings.
    train_ratings = 2 * ratings[train] + 3
    with output_file(directory / f"{name}.ratings") as output:
        np.savetxt(output, np.column_stack((train_users, train_items, train_ratings)),
                   fmt=("%d", "%d", "%.15g"), delimiter=",")

    genres = np.zeros((items.max() + 1, aspects.shape[1]), dtype=np.uint8)
    np.maximum.at(genres, train_items, aspects[train])
    with output_file(directory / f"{name}.genre") as output:
        np.savetxt(output, np.column_stack((np.arange(len(genres)), genres)),
                   fmt="%d", delimiter=",")

    for suffix, id_header, ids in (("uave", "uid", train_users), ("iave", "mid", train_items)):
        path = directory / f"{name}.{suffix}"
        if path.is_file() and not regenerate_averages:
            continue
        counts = np.bincount(ids)
        totals = np.bincount(ids, weights=train_ratings)
        present = np.flatnonzero(counts)
        with output_file(path) as output:
            np.savetxt(output, np.column_stack((present, totals[present] / counts[present])),
                       fmt=("%d", "%.3f"), delimiter=",", header=f"{id_header},ave", comments="")


def write_sulm(directory, name, users, items, ratings, aspects, train):
    directory.mkdir(parents=True, exist_ok=True)
    for split, mask in (("train", train), ("test", ~train)):
        indices = np.flatnonzero(mask)
        with output_file(directory / f"{name}_{split}.txt") as output:
            for start in range(0, len(indices), 10000):
                rows = indices[start:start + 10000]
                values = np.column_stack((users[rows], items[rows], ratings[rows] >= 0, aspects[rows]))
                np.savetxt(output, values, fmt="%d")


def prepare(name, baselines, regenerate_averages):
    source = DATA_DIR / f"{name}.mat"
    if not source.is_file():
        raise FileNotFoundError(source)

    for baseline in baselines:
        directory = BASELINE_DIR / baseline / "data"
        if baseline == "AMCF":
            link_mat(source, directory / source.name)
        elif baseline in ("PITF", "BPER-J"):
            link_mat(source, directory / name / source.name)

    if "AMCF" in baselines or "SULM" in baselines:
        raw = loadmat(source, variable_names=("iu", "is", "D", "E"))
        users = raw["iu"].ravel().astype(np.int64) - 1
        items = raw["is"].ravel().astype(np.int64) - 1
        ratings = raw["D"][:, 0]
        aspects = raw["E"].astype(np.uint8, copy=False)
        train = np.arange(len(users)) % 5 != 4
        if "AMCF" in baselines:
            write_amcf(BASELINE_DIR / "AMCF/data", name, users, items, ratings, aspects,
                       train, regenerate_averages)
        if "SULM" in baselines:
            write_sulm(BASELINE_DIR / "SULM/data" / name, name, users, items, ratings, aspects, train)

    print(f"{name}: {', '.join(baselines)}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=DATASETS, nargs="+", default=DATASETS)
    parser.add_argument("--baseline", choices=BASELINES, nargs="+", default=BASELINES)
    parser.add_argument("--regenerate-amcf-averages", action="store_true",
                        help="overwrite existing AMCF average files using the selected training split")
    args = parser.parse_args()
    for name in args.dataset:
        prepare(name, args.baseline, args.regenerate_amcf_averages)


if __name__ == "__main__":
    main()
