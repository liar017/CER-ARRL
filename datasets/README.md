# Baseline datasets

This directory holds the shared MATLAB inputs for Beeradvocate, HotelRec,
OpenRiceEmoji, OpenRiceText, and Yelp. The original interaction order and the
`iu`, `is`, `D`, and `E` fields are preserved.

Run from the repository root on a system that supports symbolic links:

```bash
python -m pip install numpy scipy
python baselines/datapreprocess.py
```

To prepare only particular datasets or baselines:

```bash
python baselines/datapreprocess.py --dataset Yelp OpenRiceEmoji --baseline AMCF SULM
```

| Baseline | Inputs prepared in its local directory |
| --- | --- |
| AMCF | Reads `datasets/<dataset>.mat` directly and uses the derived `.ratings`, `.genre`, `.iave`, and `.uave` files. The script derives averages from the shared training rows, replacing any previously supplied averages for the selected datasets. |
| BPER-J | `data/<dataset>/<dataset>.mat` for `run_bperj.py` and `eval.py`. Pass this path with `--data_path` when selecting a dataset. |
| PITF | `data/<dataset>/<dataset>.mat` for `train.py` and `predict.py`. Both accept `--dataset`. |
| SULM | `data/<dataset>/<dataset>_train.txt` and `<dataset>_test.txt`. Each row contains zero-based user and item IDs, a binary rating (`D[:, 0] >= 0`), and the original aspect labels. Every fifth row goes to the test split. Both `sulm.py` and `eval.py` accept `--dataset`. |

The script links to the shared MAT files where possible and reuses existing
byte-identical copies. Run each baseline from its own directory because the
original loaders use relative paths. Select one of the five dataset names in
the entry point or use its dataset argument where supported. The generated
local inputs are ignored by Git. The original BPER and BPERP entry points in
the BPER-J folder use a separate data format.

AMCF reads the shared source even when an older copy exists in its local data
directory. Regenerated AMCF averages that were already tracked by Git appear
as working-tree changes. Use `--no-regenerate-amcf-averages` only when you
intentionally want to retain existing averages.

All four baselines start from the same rows and use the same split positions.
BPER-J and PITF apply their existing filter for nonnegative overall ratings
within each split. SULM keeps both classes and AMCF keeps all rows.
