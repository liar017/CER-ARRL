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
| AMCF | `data/<dataset>.mat` and the derived `.ratings`, `.genre`, `.iave`, and `.uave` files. The script derives missing average files from training ratings. It preserves average files already supplied in the repository. |
| BPER-J | `data/<dataset>/<dataset>.mat` for `run_bperj.py` and `eval.py`. Pass this path with `--data_path` when selecting a dataset. |
| PITF | `data/<dataset>/<dataset>.mat` for the MAT-based `train.py` entry point. |
| SULM | `data/<dataset>/<dataset>_train.txt` and `<dataset>_test.txt`. Each row contains zero-based user and item IDs, a binary rating (`D[:, 0] >= 0`), and the original aspect labels. Every fifth row goes to the test split. |

The script links to the shared MAT files where possible and reuses existing
byte-identical copies. Run each baseline from its own directory because the
original loaders use relative paths. Select one of the five dataset names in
the entry point or use its dataset argument where supported. The generated
local inputs are ignored by Git. The original BPER and BPERP entry points in
the BPER-J folder use a separate data format.

Use `--baseline AMCF --regenerate-amcf-averages` to recompute the averages
from the shared MAT inputs, including the existing tracked averages. This
option updates those files in the working tree.
