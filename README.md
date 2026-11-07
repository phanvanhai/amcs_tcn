# AMCS-TCN (FC head) on UT-HAR

Fork of [AMCS-TCN-by-BoYuan](https://github.com/boyuanzhang759-rgb/AMCS-TCN-by-BoYuan)
(Zhang et al., *Complex & Intelligent Systems* 12:146, 2026), set up to train only `AMCS_TCN_FC` on UT-HAR.

Changes vs. upstream:
- Added `datasets/csi_dataset.py` (UT-HAR loader, missing upstream) and `requirements.txt`.
- All dataset / training settings live in `config.py` (class names, number of classes, antennas,
  subcarriers, time steps, class weights, LR schedule, early stopping, noise).
- `AMCS_TCN` / `AMCS_TCN_FC` accept any number of antennas, subcarriers and time steps;
  the FC input size is computed from `time_steps`. Old checkpoints still load.
- Other models (GAP head, baselines) are commented out in `main.py` / `run_all.py`.

## Data

UT-HAR in SenseFi format (the `.csv` files are NumPy binaries):

```
UT_HAR/
  data/X_train.csv  X_val.csv  X_test.csv   # (N, 250, 90)
  label/y_train.csv y_val.csv  y_test.csv   # (N,) labels 0..6
```

The data is not in this repo. Locally the default path is set in `config.py`; on Kaggle the
loader searches `/kaggle/input` automatically. Use `--data_path` to point anywhere else.

### SSHAR ESP (`--dataset SSHAR_ESP`)

```
sshar_amp_interp/room_XX/esp/rx_{00,01,02}/subject_ZZ/amp_actAA_posBB_dirCC_repDD.npy   # (1, 56, 1000)
```

The 3 ESP32 receivers (1 antenna each) become the 3 AMCS streams: input `(3, T, 56)`, 8 classes
(act 1..8). Split by repetition as in the benchmark: test = dir0 rep 9-10 / dir!=0 rep 5,
train = the rest (5120 / 1280, 80/20). For now there is no separate val set (`use_val: False`):
the test set also picks the best epoch and drives early stopping, so results are optimistic.
Set `use_val: True` to move dir0 rep 8 / dir!=0 rep 4 into a val split (4160 / 960 / 1280).

Defaults: full length `time_steps=1000` with `batch_size=32` (batch 64 at T=1000 needs ~22 GB of
activations). `--time_steps 250` averages blocks of 4 samples instead.

```
python run_all.py --dataset SSHAR_ESP --filters none
python run_all.py --dataset SSHAR_ESP --filters none --time_steps 250 --batch_size 64
```

## Signal preprocessing

`datasets/preprocess.py` implements the paper's serial denoising (low-pass -> Hamming -> Kalman,
along time, per sample). Defaults and parameters are in `config.PREPROCESS` (the paper gives none).
Select filters with `--filters`, e.g. the Fig. 10 ablation:

```
python run_all.py --filters none
python run_all.py --filters lowpass
python run_all.py --filters lowpass hamming
python run_all.py                      # lowpass hamming kalman (default)
```

## Run on Kaggle

1. Upload the `UT_HAR` folder as a Kaggle Dataset (or use an existing public one) and add it to the notebook.
2. Notebook settings: **Accelerator = GPU T4**, **Internet = On**.
3. Cells:

```python
!git clone https://github.com/<user>/<repo>.git
%cd <repo>
!pip install -q -r requirements.txt
```

```python
!python main.py                 # single run, seed 3407, 150 epochs
# !python run_all.py            # 5 seeds, 100 epochs, mean ± std
# !python main.py --data_path /kaggle/input/<dataset>/UT_HAR --batch_size 32
```

Outputs of `main.py`: `model/AMCS_TCN_FC/best.pth`, `logs/AMCS_TCN_FC/`.

Outputs of `run_all.py` (one folder per run, summary rewritten after every seed):

```
results/<YYYYmmdd-HHMMSS>_UT_HAR/
  config.json, environment.json
  summary.txt          # mean ± std + comparison with the paper's numbers
  summary.csv
  AMCS_TCN_FC/
    per_seed.csv, per_class_mean_std.csv, summary.json
    seed_<s>/  best.pth, history.csv, curve.png, metrics.json, test_metrics_and_size.txt,
               classification_report.txt, per_class_metrics.csv, confusion_matrix.csv,
               cm.png, cm_normalized.png, predictions.csv
```

Clone into `/kaggle/working` so outputs are kept after the session.

For a private repo, store a GitHub token in Kaggle *Add-ons → Secrets* (e.g. `GITHUB_TOKEN`) and clone with:

```python
from kaggle_secrets import UserSecretsClient
token = UserSecretsClient().get_secret("GITHUB_TOKEN")
!git clone https://{token}@github.com/<user>/<repo>.git
```

## Run locally

```
pip install -r requirements.txt
python main.py
python run_all.py --seeds 42 3407
```
