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
(act 1..8). Default split is **cross-user** 7:1:2 (`--split user`): user case 0 of the benchmark,
train subjects 1,2,3,4,9,10,11 / val 12 / test 13,14 -> 4480 / 640 / 1280 (`--user_case 0..4` for the
other cases). `--split repeat` splits by repetition instead (every subject in every set). Full length `time_steps=1000`
(`--time_steps 250` averages blocks of 4 samples instead).

## Experiment protocol

Batch 32, up to 100 epochs, Adam, lr 0.001 (paper Table 1), model = best val accuracy, early
stopping after 10 epochs without val accuracy improvement (`--patience N` to change); StepLR(20, 0.5)
as in the authors' code. Metrics: accuracy, macro precision /
recall / F1.

```
python run_all.py --dataset SSHAR_ESP --filters none   # seed 42: checkpoint by val -> TEST (+ VAL of that checkpoint)
```

Dataset, device and split details shared by all models: [docs/SSHAR_ESP_protocol.md](docs/SSHAR_ESP_protocol.md).
Shared Kaggle cell, CLI flags, run naming and result layout: [docs/kaggle_run_convention.md](docs/kaggle_run_convention.md).

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

SSHAR ESP: use the shared cell in [docs/kaggle_run_convention.md](docs/kaggle_run_convention.md)
(section 6). Edit only the config block (`SPLIT`, `USER_CASE`, `PATIENCE`, `SEEDS`, `FILTERS`);
notebook settings: **GPU**, **Internet = On**.

Equivalent command:

```
python -u run_all.py --dataset SSHAR_ESP --filters none --split user --user_case 0 \
    --patience 10 --seeds 42 --data_path <folder containing room_02> --results_dir results
```

Without `--run_name` the run folder is named by the convention, e.g.
`results/SSHAR_ESP_AMCS_TCN_FC_user0_p10_seed42/`:

```
results/<RUN>/
  config.json, environment.json
  summary.txt          # split + subjects, patience, accuracy / precision / recall / F1 on test (+ val)
  summary.csv
  AMCS_TCN_FC/
    per_seed.csv, per_class_mean_std.csv, summary.json
    seed_<s>/  best.pth, history.csv, curve.png, metrics.json, test_metrics_and_size.txt,
               classification_report.txt, per_class_metrics.csv, confusion_matrix.csv,
               cm.png, cm_normalized.png, predictions.csv
               val/   (same metrics on the val set, same checkpoint)
```

UT-HAR: `python run_all.py --dataset UT_HAR --filters none` (fixed SenseFi split).

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
