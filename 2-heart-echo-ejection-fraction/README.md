# 2. Heart pumping strength from ultrasound video

A video model watches an echocardiogram and estimates the ejection fraction: the share of blood the heart pumps out per beat.

- **Data:** [EchoNet-Dynamic](https://echonet.github.io/dynamic/) (request access through Stanford AIMI). Unpack into `data/` so it holds `FileList.csv` and `Videos/`.
- **Key ideas:** official TRAIN/VAL/TEST splits (one video per patient), 32-frame clips taking every 2nd frame from a random start, average several clips per video at test time.
- **Metric:** mean absolute error in EF percentage points, next to a predict-the-average baseline.
- **Status:** self-checked on synthetic videos only. Not yet run on the real dataset.

```bash
../.venv/bin/python test_train.py
```

```bash
../.venv/bin/python train.py --csv data/FileList.csv --videos data/Videos --subset 500
```

Add `--test` only for the final run; it scores the held-out TEST split.
