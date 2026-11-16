# 3. Bone age from hand X-rays

A CNN looks at a child's hand X-ray, plus their sex, and predicts bone age in months.

- **Data:** [RSNA Pediatric Bone Age](https://www.kaggle.com/datasets/kmader/rsna-bone-age). Unzip into `data/`.
- **Key ideas:** pad each X-ray to a square before resizing so the hand isn't squashed, feed sex as an extra input, train on a normalized target and report in months.
- **Metric:** mean absolute error in months (overall, boys, girls), next to a predict-the-average baseline.
- **Status:** run on real data.

## Result so far

500-image subset (425 train / 75 validation), 256 px, 3 epochs, Apple-silicon Mac:

| | Validation error (months) |
| --- | --- |
| Always predict the average | 37.8 |
| This model | 21.8 (boys 19.4, girls 23.9) |

This is a quick subset run, not a tuned result. The RSNA challenge winner reached 4.2 months on the full dataset.

```bash
../.venv/bin/python test_train.py
```

```bash
../.venv/bin/python train.py --csv data/boneage-training-dataset.csv --images data/boneage-training-dataset/boneage-training-dataset --subset 500 --size 256 --epochs 3
```

Drop `--subset` for the full run. The best weights are saved to `model.pt`.
