# 3. Bone age from hand X-rays

A CNN looks at a child's hand X-ray, plus their sex, and predicts bone age in months.

- **Data:** [RSNA Pediatric Bone Age](https://www.kaggle.com/datasets/kmader/rsna-bone-age). Unzip into `data/`.
- **Key ideas:** pad each X-ray to a square before resizing so the hand isn't squashed, feed sex as an extra input, train on a normalized target and report in months.
- **Metric:** mean absolute error in months (overall, boys, girls), next to a predict-the-average baseline.
- **Status:** run on real data.

## Results

Full training set, split 10,720 train / 1,891 validation (seed 0). 256 px images, 8 epochs, batch 32, AdamW at 1e-4, about 55 minutes on an Apple-silicon Mac.

| | Validation error (months) |
| --- | --- |
| Always predict the average | 33.9 |
| This model | **7.3** (boys 7.3, girls 7.3) |

Per epoch: 10.2, 10.1, 9.1, 9.1, 7.9, 7.5, 7.5, 7.3. The error was still falling at epoch 8.

For context, the RSNA challenge winner reached 4.2 months, using larger images and an ensemble. This is a single model at low resolution, scored on my own validation split rather than the challenge test set.

## Run it

```bash
../.venv/bin/python test_train.py
```

```bash
../.venv/bin/python train.py --csv data/boneage-training-dataset.csv --images data/boneage-training-dataset/boneage-training-dataset --size 256 --epochs 8 --batch 32 --workers 4
```

Add `--subset 500` for a quick trial run. The best weights are saved to `model.pt`.
