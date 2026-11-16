# 1. Brain bleed detection on CT

A CNN looks at one slice of a head CT and predicts whether there is bleeding inside the skull, and which of five types.

- **Data:** [RSNA Intracranial Hemorrhage Detection](https://www.kaggle.com/c/rsna-intracranial-hemorrhage-detection/) (non-commercial, no redistribution). Very large; download a subset of the DICOM files into `data/`.
- **Key ideas:** convert pixels to Hounsfield units, stack three contrast windows (brain, blood, soft tissue) as the 3 colour channels, split by `PatientID`, six sigmoid outputs.
- **Metric:** log loss and AUC per subtype.
- **Status:** self-checked on synthetic DICOMs only. Not yet run on the real dataset.

```bash
../.venv/bin/python test_train.py
```

```bash
../.venv/bin/python train.py --csv data/stage_2_train.csv --images data/stage_2_train --subset 5000
```

The script only uses slices whose `.dcm` file exists in `--images`, so a partial download works.
