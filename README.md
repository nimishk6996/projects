# Medical ML projects

Three small deep-learning projects on public medical imaging datasets. Each folder is self-contained: `train.py` (the whole pipeline), `test_train.py` (a self-check on synthetic data) and a `README.md`.

| Folder | Input | Predicts | Model | Status |
| --- | --- | --- | --- | --- |
| [1-brain-bleed-ct](1-brain-bleed-ct) | Head CT slice (DICOM) | Bleed yes/no + 5 bleed types | ResNet-50, 6 outputs | Code self-checked on synthetic slices; not yet run on real data |
| [2-heart-echo-ejection-fraction](2-heart-echo-ejection-fraction) | Heart ultrasound video | Ejection fraction (%) | R(2+1)D video CNN | Code self-checked on synthetic videos; not yet run on real data |
| [3-bone-age-hand-xray](3-bone-age-hand-xray) | Child's hand X-ray + sex | Bone age (months) | ResNet-50 + sex input | Trained on the full dataset: 7.3 months error vs 33.9 baseline |

## Setup

```bash
python3 -m venv .venv
```

```bash
.venv/bin/python -m pip install -r requirements.txt
```

Open this folder in VS Code and use **Tasks: Run Task** to run any self-check.

## Rules these projects follow

- **Split by patient, never by image.** Slices or videos from one patient look alike, so mixing them inflates scores.
- **Start with a subset.** Every script has `--subset N`.
- **No data in this repo.** The datasets are research-licensed; `data/` is git-ignored. Download them yourself from the links in each project.
