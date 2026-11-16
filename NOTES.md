# Project notes

Working notes: where things stand and how to pick this up on another machine.

## Status (16 Nov 2026)

| Project | State | Next step |
| --- | --- | --- |
| 3-bone-age-hand-xray | Trained on the full RSNA dataset; results in its README | Longer run at 512 px, test-set score |
| 2-heart-echo-ejection-fraction | Pipeline written and self-checked on synthetic videos | Needs EchoNet-Dynamic (about 7 GB, Stanford access request) |
| 1-brain-bleed-ct | Pipeline written and self-checked on synthetic DICOMs | Needs a subset of RSNA ICH (full set is several hundred GB) |

## Setting up on a new machine

1. Clone the repo and open the folder in VS Code (Python extension).
2. Create the environment and install the pinned packages (commands in `README.md`).
3. Run the three self-checks from **Tasks: Run Task**. They need no data.
4. For bone age, download the dataset from Kaggle and unzip it into `3-bone-age-hand-xray/data/`.

## What is not in the repo

- `data/` folders: research-licensed datasets, re-download from the links in each README.
- `model.pt`: trained weights (about 96 MB). Recreate with the training command in the bone age README.
- `.venv/`: recreate from `requirements.txt`.
