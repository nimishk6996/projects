"""Self-check on synthetic X-rays: ../.venv/bin/python test_train.py"""
import math
import tempfile

import cv2
import numpy as np
import pandas as pd
import torch

import train

with tempfile.TemporaryDirectory() as d:
    # load_hand: pads to a square, content centred, nothing stretched
    cv2.imwrite(f"{d}/wide.png", np.full((100, 300), 255, np.uint8))
    cv2.imwrite(f"{d}/tall.png", np.full((300, 100), 255, np.uint8))
    wide, tall = train.load_hand(f"{d}/wide.png", 60), train.load_hand(f"{d}/tall.png", 60)
    assert wide.shape == tall.shape == (60, 60)
    assert wide[:20].max() == 0 and wide[40:].max() == 0 and wide[20:40].min() == 255
    assert tall[:, :20].max() == 0 and tall[:, 40:].max() == 0 and tall[:, 20:40].min() == 255

    # model: output shape (N, 1)
    out = train.BoneAgeNet(pretrained=False)(torch.zeros(2, 3, 64, 64), torch.ones(2, 1))
    assert out.shape == (2, 1)

    # one epoch end to end on 16 fake X-rays of mixed, non-square sizes
    rng = np.random.default_rng(0)
    rows = []
    for i in range(16):
        cv2.imwrite(f"{d}/{i}.png", rng.integers(0, 255, (rng.integers(80, 160), rng.integers(80, 160)), np.uint8))
        rows.append({"id": i, "boneage": int(rng.integers(12, 216)), "male": bool(i % 2)})
    pd.DataFrame(rows).to_csv(f"{d}/labels.csv", index=False)
    r = train.main(["--csv", f"{d}/labels.csv", "--images", d, "--size", "64", "--epochs", "1",
                    "--batch", "4", "--no-pretrained", "--out", f"{d}/model.pt"])
    assert all(math.isfinite(v) for v in r.values()), r
    assert "model" in torch.load(f"{d}/model.pt")

print("ok")
