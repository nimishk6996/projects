"""Self-check on synthetic echo videos: ../.venv/bin/python test_train.py"""
import math
import tempfile

import cv2
import numpy as np
import pandas as pd

import train

# sample_clip: takes every 2nd frame from the start, and pads short videos with empty frames
video = np.zeros((3, 20, 4, 4), np.float32)
video[:, :, 0, 0] = np.arange(20)  # frame number stored in one pixel
clip = train.sample_clip(video, start=3, length=5, period=2)
assert clip.shape == (3, 5, 4, 4) and list(clip[0, :, 0, 0]) == [3, 5, 7, 9, 11]
short = train.sample_clip(video, start=15, length=5, period=2)  # needs frames 15..23, only 20 exist
assert list(short[0, :, 0, 0]) == [15, 17, 19, 0, 0]

with tempfile.TemporaryDirectory() as d:
    rng = np.random.default_rng(0)
    rows = []
    for i, part in enumerate(["TRAIN"] * 6 + ["VAL"] * 2 + ["TEST"] * 2):
        n = int(rng.integers(12, 30))  # videos of different lengths, some shorter than one clip
        w = cv2.VideoWriter(f"{d}/v{i}.avi", cv2.VideoWriter_fourcc(*"MJPG"), 50, (32, 32))
        for _ in range(n):
            w.write(rng.integers(0, 255, (32, 32, 3), np.uint8))
        w.release()
        rows.append({"FileName": f"v{i}", "EF": float(rng.uniform(30, 70)), "Split": part})
    pd.DataFrame(rows).to_csv(f"{d}/FileList.csv", index=False)

    # load_video: channels x frames x height x width
    v = train.load_video(f"{d}/v0.avi")
    assert v.shape[0] == 3 and v.shape[2:] == (32, 32) and v.shape[1] >= 12

    # one epoch end to end, including the held-out TEST score
    r = train.main(["--csv", f"{d}/FileList.csv", "--videos", d, "--frames", "8", "--epochs", "1", "--batch", "2",
                    "--no-pretrained", "--test", "--out", f"{d}/model.pt"])
    assert all(math.isfinite(x) for x in r.values()) and "test_mae" in r, r

print("ok")
