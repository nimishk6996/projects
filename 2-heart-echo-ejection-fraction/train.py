"""Ejection fraction from echocardiogram video (EchoNet-Dynamic). Usage: see README.md."""
import argparse

import cv2
import numpy as np
import pandas as pd
import torch
import torchvision


def load_video(path):
    cap, frames = cv2.VideoCapture(str(path)), []
    ok, frame = cap.read()
    while ok:
        frames.append(frame)
        ok, frame = cap.read()
    if not frames:
        raise FileNotFoundError(path)
    return np.stack(frames).transpose(3, 0, 1, 2).astype(np.float32)  # C, T, H, W


def sample_clip(video, start, length=32, period=2):
    c, t, h, w = video.shape
    need = start + (length - 1) * period + 1
    if t < need:  # pad short videos with empty frames
        video = np.concatenate([video, np.zeros((c, need - t, h, w), np.float32)], 1)
    return video[:, start + period * np.arange(length)]


class Clips(torch.utils.data.Dataset):
    """One random clip per video per epoch; the changing start doubles as augmentation."""

    def __init__(self, files, videos, norm, length, period):
        self.files, self.videos, self.norm, self.length, self.period = files.reset_index(drop=True), videos, norm, length, period

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        r = self.files.iloc[i]
        v = load_video(video_path(self.videos, r.FileName))
        span = (self.length - 1) * self.period + 1
        start = np.random.randint(0, max(v.shape[1] - span, 0) + 1)
        return torch.from_numpy(self.norm(sample_clip(v, start, self.length, self.period))), torch.tensor([r.EF], dtype=torch.float32)


def video_path(videos, name):
    return f"{videos}/{name}" if name.endswith(".avi") else f"{videos}/{name}.avi"


def predict_ef(model, video, norm, dev, length=32, period=2, n_clips=5):
    """Predict on several clips spread across the video and average them."""
    span = (length - 1) * period + 1
    starts = np.linspace(0, max(video.shape[1] - span, 0), n_clips).astype(int)
    clips = np.stack([norm(sample_clip(video, s, length, period)) for s in starts])
    with torch.no_grad():
        return model(torch.from_numpy(clips).to(dev)).mean().item()


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True, help="FileList.csv")
    p.add_argument("--videos", required=True, help="folder holding the .avi files")
    p.add_argument("--subset", type=int, help="use only N videos per split (start small)")
    p.add_argument("--frames", type=int, default=32)
    p.add_argument("--period", type=int, default=2, help="take every Nth frame")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--no-pretrained", action="store_true")
    p.add_argument("--test", action="store_true", help="also score the TEST split (only at the very end)")
    p.add_argument("--out", default="model.pt")
    a = p.parse_args(argv)

    files = pd.read_csv(a.csv)  # FileName, EF, ESV, EDV, FPS, ..., Split
    assert files.EF.max() > 1, "EF should be a percentage (e.g. 69.4), not a 0-1 fraction"
    split = lambda name: files[files.Split == name][:a.subset]  # official splits, one video per patient
    train, val = split("TRAIN"), split("VAL")

    # per-channel mean and std, computed once from a sample of training videos
    sample = np.concatenate([load_video(video_path(a.videos, f)).reshape(3, -1)
                             for f in train.FileName.sample(min(32, len(train)), random_state=0)], 1)
    mean, std = sample.mean(1).reshape(3, 1, 1, 1), sample.std(1).reshape(3, 1, 1, 1)
    norm = lambda x: (x - mean) / std

    dev = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    model = torchvision.models.video.r2plus1d_18(weights=None if a.no_pretrained else "KINETICS400_V1")
    model.fc = torch.nn.Linear(model.fc.in_features, 1)
    model.fc.bias.data[0] = train.EF.mean()  # start from the average EF so early training isn't spent finding it
    model.to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr)
    loss_fn = torch.nn.MSELoss()
    train_dl = torch.utils.data.DataLoader(Clips(train, a.videos, norm, a.frames, a.period),
                                           batch_size=a.batch, shuffle=True, num_workers=a.workers)

    def mae(part):
        model.eval()
        pred = [predict_ef(model, load_video(video_path(a.videos, f)), norm, dev, a.frames, a.period) for f in part.FileName]
        return float(np.abs(np.array(pred) - part.EF.values).mean())

    baseline = float(np.abs(train.EF.mean() - val.EF.values).mean())
    best = float("inf")
    for epoch in range(a.epochs):
        model.train()
        for x, y in train_dl:
            loss = loss_fn(model(x.to(dev)), y.to(dev))
            opt.zero_grad()
            loss.backward()
            opt.step()
        val_mae = mae(val)
        print(f"epoch {epoch + 1}: train loss {loss.item():.1f} | val MAE {val_mae:.2f} EF points | mean-baseline {baseline:.2f}")
        if val_mae < best:
            best = val_mae
            torch.save({"model": model.state_dict(), "mean": mean.ravel().tolist(), "std": std.ravel().tolist()}, a.out)
    result = {"mae": best, "baseline": baseline, "last_loss": loss.item()}
    if a.test:
        result["test_mae"] = mae(split("TEST"))
        print(f"TEST MAE {result['test_mae']:.2f} EF points")
    return result


if __name__ == "__main__":
    main()
