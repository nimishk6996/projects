"""Bone age from hand X-rays (RSNA Pediatric Bone Age). Usage: see README.md."""
import argparse

import cv2
import numpy as np
import pandas as pd
import torch
import torchvision
from torchvision import transforms

IMNET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMNET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def load_hand(path, size=512):
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(path)
    h, w = img.shape
    s = max(h, w)  # pad to a square first, so the hand isn't squashed
    pad = np.zeros((s, s), np.uint8)
    pad[(s - h) // 2:(s - h) // 2 + h, (s - w) // 2:(s - w) // 2 + w] = img
    return cv2.resize(pad, (size, size), interpolation=cv2.INTER_AREA)


class Hands(torch.utils.data.Dataset):
    def __init__(self, df, images, size, aug=None):
        self.df, self.images, self.size, self.aug = df.reset_index(drop=True), images, size, aug

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        x = torch.from_numpy(load_hand(f"{self.images}/{int(r.id)}.png", self.size)).float().div(255)[None]
        if self.aug:
            x = self.aug(x)
        x = (x.repeat(3, 1, 1) - IMNET_MEAN) / IMNET_STD  # pretrained ResNet wants 3 ImageNet-normalized channels
        return x, torch.tensor([r.male], dtype=torch.float32), torch.tensor([r.target], dtype=torch.float32)


class BoneAgeNet(torch.nn.Module):  # image features + sex
    def __init__(self, pretrained=True):
        super().__init__()
        self.cnn = torchvision.models.resnet50(weights="IMAGENET1K_V2" if pretrained else None)
        n = self.cnn.fc.in_features
        self.cnn.fc = torch.nn.Identity()
        self.sex = torch.nn.Linear(1, 32)
        self.head = torch.nn.Sequential(torch.nn.Linear(n + 32, 256), torch.nn.ReLU(), torch.nn.Linear(256, 1))

    def forward(self, x, male):
        return self.head(torch.cat([self.cnn(x), torch.relu(self.sex(male))], 1))


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True)
    p.add_argument("--images", required=True, help="folder holding <id>.png")
    p.add_argument("--subset", type=int, help="use only N images (start small)")
    p.add_argument("--size", type=int, default=512)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--no-pretrained", action="store_true")
    p.add_argument("--out", default="model.pt")
    a = p.parse_args(argv)

    df = pd.read_csv(a.csv)
    df["male"] = df["male"].astype(float)
    df = df.sample(frac=1, random_state=0)  # fixed shuffle; one X-ray per child, so an image split is a patient split
    if a.subset:
        df = df[:a.subset]
    print(df.groupby("male").boneage.describe())
    n_val = max(1, int(len(df) * 0.15))
    val, train = df[:n_val].copy(), df[n_val:].copy()
    mean, std = train.boneage.mean(), train.boneage.std()  # train split only, no leakage
    for d in (train, val):
        d["target"] = (d.boneage - mean) / std

    aug = transforms.Compose([transforms.RandomAffine(10, translate=(0.05, 0.05)),
                              transforms.ColorJitter(brightness=0.2)])
    load = lambda d, aug, shuffle: torch.utils.data.DataLoader(
        Hands(d, a.images, a.size, aug), batch_size=a.batch, shuffle=shuffle, num_workers=a.workers)
    train_dl, val_dl = load(train, aug, True), load(val, None, False)

    dev = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    model = BoneAgeNet(not a.no_pretrained).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr)
    loss_fn = torch.nn.L1Loss()

    truth, is_male = val.boneage.values, val.male.values == 1
    baseline = np.abs(mean - truth).mean()
    best = float("inf")
    for epoch in range(a.epochs):
        model.train()
        for x, male, y in train_dl:
            loss = loss_fn(model(x.to(dev), male.to(dev)), y.to(dev))
            opt.zero_grad()
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            pred = torch.cat([model(x.to(dev), male.to(dev)).cpu() for x, male, _ in val_dl]).numpy()[:, 0]
        err = np.abs(pred * std + mean - truth)  # back to months
        mae = err.mean()
        print(f"epoch {epoch + 1}: train loss {loss.item():.3f} | val MAE {mae:.1f} months "
              f"(boys {err[is_male].mean():.1f}, girls {err[~is_male].mean():.1f}) | mean-baseline {baseline:.1f}")
        if mae < best:
            best = mae
            torch.save({"model": model.state_dict(), "mean": float(mean), "std": float(std), "size": a.size}, a.out)
    return {"mae": float(best), "baseline": float(baseline), "last_loss": loss.item()}


if __name__ == "__main__":
    main()
