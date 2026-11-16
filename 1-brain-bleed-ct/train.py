"""Brain bleed detection on head CT slices (RSNA Intracranial Hemorrhage). Usage: see README.md."""
import argparse
import os

import cv2
import numpy as np
import pandas as pd
import pydicom
import torch
import torchvision
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit

SUBTYPES = ["any", "epidural", "intraparenchymal", "intraventricular", "subarachnoid", "subdural"]


def read_labels(csv):
    """One row per (image, subtype) -> one row per image with 6 columns."""
    df = pd.read_csv(csv)  # ID, Label
    df[["image", "subtype"]] = df["ID"].str.rsplit("_", n=1, expand=True)
    return df.drop_duplicates(["image", "subtype"]).pivot(index="image", columns="subtype", values="Label")[SUBTYPES]


def window(hu, level, width):
    lo, hi = level - width / 2, level + width / 2
    return (np.clip(hu, lo, hi) - lo) / (hi - lo)


def load_slice(path, size=256):
    # shortcut: assumes clean DICOM headers; a few RSNA files are corrupt or mis-scaled, skip or fix them for a full run
    d = pydicom.dcmread(path)
    hu = d.pixel_array.astype(np.float32) * float(d.RescaleSlope) + float(d.RescaleIntercept)  # raw values to HU
    # three windows as three channels: brain, blood, soft tissue
    img = np.stack([window(hu, 40, 80), window(hu, 80, 200), window(hu, 40, 380)], -1).astype(np.float32)
    return cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)


class Slices(torch.utils.data.Dataset):
    def __init__(self, labels, images, size):
        self.labels, self.images, self.size = labels, images, size

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        x = load_slice(f"{self.images}/{self.labels.index[i]}.dcm", self.size)
        return torch.from_numpy(x).permute(2, 0, 1), torch.tensor(self.labels.values[i], dtype=torch.float32)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True, help="stage_2_train.csv")
    p.add_argument("--images", required=True, help="folder holding ID_xxx.dcm")
    p.add_argument("--subset", type=int, help="use only N slices (start small)")
    p.add_argument("--size", type=int, default=256)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--no-pretrained", action="store_true")
    p.add_argument("--out", default="model.pt")
    a = p.parse_args(argv)

    labels = read_labels(a.csv)
    labels = labels[[os.path.exists(f"{a.images}/{i}.dcm") for i in labels.index]]  # only slices you downloaded
    labels = labels.sample(frac=1, random_state=0)[:a.subset]
    print(f"{len(labels)} slices; positives per subtype:\n{labels.sum()}")

    # split by patient, not by slice (PatientID is stored in every DICOM header)
    groups = np.array([pydicom.dcmread(f"{a.images}/{i}.dcm", stop_before_pixels=True).PatientID for i in labels.index])
    train_idx, val_idx = next(GroupShuffleSplit(test_size=0.15, random_state=0).split(labels, groups=groups))
    assert not set(groups[train_idx]) & set(groups[val_idx])
    train, val = labels.iloc[train_idx], labels.iloc[val_idx]

    load = lambda d, shuffle: torch.utils.data.DataLoader(
        Slices(d, a.images, a.size), batch_size=a.batch, shuffle=shuffle, num_workers=a.workers)
    train_dl, val_dl = load(train, True), load(val, False)

    dev = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    model = torchvision.models.resnet50(weights=None if a.no_pretrained else "IMAGENET1K_V2")
    model.fc = torch.nn.Linear(model.fc.in_features, 6)  # 6 yes/no outputs: any + 5 subtypes
    model.to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr)
    loss_fn = torch.nn.BCEWithLogitsLoss()  # sigmoid per output: a slice can have several bleed types

    for epoch in range(a.epochs):
        model.train()
        for x, y in train_dl:
            loss = loss_fn(model(x.to(dev)), y.to(dev))
            opt.zero_grad()
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            prob = torch.cat([torch.sigmoid(model(x.to(dev))).cpu() for x, _ in val_dl]).numpy()
        rows = []
        for j, name in enumerate(SUBTYPES):
            y = val[name].values
            both = len(set(y)) == 2  # AUC is undefined when a subtype has no positives (or no negatives) in validation
            rows.append({"subtype": name, "positives": int(y.sum()),
                         "log_loss": log_loss(y, prob[:, j], labels=[0, 1]),
                         "auc": roc_auc_score(y, prob[:, j]) if both else float("nan")})
        metrics = pd.DataFrame(rows).set_index("subtype")
        print(f"epoch {epoch + 1}: train loss {loss.item():.3f}\n{metrics.round(3)}")
    torch.save({"model": model.state_dict(), "size": a.size}, a.out)
    return {"metrics": metrics, "train_patients": set(groups[train_idx]), "val_patients": set(groups[val_idx])}


if __name__ == "__main__":
    main()
