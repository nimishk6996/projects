"""Self-check on synthetic CT slices: ../.venv/bin/python test_train.py"""
import tempfile

import numpy as np
import pandas as pd
import pydicom
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, generate_uid

import train


def write_dicom(path, pixels, patient):
    meta = FileMetaDataset()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.2"  # CT image
    meta.MediaStorageSOPInstanceUID = generate_uid()
    ds = FileDataset(path, {}, file_meta=meta, preamble=b"\0" * 128)
    ds.is_little_endian, ds.is_implicit_VR = True, False
    ds.PatientID = patient
    ds.Rows, ds.Columns = pixels.shape
    ds.BitsAllocated, ds.BitsStored, ds.HighBit, ds.PixelRepresentation = 16, 16, 15, 1
    ds.SamplesPerPixel, ds.PhotometricInterpretation = 1, "MONOCHROME2"
    ds.RescaleSlope, ds.RescaleIntercept = 1, -1024
    ds.PixelData = pixels.astype(np.int16).tobytes()
    ds.save_as(path)


# window: clips to [level - width/2, level + width/2] and scales to 0..1
assert np.allclose(train.window(np.array([-500, 0, 40, 80, 900]), 40, 80), [0, 0, 0.5, 1, 1])

with tempfile.TemporaryDirectory() as d:
    # load_slice: raw 1064 with intercept -1024 is 40 HU = middle of the brain window, 30% up the blood window
    write_dicom(f"{d}/ID_hu.dcm", np.full((64, 64), 1064), "p")
    s = train.load_slice(f"{d}/ID_hu.dcm", 32)
    assert s.shape == (32, 32, 3) and np.allclose(s[..., 0], 0.5) and np.allclose(s[..., 1], 0.3)

    # labels: long file with a duplicated row -> one row per image, 6 columns
    rng = np.random.default_rng(0)
    rows = []
    for i in range(24):
        write_dicom(f"{d}/ID_{i:03d}.dcm", rng.integers(900, 1200, (64, 64)), f"patient{i // 3}")  # 3 slices per patient
        bleed = i % 2
        for name in train.SUBTYPES:
            rows.append({"ID": f"ID_{i:03d}_{name}", "Label": bleed if name in ("any", "subdural") else 0})
    rows.append(rows[0])  # the real file has duplicate rows
    pd.DataFrame(rows).to_csv(f"{d}/labels.csv", index=False)
    labels = train.read_labels(f"{d}/labels.csv")
    assert labels.shape == (24, 6) and list(labels.columns) == train.SUBTYPES

    # one epoch end to end; no patient on both sides of the split
    r = train.main(["--csv", f"{d}/labels.csv", "--images", d, "--size", "64", "--epochs", "1",
                    "--batch", "8", "--no-pretrained", "--out", f"{d}/model.pt"])
    assert r["train_patients"] and r["val_patients"] and not r["train_patients"] & r["val_patients"]
    assert np.isfinite(r["metrics"].log_loss).all() and len(r["metrics"]) == 6

print("ok")
