"""push_with_shape.py — push a Kaggle kernel with an explicit GPU type (machineShape).

Why this exists
---------------
The Kaggle CLI (through at least 1.7.x) cannot choose your GPU. `kaggle kernels push`
lets Kaggle assign one — and it frequently assigns a **Tesla P100**, which:

  * some competitions BAN at submit time (your scored run is rejected), and
  * current PyTorch builds no longer support (Pascal/sm_60 was dropped), so a CUDA
    kernel launch fails with "no kernel image is available for execution on the device".

Either way a scored GPU run fails — and there is no CLI flag to force a different card.

The fix
-------
Push through the REST endpoint directly, which DOES accept a `machineShape` field.
The submitted version then runs (and, for code competitions, re-runs at scoring time)
on the GPU you asked for.

Usage
-----
    python push_with_shape.py <kernel_folder> [machineShape]

    <kernel_folder>  folder containing kernel-metadata.json + the notebook it points to
    [machineShape]   NvidiaTeslaT4 (default) | NvidiaTeslaP100 | NvidiaL4 | NvidiaTeslaV100

Auth (either works)
-------------------
    * ~/.kaggle/kaggle.json   (the standard Kaggle credentials file), or
    * environment variables KAGGLE_USERNAME and KAGGLE_KEY

This file never contains your key — it is read from the file or env at runtime.

Author:  Matthew Mitchell — MMKPC Studios
Repo:    https://github.com/MMKPC/kaggle-machineshape-gpu
License: MIT
"""
__author__ = "Matthew Mitchell (MMKPC Studios)"
__url__ = "https://github.com/MMKPC/kaggle-machineshape-gpu"
__license__ = "MIT"

import json
import os
import sys

import requests

PUSH_URL = "https://www.kaggle.com/api/v1/kernels/push"


def load_auth():
    """Return (username, key) from env vars or ~/.kaggle/kaggle.json."""
    user, key = os.environ.get("KAGGLE_USERNAME"), os.environ.get("KAGGLE_KEY")
    if user and key:
        return user, key
    path = os.path.join(os.path.expanduser("~"), ".kaggle", "kaggle.json")
    with open(path) as f:
        creds = json.load(f)
    return creds["username"], creds["key"]


def build_body(folder, shape):
    """Assemble the /kernels/push payload from a standard kernel-metadata.json folder."""
    meta = json.load(open(os.path.join(folder, "kernel-metadata.json")))
    nb = json.load(open(os.path.join(folder, meta["code_file"]), encoding="utf-8"))

    # Kaggle wants each cell's source as a single string, with no stored outputs.
    for cell in nb.get("cells", []):
        if cell.get("cell_type") == "code":
            cell["outputs"] = []
        if isinstance(cell.get("source"), list):
            cell["source"] = "".join(cell["source"])

    return {
        "id": None,
        "slug": meta["id"],
        "newTitle": meta["title"],
        "text": json.dumps(nb),
        "language": meta["language"],
        "kernelType": meta["kernel_type"],
        "isPrivate": meta.get("is_private", True),
        "enableGpu": meta.get("enable_gpu", False),
        "enableTpu": False,
        "enableInternet": meta.get("enable_internet", False),
        "datasetDataSources": meta.get("dataset_sources", []),
        "competitionDataSources": meta.get("competition_sources", []),
        "kernelDataSources": meta.get("kernel_sources", []),
        "modelDataSources": meta.get("model_sources", []),
        "categoryIds": meta.get("keywords", []),
        "machineShape": shape,  # <-- the field the CLI can't set
    }


def main(argv):
    if len(argv) < 2:
        sys.exit("usage: python push_with_shape.py <kernel_folder> [machineShape]")
    folder = argv[1]
    shape = argv[2] if len(argv) > 2 else "NvidiaTeslaT4"

    resp = requests.post(PUSH_URL, json=build_body(folder, shape),
                         auth=load_auth(), timeout=120)
    print("status", resp.status_code)
    print(resp.text[:800])
    return 0 if resp.ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
