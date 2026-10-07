"""Download EVOS recordings (Crain & Ulrich 2025, doi:10.20383/103.01538) into the current folder. No account needed.

Usage: python fetch.py                 # ground truth, calibration and licence for all 15 runs (about 8 MB)
       python fetch.py CC-T-NOM ...    # also the event recordings of the named runs (120-265 MB each)
Run names: [CC|CIRC|ROT]-[T|TR]-[NOM|DARK|SG], e.g. CIRC-TR-SG. The rotation runs are ROT-NOM, ROT-DARK, ROT-SG.
"""
import json, os, sys, urllib.parse, urllib.request

BASE, ROOT = "https://www.frdr-dfdr.ca/repo", "/1/published/publication_1533/submitted_data"
HEAD = {"User-Agent": "Mozilla/5.0", "X-Requested-With": "XMLHttpRequest"}


def get(url):
    return urllib.request.urlopen(urllib.request.Request(url, headers=HEAD))


def save(path, out):
    if not os.path.exists(out):
        os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
        with get(f"{BASE}/files{path}") as r, open(out, "wb") as f:
            while chunk := r.read(1 << 20):
                f.write(chunk)
        print("saved", out)


for name in ("README.txt", "LICENSE.txt", "CITATION.txt", "calib-intrinsics.csv", "calib-distortion.csv"):
    save(f"{ROOT}/{name}", name)
for folder in json.load(get(f"{BASE}/filesizecache?item_id=1533"))["contents"]:
    if folder["type"] != "dir":
        continue
    q = urllib.parse.urlencode({"item_id": 1533, "collection_id": 2, "endpoint_id": "f163c1b3-9c88-42f6-a7bb-5839ed6c4063", "path": folder["path"], "limit": 100, "offset": 0})
    for f in json.load(get(f"{BASE}/userfiles?{q}"))["files"]:
        if f["name"].endswith(".csv") or folder["name"] in sys.argv[1:]:
            save(f["name"], f"{folder['name']}/{os.path.basename(f['name'])}")
