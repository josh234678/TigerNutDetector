import os
import time
import requests
from pathlib import Path
from ddgs import DDGS

SAVE_DIR = "dataset/bad_tiger_nut"
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
QUERIES = [
    "bad tiger nut damaged rotten",
    "moldy chufa nut spoiled",
    "shriveled dried tiger nut discolored",
    "damaged chufa tuber black spot",
]


def download_images(query, save_dir, start_index=0):
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    saved = 0
    print(f"  Searching: {query}")
    time.sleep(2)

    try:
        with DDGS() as ddgs:
            results = list(ddgs.images(query, max_results=60))
    except Exception as e:
        print(f"  Search failed: {e}")
        return 0

    for result in results:
        url = result.get("image", "")
        if not url:
            continue
        try:
            resp = requests.get(url, timeout=6, headers=HEADERS)
            if resp.status_code == 200 and "image" in resp.headers.get("Content-Type", ""):
                ext = url.split(".")[-1].split("?")[0].lower()
                if ext not in ("jpg", "jpeg", "png"):
                    ext = "jpg"
                fname = os.path.join(save_dir, f"{start_index + saved:04d}.{ext}")
                with open(fname, "wb") as f:
                    f.write(resp.content)
                saved += 1
                print(f"  [{start_index + saved}] downloaded", end="\r")
        except Exception:
            continue
        time.sleep(0.05)

    print(f"\n  Done: {saved} images for '{query}'")
    return saved


if __name__ == "__main__":
    print("=== Downloading Bad Tiger Nut Images ===\n")
    total = 0
    for query in QUERIES:
        count = download_images(query, SAVE_DIR, start_index=total)
        total += count
        time.sleep(3)

    print(f"\nTotal bad tiger nut images: {total}")
    print("Run python train.py to retrain with 3 classes.")
