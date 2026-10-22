from pathlib import Path
import csv
import json
import struct
import tarfile
import urllib.request
import zlib

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights

DATA = Path("data/oxford102")
FLOWERS = Path("flowers")
K = 15

IMAGES = "https://thor.robots.ox.ac.uk/flowers/102/102flowers.tgz"
LABELS = "https://thor.robots.ox.ac.uk/flowers/102/imagelabels.mat"
NAMES = "https://raw.githubusercontent.com/udacity/aipnd-project/master/cat_to_name.json"


def ensure_dataset():
    DATA.mkdir(parents=True, exist_ok=True)
    if not (DATA / "imagelabels.mat").exists():
        urllib.request.urlretrieve(LABELS, DATA / "imagelabels.mat")
    if not (DATA / "cat_to_name.json").exists():
        urllib.request.urlretrieve(NAMES, DATA / "cat_to_name.json")
    if not (DATA / "jpg").exists():
        archive = DATA / "102flowers.tgz"
        print("downloading the flower dataset (about 330 MB)")
        urllib.request.urlretrieve(IMAGES, archive)
        with tarfile.open(archive) as tar:
            tar.extractall(DATA, filter="data")
        archive.unlink()


def flower_names():
    data = (DATA / "imagelabels.mat").read_bytes()
    _, size = struct.unpack_from("<II", data, 128)
    raw = zlib.decompress(data[136 : 136 + size])
    payload = raw[8:]
    name_len = struct.unpack_from("<II", payload, 32)[1]
    data_off = 40 + ((name_len + 7) & ~7)
    data_len = struct.unpack_from("<II", payload, data_off)[1]
    ids = np.frombuffer(payload[data_off + 8 : data_off + 8 + data_len], dtype=np.uint8)
    names = json.loads((DATA / "cat_to_name.json").read_text())
    return np.array([names[str(int(i))] for i in ids])


def network():
    weights = ConvNeXt_Tiny_Weights.DEFAULT
    net = convnext_tiny(weights=weights)
    net.classifier[-1] = torch.nn.Identity()
    net.eval()
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    return net.to(device), weights.transforms(), device


def soften(image):
    height, width = image.shape[:2]
    mask = np.zeros((height, width), np.uint8)
    radius = (max(1, int(width * 0.46)), max(1, int(height * 0.46)))
    cv2.ellipse(mask, (width // 2, height // 2), radius, 0, 0, 360, 255, -1)
    blurred = cv2.GaussianBlur(image, (0, 0), max(height, width) / 12)
    softened = image.copy()
    softened[mask == 0] = blurred[mask == 0]
    return softened


def embed(images, net, transform, device):
    vectors = []
    with torch.inference_mode():
        for start in range(0, len(images), 32):
            batch = []
            for image in images[start : start + 32]:
                rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                batch.append(transform(Image.fromarray(rgb)))
            encoded = net(torch.stack(batch).to(device)).cpu().numpy()
            vectors.append(encoded)
            done = min(start + 32, len(images))
            if len(images) > 32 and (done == len(images) or done % 512 == 0):
                print(f"  {done}/{len(images)}", flush=True)
    vectors = np.concatenate(vectors).astype(np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-8
    return vectors


def dataset_vectors(net, transform, device):
    names = flower_names()
    cache = DATA / "vectors.npy"
    if cache.exists() and len(np.load(cache)) == len(names):
        return np.load(cache), names

    paths = [DATA / "jpg" / f"image_{i:05d}.jpg" for i in range(1, len(names) + 1)]
    print("reading dataset photos")
    images = [cv2.imread(str(path)) for path in paths]
    vectors = embed(images, net, transform, device)
    np.save(cache, vectors)
    return vectors, names


def knn(vectors, names, query):
    distances = ((vectors - query) ** 2).sum(axis=1)
    nearest = np.argpartition(distances, K)[:K]
    nearest = nearest[np.argsort(distances[nearest])]
    votes = {}
    for index in nearest:
        votes[names[index]] = votes.get(names[index], 0) + 1
    best = max(votes.values())
    for index in nearest:
        if votes[names[index]] == best:
            return str(names[index])


def describe(path, net, transform, device):
    image = cv2.imread(str(path))
    full, focused = embed([image, soften(image)], net, transform, device)
    query = full + focused
    return query / (np.linalg.norm(query) + 1e-8)


def identify(folder, vectors, names, net, transform, device):
    images = sorted(folder.glob("flower_*.jpg"))
    if not images:
        return
    print(folder.name)
    rows = []
    for image in images:
        name = knn(vectors, names, describe(image, net, transform, device))
        rows.append((image.name, name))
        print(f"  {image.name}: {name}")
    with (folder / "names.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["image", "name"])
        writer.writerows(rows)


def main():
    ensure_dataset()
    net, transform, device = network()
    vectors, names = dataset_vectors(net, transform, device)
    folders = [path for path in sorted(FLOWERS.iterdir()) if path.is_dir()]
    for folder in folders:
        identify(folder, vectors, names, net, transform, device)


if __name__ == "__main__":
    main()
