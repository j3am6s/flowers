from pathlib import Path
import csv

import chart
import identify
import isolate
import meaning

BOUQUET = "bouquet4.jpg"


def write_meanings(folder):
    rows = meaning.load()
    with (folder / "names.csv").open(newline="") as handle:
        flowers = list(csv.DictReader(handle))
    out = [(row["image"], row["name"], meaning.meaning_for(row["name"], rows)) for row in flowers]
    with (folder / "meaning.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["image", "name", "meaning"])
        writer.writerows(out)
    for image, name, text in out:
        print(f"  {image}: {text}")


def main():
    photo = Path("test") / BOUQUET
    print("1. isolating flowers", flush=True)
    isolate.isolate(photo)
    print("2. identifying flowers", flush=True)
    folder = Path("flowers") / photo.stem
    identify.ensure_dataset()
    net, transform, device = identify.network()
    vectors, names = identify.dataset_vectors(net, transform, device)
    identify.identify(folder, vectors, names, net, transform, device)
    print("3. looking up meanings", flush=True)
    write_meanings(folder)
    print("4. drawing the pie chart", flush=True)
    chart.chart(folder)


if __name__ == "__main__":
    main()
