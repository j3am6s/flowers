from pathlib import Path
import csv

DATA = Path("data/language-of-flowers.csv")
FLOWERS = Path("flowers")


def load():
    with DATA.open(newline="") as handle:
        return list(csv.DictReader(handle))


def base(name):
    return name.lower().split(",")[0].strip()


def meaning_for(name, rows):
    key = name.lower()
    for row in rows:
        if base(row["flower"]) == key:
            return row["meaning"]
    last = key.split()[-1]
    for row in rows:
        if base(row["flower"]) == last:
            return row["meaning"]
    return ""


def main():
    rows = load()
    folders = [path for path in sorted(FLOWERS.iterdir()) if (path / "names.csv").exists()]
    for folder in folders:
        with (folder / "names.csv").open(newline="") as handle:
            flowers = list(csv.DictReader(handle))
        out_rows = [
            (row["image"], row["name"], meaning_for(row["name"], rows))
            for row in flowers
        ]
        with (folder / "meaning.csv").open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["image", "name", "meaning"])
            writer.writerows(out_rows)
        print(folder.name)
        for image, name, meaning in out_rows:
            print(f"  {image}: {meaning}")


if __name__ == "__main__":
    main()
