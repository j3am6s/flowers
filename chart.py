from pathlib import Path
import csv
from collections import Counter

import matplotlib.pyplot as plt

FLOWERS = Path("flowers")


def chart(folder):
    with (folder / "meaning.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    counts = Counter(row["meaning"] or "unknown" for row in rows)
    meanings = [meaning for meaning, _ in counts.most_common()]
    sizes = [counts[meaning] for meaning in meanings]
    total = sum(sizes)

    figure, axis = plt.subplots(figsize=(8, 5))
    wedges, _, _ = axis.pie(sizes, startangle=90, autopct=lambda share: f"{share:.0f}%")
    axis.legend(wedges, meanings, loc="center left", bbox_to_anchor=(1, 0.5))
    figure.savefig(folder / "meanings.png", bbox_inches="tight")
    plt.close(figure)

    print(folder.name)
    for meaning, count in counts.most_common():
        print(f"  {count / total:.0%}  {meaning}")


def main():
    folders = [path for path in sorted(FLOWERS.iterdir()) if (path / "meaning.csv").exists()]
    for folder in folders:
        chart(folder)


if __name__ == "__main__":
    main()
