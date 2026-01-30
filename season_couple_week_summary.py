import csv
import re
from collections import defaultdict


def main() -> None:
    path = r"C:\Users\zhaoh\Desktop\C\2026_MCM_Problem_C_Data.csv"

    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)

        week_cols: dict[int, list[int]] = defaultdict(list)  # week -> col indices
        week_re = re.compile(r"^week(\d+)_judge(\d+)_score$")
        for i, col in enumerate(header):
            m = week_re.match(col)
            if m:
                week = int(m.group(1))
                week_cols[week].append(i)

        season_idx = header.index("season")

        couples: dict[int, int] = defaultdict(int)
        week_exists: dict[int, dict[int, bool]] = defaultdict(lambda: defaultdict(bool))

        for row in reader:
            if not row:
                continue
            season = int(row[season_idx])
            couples[season] += 1

            # A week "exists" for a season if any judge score entry in that week is not N/A/blank
            for week, cols in week_cols.items():
                for ci in cols:
                    if ci >= len(row):
                        continue
                    v = row[ci].strip()
                    if v and v.upper() != "N/A":
                        week_exists[season][week] = True
                        break

    seasons = sorted(couples.keys())
    weeks_sorted = sorted(week_cols.keys())

    print("season,couples,weeks")
    for s in seasons:
        weeks = sum(1 for w in weeks_sorted if week_exists[s].get(w, False))
        print(f"{s},{couples[s]},{weeks}")


if __name__ == "__main__":
    main()

