#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["requests", "pandas"]
# ///
"""Build the sample dataset for Notebook 3's "your own data" template.

arXiv abstracts labelled with the broad field of their primary category
(Astrophysics, Quantitative biology, Economics, Mathematics). The rows come from
librarian-bots/arxiv-metadata-snapshot, a Hugging Face mirror of arXiv's own
metadata snapshot, queried through the Hub's dataset-viewer /filter endpoint so
the 2.7 GB snapshot isn't downloaded. (The arXiv API itself works too, but rate
limits are strict: one request every 3 seconds, per network.)

arXiv's descriptive metadata, abstracts included, is CC0 1.0:
https://info.arxiv.org/help/api/tou.html

Outputs (in --out, default <repo>/data):
    arxiv_abstracts.csv        labelled rows: id, title, abstract, field
    arxiv_abstracts_new.csv    unlabelled rows for "apply to new rows": id, title, abstract
    arxiv_abstracts_meta.json  query, filters, dataset revision, counts
    arxiv_abstracts_README.md  how it was made, licence

Run (no GPU needed):
    uv run scripts/make_arxiv_dataset.py
"""
import argparse
import json
import random
import time
from datetime import date
from pathlib import Path

import pandas as pd
import requests

DATASET = "librarian-bots/arxiv-metadata-snapshot"
FILTER_URL = "https://datasets-server.huggingface.co/filter"

# Field name -> arXiv archive prefix. The first entry in `categories` is the primary category.
FIELDS = {
    "Astrophysics": "astro-ph",
    "Quantitative biology": "q-bio.",
    "Economics": "econ.",
    "Mathematics": "math.",   # the dot excludes math-ph (mathematical physics)
}


def field_of(category):
    for field, prefix in FIELDS.items():
        if category.startswith(prefix):
            return field
    return None


def fetch(where, n, page=100, retries=8):
    """Return up to n rows matching an SQL-style `where` clause."""
    rows = []
    for offset in range(0, n, page):
        params = {"dataset": DATASET, "config": "default", "split": "train",
                  "where": where, "offset": offset, "length": min(page, n - offset)}
        for attempt in range(retries):
            r = requests.get(FILTER_URL, params=params, timeout=120)
            if r.ok:
                break
            print(f"  retry {attempt + 1} ({r.status_code}): {r.text[:100]}")
            time.sleep(min(60, 10 * (attempt + 1)))   # the viewer index can take a minute to load
        r.raise_for_status()
        batch = [x["row"] for x in r.json()["rows"]]
        rows += batch
        if len(batch) < params["length"]:
            break
    return rows


def main():
    repo = Path(__file__).resolve().parent.parent
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--month", default="2503", help="arXiv YYMM to sample from (default March 2025)")
    ap.add_argument("--per-field", type=int, default=100, help="labelled rows per field")
    ap.add_argument("--new-per-field", type=int, default=5, help="unlabelled rows per field")
    ap.add_argument("--fetch", type=int, default=300, help="candidate rows to fetch per field")
    ap.add_argument("--min-chars", type=int, default=300)
    ap.add_argument("--max-chars", type=int, default=1500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=repo / "data")
    a = ap.parse_args()

    info = requests.get(f"https://huggingface.co/api/datasets/{DATASET}", timeout=60).json()
    revision = info.get("sha")

    rng = random.Random(a.seed)
    labelled, new, candidates = [], [], {}
    for field, prefix in FIELDS.items():
        where = f"\"categories\" LIKE '{prefix}%' AND \"id\" LIKE '{a.month}.%'"
        print(f"{field}: {where}")
        kept = []
        for row in fetch(where, a.fetch):
            cats = row["categories"].split()
            abstract = " ".join(row["abstract"].split())
            title = " ".join(row["title"].split())
            # Primary category in this field, and no cross-list to one of the other fields
            if field_of(cats[0]) != field:
                continue
            if any(field_of(c) not in (None, field) for c in cats[1:]):
                continue
            if not a.min_chars <= len(abstract) <= a.max_chars:
                continue
            kept.append({"id": row["id"], "title": title, "abstract": abstract, "field": field})
        candidates[field] = len(kept)
        need = a.per_field + a.new_per_field
        if len(kept) < need:
            raise SystemExit(f"Only {len(kept)} usable rows for {field}; raise --fetch or change --month")
        rng.shuffle(kept)
        labelled += kept[:a.per_field]
        new += kept[a.per_field:need]
        time.sleep(1)

    rng.shuffle(labelled)
    rng.shuffle(new)
    a.out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(labelled).to_csv(a.out / "arxiv_abstracts.csv", index=False)
    pd.DataFrame(new).drop(columns="field").to_csv(a.out / "arxiv_abstracts_new.csv", index=False)

    meta = {
        "source": DATASET, "source_revision": revision, "month": a.month,
        "fields": FIELDS, "label": "field of the primary (first-listed) category",
        "filters": {"no_cross_list_to_other_fields": True,
                    "abstract_chars": [a.min_chars, a.max_chars]},
        "candidates_after_filter": candidates, "seed": a.seed,
        "labelled": len(labelled), "new": len(new), "date": date.today().isoformat(),
    }
    (a.out / "arxiv_abstracts_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    (a.out / "arxiv_abstracts_README.md").write_text(README.format(
        n=len(labelled), per=a.per_field, n_new=len(new), month=a.month,
        lo=a.min_chars, hi=a.max_chars, dataset=DATASET, revision=revision,
        today=date.today().isoformat()))
    print(f"Wrote {len(labelled)} labelled + {len(new)} new rows to {a.out}")


README = """# arXiv abstracts by field

Sample dataset for the "your own data" template in Notebook 3: classify an abstract into its broad field.

- **Files:**
  - `arxiv_abstracts.csv` ({n} rows, {per} per field): `id`, `title`, `abstract`, `field`.
  - `arxiv_abstracts_new.csv` ({n_new} rows, no `field` column): used to show the trained model labelling new data.
- **Fields:** Astrophysics (`astro-ph*`), Quantitative biology (`q-bio.*`), Economics (`econ.*`), Mathematics (`math.*`).
  The label is the field of the paper's primary (first-listed) category.
- **Selection:** papers first submitted in {month} (arXiv IDs `{month}.*`), primary category in one of the four fields,
  not cross-listed to another of the four, abstract {lo}-{hi} characters. Whitespace in titles and abstracts is collapsed.
- **Source:** [`{dataset}`](https://huggingface.co/datasets/{dataset}) (revision `{revision}`),
  a mirror of arXiv's metadata snapshot.
- **Generated:** {today} by `scripts/make_arxiv_dataset.py`.

## Licence

arXiv's descriptive metadata, including titles and abstracts, is released under
[CC0 1.0](https://creativecommons.org/publicdomain/zero/1.0/)
(see [arXiv API Terms of Use](https://info.arxiv.org/help/api/tou.html)), so this dataset is CC0 too.
The papers themselves keep their own licences. Thank you to arXiv for use of its open access interoperability.
"""

if __name__ == "__main__":
    main()
