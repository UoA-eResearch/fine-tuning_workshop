# arXiv abstracts by field

Sample dataset for the "your own data" template in Notebook 3: classify an abstract into its broad field.

- **Files:**
  - `arxiv_abstracts.csv` (400 rows, 100 per field): `id`, `title`, `abstract`, `field`.
  - `arxiv_abstracts_new.csv` (20 rows, no `field` column): used to show the trained model labelling new data.
- **Fields:** Astrophysics (`astro-ph*`), Quantitative biology (`q-bio.*`), Economics (`econ.*`), Mathematics (`math.*`).
  The label is the field of the paper's primary (first-listed) category.
- **Selection:** papers first submitted in 2503 (arXiv IDs `2503.*`), primary category in one of the four fields,
  not cross-listed to another of the four, abstract 300-1500 characters. Whitespace in titles and abstracts is collapsed.
- **Source:** [`librarian-bots/arxiv-metadata-snapshot`](https://huggingface.co/datasets/librarian-bots/arxiv-metadata-snapshot) (revision `c428226b689a0e17db2cd55265133b9eca15d97f`),
  a mirror of arXiv's metadata snapshot.
- **Generated:** 2026-11-13 by `scripts/make_arxiv_dataset.py`.

## Licence

arXiv's descriptive metadata, including titles and abstracts, is released under
[CC0 1.0](https://creativecommons.org/publicdomain/zero/1.0/)
(see [arXiv API Terms of Use](https://info.arxiv.org/help/api/tou.html)), so this dataset is CC0 too.
The papers themselves keep their own licences. Thank you to arXiv for use of its open access interoperability.
