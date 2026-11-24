# ViDoRe V3 Markdown · Computer Science

This bundle is licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
The two OpenStax source PDFs supplied in the pinned release carry a © 2024 Rice University notice and a CC BY 4.0 license.

| Source textbook | Authors |
| --- | --- |
| [Introduction to Computer Science](https://openstax.org/details/books/introduction-computer-science) | Jean-Claude Franchitti |
| [Introduction to Python Programming](https://openstax.org/details/books/introduction-python-programming) | Udayan Das, Aubrey Lawson, Chris Mayfield, Narges Norouzi |

The [ViDoRe V3 corpus snapshot](https://huggingface.co/datasets/vidore/vidore_v3_computer_science/tree/d5cc75883d92e294f0c0fc2662551c9708a06ebc) supplies the page IDs and pre-existing OCR Markdown for both books.
The [ViDoRe V3 MTEB snapshot](https://huggingface.co/datasets/vidore/vidore_v3_computer_science_mteb_format/tree/fb7fb69f81f7db62790f40494124b8ad22b424ab) supplies the multilingual queries and relevance judgments.
The ViDoRe team distributes its annotations and relevance judgments under CC BY 4.0.

AMB repackages those fields as XZ-compressed JSON for offline use. It retains the page text, query text, source identifiers, ordering, and positive relevance grades. The evaluation reads from this bundle and omits pages with no searchable Markdown, as it does with the downloaded corpus.

The archive contains 1,360 pages and 1,290 questions, with 215 questions per language across six languages. The Markdown index includes 1,358 nonblank pages.

SHA-256 of `computer_science.json.xz`

```text
15af9197f9d35b69478cb51d94c85c44eca8d7b568e224a7ff94b95bd5651b79
```
