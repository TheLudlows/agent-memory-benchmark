# LegalBench-RAG mini

This bundle comes from the [benchmark release by Nicholas Pipitone and Ghita Houir Alami](https://github.com/ZeroEntropy-AI/legalbenchrag#download) ([paper](https://arxiv.org/abs/2408.10343)). It contains the 714 legal documents used by AMB and 776 benchmark queries, 194 per source.

| Corpus source | Documents | Attribution | License |
| --- | ---: | --- | --- |
| [PrivacyQA](https://github.com/AbhilashaRavichander/PrivacyQA_EMNLP) | 7 | Ravichander et al., 2019 | [MIT](LICENSE-MIT.txt) |
| [ContractNLI](https://stanfordnlp.github.io/contract-nli/) | 95 | Koreeda and Manning, 2021 | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) |
| [MAUD](https://www.atticusprojectai.org/maud/) | 150 | The Atticus Project and Wang et al., 2023 | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) |
| [CUAD](https://www.atticusprojectai.org/cuad/) | 462 | The Atticus Project and Hendrycks et al., 2021 | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) |

The upstream LegalBench-RAG repository is published under MIT. Its notice and the PrivacyQA notice are included in [LICENSE-MIT.txt](LICENSE-MIT.txt). The source licenses apply to their respective corpus components.

AMB repackages the official archive as XZ-compressed JSON. It sorts documents by original relative path and uses neutral file names such as `lb-0001.txt`. The mini selection keeps all PrivacyQA questions and the first 194 from each other category after sorting by the seeded key in the upstream runner. It retains the query texts and character spans and maps the spans to the neutral document IDs. Text decoding and newline handling match the existing cache loader.

| File | SHA-256 |
| --- | --- |
| Original archive | `27431be37db9b1db23f8ab790a42d076adb1f72d7f9e7562e36a10573405f88d` |
| `mini.json.xz` | `b7f6370e901c642526d0268f8d24053b22ceaadd09434b2ad04361060c90310d` |
