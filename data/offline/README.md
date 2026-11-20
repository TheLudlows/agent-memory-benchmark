# Bundled Workbench data

The Workbench and the corresponding CLI datasets read this bundle and the existing
repository exports in `data/beam`, `data/lifebench`, `data/longmemeval`, and `data/personamem`.

| Data | Source revision | Local format |
| --- | --- | --- |
| LoCoMo | [snap-research/locomo@3eb6f2c](https://github.com/snap-research/locomo/tree/3eb6f2c585f5e1699204e3c3bdf7adc5c28cb376) | Original `locomo10.json` in XZ |
| LongMemEval-S | [xiaowu0162/longmemeval-cleaned@98d7416](https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/tree/98d7416c24c778c2fee6e6f3006e7a073259d48f) | Existing questions and session exports, with repeat-session IDs and context reconstructed by the loader |
| BEAM 100K, 500K, 1M | [Mohammadta/BEAM@3205395](https://huggingface.co/datasets/Mohammadta/BEAM/tree/3205395e897e7318c7b094ef4e6047b9b82dbb03) | Existing questions and transcript exports, with a per-conversation index of document IDs, character lengths, timestamps, and context |
| BEAM 10M | [Mohammadta/BEAM-10M@9b20961](https://huggingface.co/datasets/Mohammadta/BEAM-10M/tree/9b2096193fe74e2837e4713e483351e19817773c) | Same format as the other BEAM splits |
| LifeBench 1.0 English | [1754955896/LifeBench@37be813](https://github.com/1754955896/LifeBench/blob/37be8131b602001b2662b6d02c6eccadad099182/life_bench_data/version1/locomo_format/our_en.json) | `queries.json.gz` and `documents.json.gz` in `data/lifebench/en` contain 2,003 questions and 3,605 daily session documents for 10 users |
| LifeBench 2.0 Chinese | [1754955896/LifeBench@b4adbe3](https://github.com/1754955896/LifeBench/tree/b4adbe3d0afa66efa4ab362e9d67160089f860dd/life_bench_data/version2/data) | One XZ-compressed JSON object per user, keyed by the original relative filenames used by the loader |
| PersonaMem v1 | [bowen-upenn/PersonaMem-v1@a8076d5](https://huggingface.co/datasets/bowen-upenn/PersonaMem-v1/tree/a8076d5608c93ba2a28983cd78aa99b01a163ae7) | Existing question and session exports, plus each question’s original turn cutoff and the start and omitted empty turns of each exported session |
| PrecisionMemBench | [tenurehq/precisionMemBench@b95d6ab](https://github.com/tenurehq/precisionmembench/tree/b95d6abb471c0d591c440172283dd74e7af000df/fixtures) | Original `beliefs.seed.json` and `retrieval.cases.json` |
| cl100k_base | [OpenAI `cl100k_base.tiktoken`](https://openaipublic.blob.core.windows.net/encodings/cl100k_base.tiktoken) | Original ranks in XZ, checked against the SHA-256 pinned by tiktoken |

The LifeBench 1.0 question labels follow the upstream [per-user English QA annotations](https://github.com/1754955896/LifeBench/tree/37be8131b602001b2662b6d02c6eccadad099182/life_bench_data/version1/data_en). Category 2 is nondeclarative memory (429 questions) and category 3 is temporal and knowledge updating (229). Older AMB exports interchanged those two labels.

Both LifeBench releases use the upstream license included at [lifebench-v2/LICENSE](lifebench-v2/LICENSE).

The BEAM index splits each conversation's export after its segments have been joined
with two newlines. It stores the current loader's document boundaries from the
original Parquet data. The joined conversation text was compared byte for byte
before building the index.

PersonaMem’s index interprets `end_index_in_shared_context` as the endpoint of a Python slice, including negative endpoints. The formatted sessions in all three splits match the original transcripts after the export’s whitespace normalization.
