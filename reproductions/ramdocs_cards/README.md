# RAMDocs reproducibility package

This is the RAMDocs 500-question **Plain RAG vs. original documents + extracted evidence (Ours)** comparison. Existing `src/` and the 72B setup are unchanged.

1. Download [RAMDocs_Reproducible_Package.zip](RAMDocs_Reproducible_Package.zip?raw=true) and extract it.
2. Start with `ramdocs_repro_bundle/README.txt` for independent saved-result replay and optional new inference.
3. Read [the English report](Conflict_Aware_RAG_Simple_Report.pdf) and [summary.csv](summary.csv).

The ZIP contains code, questions and annotations, saved evidence cards, original outputs, requests, CSVs, tokenizer files and Linux CPython 3.12 offline replay dependencies. It excludes approximately 15 GB of model weights; pinned download commands are included. Single-source diagnostics are outside this report.

Verified in an isolated fresh Python environment: all 500 prompts/parses and every score/bootstrap reproduced; both CSVs and summary matched byte-for-byte. Optional fresh GPU inference has not been tested on another machine.

| RAMDocs metric                                    | Plain RAG | Docs + cards (Ours) |
| ------------------------------------------------- | --------: | ------------------: |
| Official-compatible                               | 15.2%     | **24.8%**           |
| Strict answer-set match                           | 13.0%     | **17.6%**           |
| Labeled wrong-answer acceptance (lower is better) | 18.2%     | 26.8%               |

Previously observed data: exploratory comparison, not independent confirmation. The official string score can miss wrong-answer aliases. Generation caps/budget and increased wrong-answer acceptance remain limitations.
