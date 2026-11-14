# Sources and scope

- Backbone: Qwen/Qwen2.5-7B-Instruct, revision `a09a35458c702b33eeacc393d103063234e8bc28`. Tokenizer files copied from that snapshot; model weights are downloaded separately with `python run.py download-model`.
- Original plain prompt: repository `SusrikMaverick/conflict_aware_RAG`, snapshot `5d70279f0357b5c6b67a0d84e2b2e2cbc655a932`, `src/eval.py`. The prompt is retained verbatim; our method appends extracted cards.
- RAMDocs: the same frozen 500-question prepared collection used throughout this project. Input documents and labels remain unchanged by this package.
- RGB: all 100 English fact records at revision `65ec39e40e7dc9abb50e9bf1b4f32be3f6f16615`; paired original/edited evidence and distractors follow the saved selection protocol. Original README and its CC BY-NC-SA 4.0 notice are included as `rgb_readme.md`.
- ConflictBank: `Warrieryes/CB_qa`, revision `b3b12e1dba7097151312c29c13e2a76f47d726c9`; deterministic 500-question sample from the full stream. This is a short-answer mixed-document adaptation, not the native multiple-choice benchmark score. See `selection_protocol.json`.
- Seven baseline adapters and upstream snapshots are included. Source locations, local adaptations and license availability are documented in `../BASELINES.md`; hashes are in `upstream_source_hashes.json`.
- The original RAMDocs source revision and raw-to-frozen preparation scripts were not recorded in this delivery. Frozen inputs are complete; reconstructing the original sample from upstream data is outside the verified scope.
