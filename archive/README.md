# Legacy reference material

The old standalone replication suite is retained in `legacy_replication_suite/`, separate from the supported `pipeline/` interface. Its original files, generated outputs and duplicate local pretrained checkpoints were retained byte-for-byte. The duplicate checkpoints remain local and are ignored by Git; use the canonical published models in `models/weights/` for current inference.

`legacy_tools/generate_docs_and_agents.py` is a preserved reference copy of an old document generator. Its assumptions and some scientific claims are superseded. Do not run it to generate current reports. The original source file remains in `src/` for compatibility.

`organization_manifest_20261122.json` records every retained file's old/new path and SHA-256. This archive is historical context, not a second maintained pipeline. The project map is in `docs/PROJECT_STRUCTURE.md`.
