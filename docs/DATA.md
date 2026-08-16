# Data and Provenance

## Included data

- Call-level primary reliability CSV and final audit tables.
- Qwen model-generalization call log and tables.
- Bridge, repeatability, and corrected deterministic-baseline CSVs.
- Controlled P2 reconciliation and counterfactual calls, prompt audits, manifests, environments, summaries, and decision dates.
- Derived WFCV performance, fold, fidelity, diagnostic, and out-of-sample files.
- Manuscript analysis tables and source-file checksums.

## Excluded data

- Third-party adjusted-close market-price downloads.
- Virtual environments and package caches.
- SQLite resume ledgers and transient lock files.
- Smoke-test and ad hoc live-test outputs.
- Duplicate JSONL logs where a canonical CSV is retained.
- Temporary LaTeX files and intermediate logs.

## Why market prices are excluded

Market prices were acquired from a third-party provider. This release does not grant redistribution rights for those downloads. Users performing an end-to-end WFCV rerun must obtain compatible adjusted-close data under the provider's current terms and preserve the ticker/date configuration in `configs/paper_protocol.json`.

## Integrity

`validate_release.py` recomputes all principal counts and rates directly from the retained CSV files. Checksums for every file in this package are listed in `SHA256SUMS.txt`.
