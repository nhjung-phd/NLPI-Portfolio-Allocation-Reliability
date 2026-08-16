# GitHub upload guide

## Recommended repository name

`NLPI-Portfolio-Allocation-Reliability`

## Upload

Extract the release ZIP, open a terminal in the extracted repository directory, and run:

```bash
git init
git add .
git commit -m "Initial reproducibility release"
git branch -M main
git remote add origin https://github.com/YOUR-ACCOUNT/NLPI-Portfolio-Allocation-Reliability.git
git push -u origin main
```

Replace `YOUR-ACCOUNT` with the repository owner's GitHub account. Do not commit local virtual environments, Ollama model files, SQLite checkpoints, caches, or temporary logs; the included `.gitignore` excludes these items.

## Before making the repository public

1. Select and add a software license if reuse is intended.
2. Add the final repository URL to `CITATION.cff` as `repository-code`.
3. Add the accepted article DOI when available.
4. Create a GitHub release using the same version recorded in `CITATION.cff`.

GitHub displays the root `README.md` automatically. All headline findings can be checked without Ollama by running `python validate_release.py` and `./run_audit_cluster_bootstrap.sh`.
