# AGENTS.md

Guidance for Codex, OpenAI coding agents, Claude Code, Cursor, and other AI assistants working in this repository.

## Project Summary

SeedLearn develops AI-assisted tools for identifying tropical tree seedlings from field-collected image sets. The project combines BioCLIP 2 visual embeddings, few-shot classifiers, vision-LLM morphological extraction, and literature RAG to evaluate family-, genus-, species-, and trait-level signals for tropical forest research.

## Repository Status

This is Nohemi's independent working repository for SeedLearn. It was initialized as a clean snapshot from the more complete development branch so collaborators can work without depending on Mitch's personal fork.

Treat external or historical repositories as references only unless Nohemi explicitly asks to push or modify them.

## Data Boundaries

- Do not commit raw images, embeddings, model weights, secrets, or large generated experiment outputs.
- The local `data` path may be a symlink to `/nfs/roberts/project/pi_lsc4/shared/seedlearn/data` on the Yale cluster.
- Verified BioCLIP 2 embeddings are expected at `data/embeddings/2026-01-29_v2026-01-29_12K/features.npz` with shape `(10407, 768)`.
- Human validation files may contain private or unpublished annotations. Review before publishing, copying, or broad sharing.
- Never add API keys to source files. Use environment variables such as `OPENAI_API_KEY`.

## Key Paths

- `src/seedlearn/clip/`: BioCLIP embedding and classification code.
- `scripts/extract_embeddings.py`: embedding extraction entrypoint.
- `scripts/run_simpleshot.py`: SimpleShot evaluation entrypoint.
- `src/seedlearn/pipeline/`: pipeline orchestration code.
- `trait_grading/`: trait grading and human/model comparison tools.
- `docs/status/`: current project status and consolidation notes.
- `docs/data.md`: data layout and NFS reference.
- `docs/clip/`: BioCLIP and embedding documentation.

## Workflow Guidance

- Prefer the packageized `src/seedlearn` code and documented scripts over older one-off scripts.
- For branch consolidation, compare old `main` and `dev` content carefully and copy only useful missing pieces.
- Before running heavy embedding, PCA, UMAP, or model jobs, use a compute node or Slurm job rather than a login node.
- For BioCLIP 2 baselines, start with family-level SimpleShot, then genus/species after the workflow is confirmed.
- Keep small summaries, figures, and reports in the repo when useful. Keep large artifacts on NFS and document their paths.

## Validation

- For code changes, run focused tests with `pytest` when available.
- For docs-only changes, tests are usually unnecessary, but check links and scan for secrets before committing.
- Before pushing, confirm that no raw data, large binaries, or credentials were added.
