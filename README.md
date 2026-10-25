# Islet-cell transcriptomes, electrophysiology and the beta-cell exocytotic defect of type 2 diabetes

Analysis code for a study of human pancreatic islet cells profiled by Patch-seq (single-cell transcriptome and
patch-clamp electrophysiology of the same cell). The code asks two questions:

1. Which electrophysiological measures of alpha and beta cells (cell size, Na⁺ and Ca²⁺ currents, exocytosis) can be
   predicted from a cell's transcriptome in donors that were held out from training, including an external donor
   cohort? Classical models, neural networks and single-cell foundation-model embeddings (scGPT, Geneformer,
   Universal Cell Embeddings) are compared.
2. Is the smaller exocytosis of beta cells from donors with type 2 diabetes carried by their transcriptomes? Measured
   differences are conditioned on expression components, on internal and external type 2 diabetes signatures, on
   exocytosis and granule genes and on *ETV1*, with placebo and permutation null distributions and simulated carriers.

Evaluation always holds out whole donors, and inference resamples or permutes donors, the unit of biological replication.

## Data

All inputs are public. `python scripts/stages/00_download.py` downloads them into `data/raw/` and records MD5 checksums
in `metadata/download_md5.tsv`; `metadata/data_sources.tsv` lists every source.

| Resource | Content |
|---|---|
| HumanIslets.com | Paired single-cell expression and electrophysiology of alpha and beta cells; donor tables; islet perifusion |
| GEO GSE124742, GSE164875, GSE270484 | Raw Patch-seq count tables used to verify recovered counts |
| PanKbase | Donor records and summed profiles of the integrated droplet single-cell atlas |
| Bandesh et al. 2026, EMBO J (Dataset EV7) | Beta-cell differential expression between donors with and without type 2 diabetes |
| Dos Santos et al. 2025, JCI (supplement) | Donor table |
| Enrichr (via GSEApy) | GO Biological Process 2025 and MSigDB Hallmark 2020 gene sets |

## Environment

```bash
conda env create -f environment.yml          # main environment: islet_patchseq
conda env create -f environment_fm.yml       # foundation-model embeddings and fine-tuning: islet_patchseq_fm
conda env create -f environment_scvi.yml     # scVI comparison: islet_patchseq_scvi
```

Neural networks and foundation models need a CUDA GPU. Pretrained foundation-model weights are downloaded into
`data/models/` as described in `config/run.yaml`.

## Running

```bash
bash run_all.sh               # all stages in dependency order, then all figures (skips finished stages)
bash run_all.sh --force       # rerun everything
bash run_all.sh --skip=03d_embed,03f_finetune
```

Every tunable parameter, including random seeds, is in `config/run.yaml`. Each stage writes tidy tables to
`results/<stage>/` together with a `PROVENANCE.json` (input and output checksums, seed, configuration hash and
software versions). Figure scripts in `scripts/figures/` read only these tables and save their plotting inputs to
`figures/source_data/`.

| Stages | Content |
|---|---|
| 00–02 | Download; paired dataset (counts, electrophysiology, donors); variance within and between donors |
| 03–03h | Donor-held-out cross-validation of all model families; foundation-model embeddings and fine-tuning |
| 04–04j | Donor-reference tests, robustness, reliability references, training size, null calibration, coupling efficiency |
| 05–06d | Donor-level models; external donor cohort; calibration and transfer to droplet profiles |
| 08–08b | Gene programmes within and between donors |
| 09–09h | Type 2 diabetes: two-part models, sensitivity analyses, models trained without diabetes, signatures, conditioning on the transcriptome, null distributions, simulations, Ca²⁺ measurement error, exocytosis genes, glucose strata |
| 10–11 | Summary tables; islet insulin secretion of the same donors |

## Licence

MIT (see `LICENSE`).
