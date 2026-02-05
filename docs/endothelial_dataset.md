# Human Hemogenic Endothelium Multiome Dataset (GSE270141)

## Overview

- Paired scRNA-seq and scATAC-seq generated with the 10x Genomics Multiome assay.
- Hemogenic endothelium–derived populations cultured under hypoxia (4% O2) or normoxia.
- Enables comparison of chromatin and transcriptional responses under oxygen modulation.

## Inputs

- Raw GEO files from accession `GSE270141`.
- GEX matrices: barcodes, features, and count matrices (`.tsv.gz` or `.mtx.gz`).
- ATAC fragments: per-sample fragment files.
- Local staging directory: `data/endothelial/raw/`.

## Outputs

- Processed RNA: `data/endothelial/processed/combined_RNA_qc.h5ad`.
- Processed ATAC: `data/endothelial/processed/combined_ATAC_qc.h5ad`.

## Usage

### Data Access

The raw and processed data files are available from GEO accession
[GSE270141](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE270141).
Download the required files for your analysis and store them under
`data/endothelial/raw/`.

### Dataset Summary (this study)

| Field                    | Value                                                                                         |
| ------------------------ | --------------------------------------------------------------------------------------------- |
| Conditions used          | normoxia only                                                                                 |
| Cells (raw)              | 5,621                                                                                         |
| Cells (post-QC)          | 4,735                                                                                         |
| Genes (raw)              | 21,134                                                                                        |
| Genes (post-QC)          | 17,351                                                                                        |
| Peaks (post-QC)          | 396,920                                                                                       |
| QC applied in this study | Barcode intersection; RNA mito <15%; min genes/cells: 200/3 for RNA + ATAC; re-align after QC |
| Genes modeled            | Configurable; 1000 (100 genes fallback)                                                       |
| Annotation               | GENCODE v44                                                                                   |

## References

- Primary study: _Hematopoietic cells emerging from hemogenic endothelium exhibit lineage-specific oxidative stress responses_.
- GEO accession: [GSE270141](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE270141).
