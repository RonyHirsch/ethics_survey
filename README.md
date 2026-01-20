# The perceived role of consciousness in moral status attributions
This repo contains all codes for processing, analyzing, and visualizing data from our large-scale survey on public perceptions of consciousness and moral status [(Hirschhorn, Negro & Mudrik, 2026)](). 

The interactive website containing all survey questions and results can be found [here](https://ronyhirsch.github.io/minds-matter/?utm_source=github&utm_campaign=ethics_survey_readme). 

## Overview

The survey examined how the people perceive the relationship between consciousness and moral consideration across humans, non-human animals, and artificial systems. This repo includes:
- **Data preprocessing pipelines** processing raw responses from the online survey
- **Statistical analysis code** for all preregistered and exploratory analyses
- **Visualization functions** for manuscript figures and supplementary materials
- **Privacy-preserving data aggregation** for building the results website

## Repository Structure

### Core Analysis Pipeline

| Script | Description                                                                                                                                                                                              |
|--------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `survey_mapping.py` | Central configuration file containing all survey question mappings, response option definitions, column name constants, and entity metadata (icons, categories). Referenced by nearly all other scripts. |
| `process_survey.py` | Data preprocessing pipeline. Processes raw survey responses and splits the unified sample into exploratory (30%) and replication (70%) datasets using iterative stratification.                          |
| `analyze_survey.py` | Main analysis pipeline. Produces all descriptive statistics, preregistered analyses, and prepares data files for R mixed-effects modeling.                                                               |
| `helper_funcs.py` | Statistical and ML utilities. Includes the actual statistical analysis functions called by `analyze_survey.py` (e.g., random forest pipelines, χ<sup>2</sup> tests, MW U-tests, k-means clustering).     |
| `follow_up_analysis.py` | Processing and analysis pipeline for the follow-up study, structured similarly to `process_survey.py`.                                                                                                   |

### FDR Correction

| Script | Description                                                                                                                                                                                                                                          |
|--------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `treeBH.py` | Python implementation of the TreeBH hierarchical multiple testing correction procedure [(Bogomolov et al., 2021, *Biometrika*)](https://doi.org/10.1093/biomet/asaa086). Supports tree-structured hypotheses with Simes for computing parent-level p-values. |

### Visualization

| Script | Description |
|--------|-------------|
| `plotter.py` | General-purpose plotting utilities. Includes functions for histograms, scatter plots, pie charts, stacked bar plots, world maps, PCA visualizations, and cluster centroid plots. |
| `manuscript_figures.py` | Generates all publication-quality figures for the main manuscript and supplementary materials. Produces scatter plots with entity icons, demographic distributions, response pattern visualizations, and multi-panel figures. |

### Exploratory Thematic Tagging

| Script | Description                                                                                                                                                                                                                                                                                                                                                      |
|--------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `freetext_explorer.py` | Thematic coding of free-text responses about two specific questions probing what characterizes people/non-human animals with higher moral status (for people who thought some deserved higher moral status than others). Uses keyword dictionaries with support for stems and regex patterns to identify themes in open-ended (but mostly very short) responses. |


### Free-Text Analysis

| Script | Description |
|--------|-------------|
| `freetext_explorer.py` | Thematic coding of free-text responses about moral status determinants. Uses keyword dictionaries with support for stems and regex patterns to identify themes (consciousness, intelligence, suffering capacity, ecological role, etc.) in open-ended responses. |

### Interactive Website Data Generation

| Script | Description |
|--------|-------------|
| `survey_website.py` | Generates privacy-preserving aggregated JSON files for the [Minds Matter](https://ronyhirsch.github.io/minds-matter/?utm_source=github&utm_campaign=ethics_survey_readme) interactive website. Applies k-anonymity suppression to protect respondent privacy. |
| `cross_questions.py` | Generates cross-tabulation data for the website's cross-question feature. Produces pre-aggregated breakdowns that allow filtering responses by demographic and attitudinal variables without exposing individual-level data. |

---

## Script Dependencies

```
survey_mapping.py          ← Referenced by all scripts
        │
        ├─────────────────────┐──────────────────────┐
        │                     │                      │
        ▼                     │                      ▼
process_survey.py             │              survey_website.py
        │                     │                      │
        ▼                     ▼                      ▼
analyze_survey.py ──► follow_up_analysis.py  cross_questions.py
        │                                            │
        ├──► helper_funcs.py                         ▼
        │         │                          [Website JSON files]
        │         ▼
        │    plotter.py
        │
        ├──► treeBH.py
        │
        ▼
manuscript_figures.py
        │
        ▼
   plotter.py


freetext_explorer.py       ← Standalone thematic coding module
```

---


## Data Flow

1. **Raw Data** → `process_survey.py` → Cleaned, filtered, and split datasets
2. **Processed Data** → `analyze_survey.py` → Statistical results, CSVs for R modeling, intermediate data files
3. **Analysis Results** → `manuscript_figures.py` → Publication figures (SVG/PNG)
4. **Processed Data** → `survey_website.py` + `cross_questions.py` → Privacy-preserving JSON for interactive website

---


## Citation

Hirschhorn, R., Negro, N., & Mudrik, L. (2026). The perceived role of consciousness in moral status attributions.
[doi.org/10.17605/OSF.IO/XXXXX]()

---



