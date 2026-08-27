<div align="center">

# Agentic AI
### Building AI Agents and Retrieval Systems

**A Masterclass in LLM Agents, RAG, and Production Deployment**

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22049522.svg)](https://doi.org/10.5281/zenodo.22049522)
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebooks-F37626?logo=jupyter&logoColor=white)](https://jupyter.org/)
[![License](https://img.shields.io/badge/License-Educational%20Use-2EA44F)](#permitted-use)
[![Publisher](https://img.shields.io/badge/Publisher-Patria%20%26%20Co.-111111)](https://www.patriaco.co.uk)

**Official code supplement** to the handbook  
*First Edition · Glasgow 2026 · Published exclusively on [Convergeni](https://convergeni.com)*

[**Purchase the Handbook**](https://convergeni.com/browsebooks) · [DOI](https://doi.org/10.5281/zenodo.22049522) · [Publisher](https://www.patriaco.co.uk)

</div>

---

## Get the Handbook

The complete handbook is available **exclusively** on the Convergeni platform:

**→ [https://convergeni.com/browsebooks](https://convergeni.com/browsebooks)**

Purchase and access the full text, worked case studies, and supporting materials directly from the official distribution platform. This code repository is the companion resource for readers of the published handbook.

---

## Overview

This repository is the complete, production-ready code companion to *Agentic AI: Building AI Agents and Retrieval Systems*.

It contains every runnable example from all **sixteen chapters**, organised one folder per chapter. Each example ships as both a plain `.py` script and a fully executed `.ipynb` notebook with real output saved inline.

Every line of code was executed live before it was written into the handbook. Where a chapter builds on earlier material, the dependency is resolved inside the file itself, so each script runs standalone unless its own docstring states otherwise.

---

## Chapter Structure

| # | Chapter |
|---|---------|
| 01 | Foundations of Artificial Intelligence and the Agentic Paradigm |
| 02 | Prompt Engineering — Twenty Techniques from Zero-Shot to Optimisation |
| 03 | Retrieval-Augmented Generation Foundations |
| 04 | Advanced RAG Patterns, Evaluation, and Production Case Studies |
| 05 | Programming Foundations for AI Engineering |
| 06 | Building Your First AI Agent |
| 07 | No-Code Agent Automation with n8n |
| 08 | Advanced Agent Architectures & Multi-Agent Orchestration |
| 09 | Reasoning Strategies, Human-in-the-Loop, and Observability |
| 10 | Machine Learning Integration for Agentic Systems |
| 11 | Multimodal Agents and Fine-Tuning with LoRA & QLoRA |
| 12 | Deploying Agentic AI to Production |
| 13 | Security, Cost, and CI/CD for Production AI Systems |
| 14 | The ITDO Framework — From Prediction to Operational Action |
| 15 | Capstone Project — From Scoping to Demo Day |
| 16 | Real-World Agent Patterns & Case Studies from the Field |


Each chapter folder holds numbered scripts (`01_...py`, `02_...py`, …) and matching notebooks. Module docstrings name the exact handbook section, list required API keys or GPU needs, and note any adaptations made for standalone execution.

Chapters 10, 12 and 14 include the real `diabetes.csv` dataset and a trained model file so the ML walkthrough, FastAPI deployment and SHAP explainability run immediately.

---

## Notebooks

Every `.py` file is mirrored as a fully executed `.ipynb` notebook:

- Split into the same numbered steps as separate cells  
- Preceded by short markdown cells explaining the logic  
- Executed top-to-bottom inside the chapter folder (relative paths resolve correctly)  
- Real printed values, tables and errors saved inline  

A small number of cells cannot produce real results in a clean checkout and state this clearly:

- Cells requiring `OPENAI_API_KEY`, `ANTHROPIC_API_KEY` or `COHERE_API_KEY` when keys were not set at build time  
- **Chapter 11** QLoRA fine-tuning (requires CUDA GPU + multi-gigabyte model download)  
- **Chapter 4** `raglite` insert API and Ragas evaluation (upstream library version drift)  
- **Chapter 6** LangChain `AgentExecutor` (moved to `langgraph.prebuilt` in LangChain 1.x)

```bash
jupyter notebook chapter_10/01_the_ai_health_copilot.ipynb
```
```Set up
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Citation
@book{patria2026agentic,
  author    = {Patria, Harry and Setiadi, Felix A.},
  title     = {Agentic AI: Building AI Agents and Retrieval Systems},
  subtitle  = {A Masterclass in LLM Agents, RAG, and Production Deployment},
  year      = {2026},
  publisher = {Patria \& Co.},
  doi       = {10.5281/zenodo.22049522},
  url       = {https://doi.org/10.5281/zenodo.22049522}
}

Authors
Dr. Harry Patria, PMP®, PMI-CPMAI®

CEO & Chief Data and AI Officer, Patria & Co.

Principal AI Software Engineer, i-Vigilant Technologies UK

Co-founder, Convergence AI
Seven degrees including MSc Data Science (Newcastle) and MSc Business Analytics (Imperial College London), plus dual doctorates from Universitas Indonesia and the University of Strathclyde.
Felix A. Setiadi, MSc.

Vice President of Data and AI

Co-founder, Convergence AI

Full-stack AI practitioner across oil & gas, mining, palm oil, finance and retail
MSc Business Analytics (Imperial College London, Distinction) and BEng Industrial Engineering (University of Hong Kong).
Full biographies appear in the handbook.

Permitted Use
This code supplement is offered under the same terms as the handbook’s Permitted Use page:

Readers may use it for research, publications, teaching and social media content
Use must remain consistent with its educational purpose and must cite the handbook
Does not extend to repackaging this code or the handbook as a competing commercial product

For any other use, contact the copyright holders directly.

Publisher
Patria & Co.

Artificial Intelligence Research Division

www.patriaco.co.uk
Exclusive distribution

Convergeni

Sovereign Plaza, 12th Floor

Jl. TB Simatupang Kav. 36

South Jakarta 13730, Indonesia
Purchase the handbook: https://convergeni.com/browsebooks
Email: contact@convergeni.com


Copyright © 2026

Dr. Harry Patria, PMP®, PMI-CPMAI® and Felix A. Setiadi, MSc.

All rights reserved.
