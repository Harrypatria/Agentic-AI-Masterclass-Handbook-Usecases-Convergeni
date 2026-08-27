<div align="center">

# Agentic AI
### Building AI Agents and Retrieval Systems

**A Masterclass in LLM Agents, RAG, and Production Deployment**

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22049522.svg)](https://doi.org/10.5281/zenodo.22049522)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebooks-orange?logo=jupyter&logoColor=white)](https://jupyter.org/)
[![License](https://img.shields.io/badge/License-Educational%20Use-green)](#permitted-use)
[![Publisher](https://img.shields.io/badge/Publisher-Patria%20%26%20Co.-black)](https://www.patriaco.co.uk)

**Official code supplement** to the handbook  
*First Edition · Glasgow 2026 · Published exclusively on [Convergeni](https://www.convergeni.com)*

[Read the Handbook](https://www.convergeni.com) · [DOI](https://doi.org/10.5281/zenodo.22049522) · [Publisher](https://www.patriaco.co.uk)

</div>

---

## Overview

This repository is the **complete, production-ready code companion** to *Agentic AI: Building AI Agents and Retrieval Systems*.

Every example from all **sixteen chapters** is included — both as clean `.py` scripts and fully executed `.ipynb` notebooks with real outputs saved inline.  

All code was executed live before being written into the handbook. Dependencies between chapters are resolved inline, so every file runs independently (unless its docstring states otherwise).

---

## What’s Inside

| Chapter | Focus |
|---------|-------|
| **01** | Foundations of Artificial Intelligence and the Agentic Paradigm |
| **02** | Prompt Engineering — Twenty Techniques from Zero-Shot to Optimisation |
| **03** | Retrieval-Augmented Generation Foundations |
| **04** | Advanced RAG Patterns, Evaluation & Production Case Studies |
| **05** | Programming Foundations for AI Engineering |
| **06** | Building Your First AI Agent |
| **07** | No-Code Agent Automation with n8n |
| **08** | Advanced Agent Architectures & Multi-Agent Orchestration |
| **09** | Reasoning Strategies, Human-in-the-Loop & Observability |
| **10** | Machine Learning Integration for Agentic Systems |
| **11** | Multimodal Agents and Fine-Tuning with LoRA & QLoRA |
| **12** | Deploying Agentic AI to Production |
| **13** | Security, Cost & CI/CD for Production AI Systems |
| **14** | The ITDO Framework — From Prediction to Operational Action |
| **15** | Capstone Project — From Scoping to Demo Day |
| **16** | Real-World Agent Patterns & Field Case Studies |

Each chapter folder contains numbered scripts (`01_...py`, `02_...py`, …) and matching notebooks. Module docstrings document exact handbook sections, required API keys, GPU needs, and any adaptations made for standalone execution.

Chapters 10, 12 and 14 ship with the real `diabetes.csv` dataset and trained model so the ML walkthrough, FastAPI deployment and SHAP explainability run immediately.

---

## Notebooks

Every `.py` file is mirrored as a fully executed `.ipynb` notebook:

- Split into the same numbered steps as individual cells
- Preceded by concise markdown explanations
- Executed top-to-bottom inside the chapter folder (relative paths resolve correctly)
- Real printed output, tables and errors saved inline

**Known non-executable cells** (clearly marked in markdown):

- Cells requiring `OPENAI_API_KEY` / `ANTHROPIC_API_KEY` / `COHERE_API_KEY` when keys were not present at build time
- Chapter 11 QLoRA fine-tuning (requires CUDA GPU + multi-GB model download)
- Chapter 4 `raglite` and Ragas steps (upstream library API drift)
- Chapter 6 LangChain `AgentExecutor` (LangChain 1.x → LangGraph migration)

```bash
jupyter notebook chapter_10/01_the_ai_health_copilot.ipynb

Quick Start
Bashpython -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
Most files require nothing further.
API Keys
Where a file’s docstring indicates keys are needed, create a .env in the same folder (or rename the provided example):
envOPENAI_API_KEY="your-key-here"
ANTHROPIC_API_KEY=
COHERE_API_KEY="your-key-here"
Bashpip install python-dotenv
Load with python-dotenv (never hard-code keys — follow Chapter 5 security practice).
Special Requirements

Chapter 11 QLoRA → CUDA-capable GPU
Chapter 15 01a_... → Reading exercise only (not runnable)
Chapters 3, 4, 6, 8, 11 → Extra packages listed in each file’s docstring beyond requirements.txt


Citation
Handbook
bibtex@book{patria2026agentic,
  author    = {Patria, Harry and Setiadi, Felix A.},
  title     = {Agentic AI: Building AI Agents and Retrieval Systems},
  subtitle  = {A Masterclass in LLM Agents, RAG, and Production Deployment},
  year      = {2026},
  publisher = {Patria \& Co.},
  doi       = {10.5281/zenodo.22049522}
}
In-text: (Patria & Setiadi, 2026)
Code Supplement
Patria, H., & Setiadi, F. A. (2026). Code supplement to Agentic AI: Building AI agents and retrieval systems [Computer software]. Patria & Co. https://doi.org/10.5281/zenodo.22049522
When citing a specific file:

Chapter 10, 01_the_ai_health_copilot.py (Patria & Setiadi, 2026)

Authors
Dr. Harry Patria, PMP®, PMI-CPMAI®

CEO & Chief Data and AI Officer, Patria & Co.

Principal AI Software Engineer, i-Vigilant Technologies UK

Co-founder, Convergence AI
Felix A. Setiadi, MSc.

Vice President of Data and AI

Co-founder, Convergence AI

Full-stack AI developer across oil & gas, mining, finance and retail
Full biographies appear in the handbook.

Permitted Use
This code supplement is provided under the same terms as the handbook’s Permitted Use page:

Research, publications, teaching and social media content are welcome
Must remain consistent with educational purpose and cite the handbook
Does not extend to repackaging as a competing commercial product

For uses beyond this scope, contact the copyright holders.

Publisher
Patria & Co.

Artificial Intelligence Research Division

www.patriaco.co.uk
Exclusive distribution

Convergeni

Sovereign Plaza, 12th Floor, Jl. TB Simatupang Kav. 36

South Jakarta 13730, Indonesia
Email: contact@convergeni.com


Copyright © 2026

Dr. Harry Patria, PMP®, PMI-CPMAI® & Felix A. Setiadi, MSc.

All rights reserved.

```
