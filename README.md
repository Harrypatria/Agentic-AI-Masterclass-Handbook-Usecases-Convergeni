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
