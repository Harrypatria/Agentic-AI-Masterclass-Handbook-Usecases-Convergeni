# Agentic AI: Building AI Agents and Retrieval Systems

**Code supplement to the handbook**
*A Masterclass in LLM Agents, RAG, and Production Deployment*

This folder is the official code supplement to the handbook above. It contains every runnable code example from all sixteen chapters, organised one folder per chapter and named by chapter number, so a reader can go straight to the chapter they are working through, copy the file they need, and run it. Each example ships as both a plain `.py` script and a `.ipynb` notebook with its real, executed output saved inline (see "Notebooks" below).

Every line of code in this folder was executed directly before it was written into the handbook itself, not merely described from memory of how it should behave. Where a chapter's own hands-on section builds on an earlier chapter (Chapter 14's SHAP walkthrough on Chapter 10's trained model, for instance), the dependency is resolved inline in the file itself, so each script here runs on its own, with no need to run other chapters' files first, unless a file's own docstring says otherwise.

## About the handbook

*Agentic AI: Building AI Agents and Retrieval Systems, A Masterclass in LLM Agents, RAG, and Production Deployment* is a sixteen-chapter, hands-on handbook covering the full stack a modern agentic AI engineer needs: prompt engineering, retrieval-augmented generation, single and multi-agent architectures, machine learning integration, multimodal agents and fine-tuning, production deployment, security, and the ITDO framework connecting a model's prediction to a genuine operational action. Sixteen worked case studies close the book, each mapped back to the specific chapter and technique it draws on.

First Edition, published in Glasgow, 2026, by Patria & Co., Artificial Intelligence Research Division ([www.patriaco.co.uk](https://www.patriaco.co.uk)). Published exclusively on Convergeni ([www.convergeni.com](https://www.convergeni.com)).

DOI: [https://doi.org/10.5281/zenodo.22049522](https://doi.org/10.5281/zenodo.22049522)

## About the authors

**Dr. Harry Patria, PMP®, PMI-CPMAI®** is CEO and Chief Data and AI Officer at Patria & Co., and Principal AI Software Engineer at i-Vigilant Technologies UK, building AI software solutions for major global oil companies. His academic path runs across seven degrees, including a Master of Science in Data Science from Newcastle University and a Master of Science in Business Analytics from Imperial College London, closing with two doctorates from Universitas Indonesia and the University of Strathclyde. He is co-founder of Convergence AI, a leading learning platform in Indonesia.

**Felix A. Setiadi, MSc.** is Vice President of Data and AI, a practitioner, consultant, and full-stack AI developer whose work spans oil and gas, coal, mining, palm oil, finance, and retail. He holds a Master of Science in Business Analytics from Imperial College London, completed with distinction, built on an undergraduate foundation in Industrial Engineering from the University of Hong Kong. He is co-founder of Convergence AI alongside Dr. Harry Patria.

The full biographies appear in the handbook's own "About the Authors" section.

## Folder structure

```
code_examples/
├── README.md                  (this file)
├── requirements.txt            (every package used across all sixteen chapters)
├── chapter_01/  Foundations of Artificial Intelligence and the Agentic Paradigm
├── chapter_02/  Prompt Engineering, Twenty Techniques from Zero-Shot to Optimisation
├── chapter_03/  Retrieval-Augmented Generation Foundations
├── chapter_04/  Advanced RAG Patterns, Evaluation, and Production Case Studies
├── chapter_05/  Programming Foundations for AI Engineering
├── chapter_06/  Building Your First AI Agent
├── chapter_07/  No-Code Agent Automation with n8n
├── chapter_08/  Advanced Agent Architectures, Multi-Agent Orchestration
├── chapter_09/  Reasoning Strategies, Human-in-the-Loop, and Observability
├── chapter_10/  Machine Learning Integration for Agentic Systems
├── chapter_11/  Multimodal Agents and Fine-Tuning with LoRA and QLoRA
├── chapter_12/  Deploying Agentic AI to Production
├── chapter_13/  Security, Cost, and CI/CD for Production AI Systems
├── chapter_14/  The ITDO Framework, From Prediction to Operational Action
├── chapter_15/  The Capstone Project, From Scoping to Demo Day
└── chapter_16/  Real-World Agent Patterns, Case Studies from the Field
```

Each chapter folder holds one `.py` file per hands-on section in that chapter, numbered in the order they appear in the book (`01_...py`, `02_...py`, and so on), plus a matching `.ipynb` notebook of the same name. A file's own module docstring names the exact section it was drawn from, states plainly whether it needs an API key, a GPU, or an extra package install, and flags anywhere its code was adapted (never its logic or its numbers, only its plumbing) to run standalone without a second terminal or an earlier chapter's script already having been run.

Chapters 10, 12, and 14 additionally bundle the real `diabetes.csv` dataset and a trained model file, so the machine-learning walkthrough, its FastAPI deployment, and its SHAP explainability follow-up all run immediately, with no separate download step.

## Notebooks

Every `.py` file also ships as a `.ipynb` notebook of the same name, split into the same numbered steps as separate cells, each preceded by a short markdown cell explaining that step's logic. Every notebook was actually executed, top to bottom, inside this chapter's own folder as its working directory (so the same relative dataset and model paths the `.py` files use resolve correctly), and its real output, printed values, tables, errors alike, is saved inline in the `.ipynb` file itself; opening one in Jupyter or VS Code shows genuine results, not blank cells.

A small number of cells cannot run to a real result in a clean checkout, and each says so plainly in its own markdown, rather than being silently skipped:

- Cells needing `OPENAI_API_KEY` / `ANTHROPIC_API_KEY` / `COHERE_API_KEY` that were not set at build time.
- **Chapter 11's QLoRA fine-tuning notebook**, which needs a CUDA-capable GPU and a multi-gigabyte model download; it is saved unexecuted with a markdown note, rather than attempting a download that would fail or hang on most machines.
- **Chapter 4's `raglite`-based file**, where the installed `raglite` release has moved `insert_document` to a plural, list-of-`Document`-object API since this section was written, and **Chapter 4's Ragas evaluation step**, where the current `ragas` release pulls in a `langchain-community` version with an unrelated missing import (`vertexai`) — both genuine upstream library version drift, documented in the file's own docstring, not a bug in the code.
- **Chapter 6's LangChain `AgentExecutor` file**, where current `langchain` releases (1.x) have moved `create_react_agent` to `langgraph.prebuilt` with an incompatible calling pattern, documented the same way.

To run and re-execute a notebook yourself with Jupyter installed:

```bash
jupyter notebook chapter_10/01_the_ai_health_copilot.ipynb
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Most files need nothing further. Where a file's own docstring says it needs `OPENAI_API_KEY` or `ANTHROPIC_API_KEY`, create a `.env` file in the same folder as that script. You may rename env example uploaded on the repository to be .env and put the API key accordingly:

```
OPENAI_API_KEY="Put your Open AI API Key here"
ANTHROPIC_API_KEY=
COHERE_API_KEY="Put your Cohere API Key here"
```

and load it before running, following the handbook's own Chapter 5 security discipline of never hardcoding a key into source:

```bash
pip install python-dotenv
python -c "from dotenv import load_dotenv; load_dotenv()" 2>/dev/null  # or add this line inside the script itself
```

A handful of files need more than `pip install`:

- **Chapter 11's QLoRA fine-tuning file** needs a CUDA-capable GPU; it will not run on a CPU-only machine, exactly as that chapter's own Technical Requirements line states.
- **Chapter 15's `01a_scoping_reading_exercise_not_runnable.py`** is a real excerpt quoted from another project for a reading exercise, not code meant to execute; its own docstring says so.
- **Chapter 3, 6, 8, and 11's `phidata`-based files**, **Chapter 4's `raglite`-based file**, and **Chapter 4's Ragas evaluation step** need the specific extra packages named in each file's own docstring, beyond the shared `requirements.txt`.

## How to cite

### Citing the handbook

> Patria, H., & Setiadi, F. A. (2026). *Agentic AI: Building AI agents and retrieval systems, a masterclass in LLM agents, RAG, and production deployment*. Patria & Co. https://doi.org/10.5281/zenodo.22049522

In-text citation: (Patria & Setiadi, 2026)

### Citing this code supplement

> Patria, H., & Setiadi, F. A. (2026). *Code supplement to Agentic AI: Building AI agents and retrieval systems* [Computer software]. Patria & Co. https://doi.org/10.5281/zenodo.22049522

When citing a specific chapter's code directly, name the chapter and file alongside the citation above, for example: "Chapter 10, `01_the_ai_health_copilot.py` (Patria & Setiadi, 2026)."

DOI: [https://doi.org/10.5281/zenodo.22049522](https://doi.org/10.5281/zenodo.22049522)

## Permitted use

This code supplement is offered under the same terms as the handbook's own Permitted Use and Citation page: readers may draw on it for research, publications, teaching, and social media content, provided the use stays consistent with its educational purpose and cites the handbook per the section above. It does not extend to repackaging this code, or the handbook itself, as a competing commercial product. For uses beyond this scope, contact the copyright holders directly.

## Publisher

**Patria & Co.**
Artificial Intelligence Research Division
[www.patriaco.co.uk](https://www.patriaco.co.uk)

**Exclusive distribution: Convergeni**
[www.convergeni.com](https://www.convergeni.com)
Head Office: Sovereign Plaza, 12th Floor, Jl. TB Simatupang Kav. 36, South Jakarta, 13730, Indonesia
Email: [contact@convergeni.com](mailto:contact@convergeni.com)

---

Copyright © 2026 Dr. Harry Patria, PMP®, PMI-CPMAI® and Felix A. Setiadi, MSc. All rights reserved.
