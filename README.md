# Prompt Engineering Toolkit & LLM Lab

Complete implementation and technical documentation for the Prompt Engineering Lab, covering LLM foundations, capabilities evaluation, transformer architecture, and advanced prompt engineering.

---

## 📂 Repository Structure & Tasks

- **Task 1: Models, Tokens, and Cost** (`src/task1/`)
  - Model comparison matrix, `tiktoken` token counting, monthly cost simulation, context window analysis, and production justifications (`model_comparison.md`).

- **Task 2: Capabilities & Safety Evaluation Pipeline** (`src/task2/`)
  - Automated testing pipeline using Gemini API covering summarisation, classification, extraction, translation, hallucination control, and multi-step logic, logged into `test_results.csv`.

- **Task 3: Transformer Concepts in Code** (`src/task3/`)
  - NumPy implementation of Single-Head & Multi-Head Self-Attention, sinusoidal positional encoding with saved visualization (`attention_matrix.png`), and Hugging Face pipelines (`bert` & `gpt2`).

- **Task 4: Advanced Prompt Engineering Toolkit** (`src/task4/`)
  - Groq API integration, golden dataset initialization, prompt versioning (v1-v3), Chain-of-Thought reasoning, temperature/top-p sampling controls, and Pydantic-validated structured outputs.

---

## 🚀 Setup & Execution

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt