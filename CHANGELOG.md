# CHANGELOG

## [2026-05-02] RAGAS + Judge Model Discussion

### Added

#### Chapter 2 — Tinjauan Pustaka (Section 2.5)
- **New subsection:** `\subsubsection{Konsep Judge Model dalam Evaluasi RAG}`
- Explains the LLM-as-a-Judge mechanism
- Details the judge model's role and input/output
- Explains why independence matters (self-bias avoidance)
- Lists 3 criteria for judge model selection:
  1. Larger capacity than SLM being evaluated
  2. Independence from evaluated model
  3. Multilingual capability for Indonesian text

#### Chapter 3 — Metodologi Penelitian

**Section 3.6 (Metrik Evaluasi):**
- Added RAGAS explanation with LLM-as-a-Judge approach
- Specified **Kimi K2.5** as the judge model
- Explained rationale: superior reasoning capacity and strong multilingual capability for Indonesian text
- Added citation for self-bias concern

**Section 3.7 (Prosedur Eksperimen):**
- Detailed judge model execution via Kimi API
- Specified evaluation criteria: faithfulness, answer relevancy, context precision/recall

**Section (Definisi Operasional Variabel):**
- Added Judge Model (Kimi K2.5) as a control variable

### Notes
- Judge model corrected from Qwen2.5-72B-Instruct → Kimi K2.5 based on discussion
- Rationale changed from "same model family consistency" → "independent model with superior reasoning"

### Todo
- [ ] Add citation for `zheng2023judging` to `referensi.bib`
- [ ] Compile thesis to verify no LaTeX errors
- [ ] Review with supervisor

---

## [2026-05-02] Thesis Structure Reorganization

### Changed
- Re-organized references in `referensi.bib`
- Re-organized Chapter 2 content flow
- Added workflows for Chapter 3 (Gantt chart, experiment design)

---

## [2026-05-01] Initial Setup

### Added
- Initial LaTeX project structure
- Chapter 1: Pendahuluan (background, problem statement, objectives)
- Chapter 2: Tinjauan Pustaka (early literature review)
- Chapter 3: Metodologi Penelitian (initial methodology)
- Basic LaTeX configuration (margins, fonts, spacing, headings)
