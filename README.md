# [UPDATE] RAGAS + Judge Model Discussion Added

## What was added:

### 1. Chapter 2 - Tinjauan Pustaka (Section 2.5)
**New subsection added:** `\subsubsection{Konsep Judge Model dalam Evaluasi RAG}`
- Explains the LLM-as-a-Judge mechanism
- Details the judge model's role and input/output
- Explains why independence matters (self-bias avoidance)
- Lists 3 criteria for judge model selection:
  1. Larger capacity than SLM being evaluated
  2. Independence from evaluated model
  3. Multilingual capability for Indonesian text

### 2. Chapter 3 - Metodologi Penelitian
**Section 3.6 (Metrik Evaluasi):**
- Added RAGAS explanation with LLM-as-a-Judge approach
- Specified **Qwen2.5-72B-Instruct** as the judge model
- Explained rationale: larger parameter count (72B vs 3B) but same model family for consistency
- Added citation for self-bias concern

**Section 3.7 (Prosedur Eksperimen):**
- Detailed judge model execution via API platforms (Together AI, Fireworks, OpenRouter)
- Specified evaluation criteria: faithfulness, answer relevancy, context precision/recall

**Section (Definisi Operasional Variabel):**
- Added Judge Model (Qwen2.5-72B-Instruct) as a control variable

## Next Steps:
- [ ] Add citation for `zheng2023judging` to `referensi.bib`
- [ ] Compile thesis to verify no LaTeX errors
- [ ] Review with supervisor
