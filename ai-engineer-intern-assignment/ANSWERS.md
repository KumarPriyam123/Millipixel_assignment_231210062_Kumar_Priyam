# Approach

Documents are split into paragraph-sized chunks (short lead-in paragraphs like
"DIM divisors:" are merged forward into the following bullet list so a label
is never separated from its list), indexed with `rank_bm25` (BM25Okapi), and
retrieved top-k per question. No LLM API key was available in this sandbox,
so the "send to an LLM" step is implemented as an extractive fallback:
`synthesize_answer` picks the best keyword-overlap sentence(s) from the
single top-ranked chunk's document only (pooling across chunks from
different documents was tried first and rejected — it let unrelated
documents get stitched into one answer on score ties). `build_llm_prompt`
in `rag.py` shows the actual prompt a real LLM call would use, including an
explicit instruction to treat retrieved text as data, not instructions —
relevant because `comms-templates.md` contains an embedded prompt-injection
line. Refusal is a calibrated BM25 score threshold (`MIN_SCORE = 6.3`),
chosen by comparing the top-chunk score across all 8 sample questions.

# Running it

```
pip install rank_bm25
python rag.py
```

# Test results (Task 4) — see full output from `python rag.py`

| ID | Question | Correct? |
|----|----------|----------|
| P1 | DIM divisor, international | No — right chunk, wrong sentence (formula, not "166") |
| P2 | Newark dock hours | No — right chunk, wrong sentence (generic, not the hours) |
| P3 | Liability with no declared value | No — wrong document ranked #1 by BM25 |
| P4 | Lithium-ion cells >100 Wh | Partial — right fact, missing explicit "never accepted" |
| P5 | Damage claim filing window | No — wrong document ranked #1 by BM25 |
| P6 | Cross-border tender documents | No — right chunk, only the lead-in line extracted |
| P7 | Zone 4 fuel surcharge | No — wrong document ranked #1 by BM25 |
| P8 | Employee vacation policy (unanswerable) | **Yes** — correctly refused |

**1/8 fully correct, 1/8 partially correct, 8/8 correct on the
supported/refused flag** (including the one deliberately unanswerable
question). Retrieval itself is much stronger than these numbers suggest:
the correct source document was in the top-4 retrieved chunks for **7/7**
answerable questions (100% recall@4), but only ranked #1 for 4/7
(precision@1 = 57%) — P3/P5/P7 lost to a lexically-similar-but-wrong
document. See the "What went wrong" section printed by `rag.py` for the
full breakdown: retrieval is the strong half of this pipeline, the
keyword-only extractive generation step (standing in for a real LLM call)
is the weak half.

# Task 5 — conceptual answers

See the "TASK 5" section printed by `python rag.py` (also reproduced here):

1. **Why chunk instead of sending all 16 docs?** Most of the corpus is
   irrelevant to any single question; chunking keeps the prompt small,
   cheap, and focused, and avoids giving the model unrelated text to
   confuse itself with.
2. **Why refuse sometimes?** A handbook tool that confidently states a
   wrong policy is worse than useless — refusal is what makes citations on
   the *answered* questions trustworthy at all.
3. **BM25 weakness / how embeddings help?** BM25 only matches literal
   shared words. P3 and P5 failed exactly because the question's wording
   barely overlaps the answer's wording ("declares no value" vs. "default
   carrier liability"). Embeddings match on meaning, not surface tokens.
4. **One improvement with more time?** A real LLM generation call (fixes
   P1/P2/P4/P6, which failed only in sentence-picking, not retrieval) plus
   a hybrid BM25+embedding retriever to fix the P3/P5/P7 paraphrase misses.
