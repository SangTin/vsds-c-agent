# RAG Experiment — Vietnamese Wikipedia Retrieval (and why we dropped it)

> Methodology log for the VSDS HackAIthon 2026 Bảng C submission. This documents
> a negative result reached by rigorous measurement: a 1.16M-chunk Wikipedia-VN
> RAG layer **lowered** accuracy on this dataset, and we removed it based on data
> rather than intuition.

## 1. Hypothesis

The base model (Qwen2.5-7B-Instruct Q4_K_M) scored **66.52%** on the public-test
leaderboard with no retrieval. Error analysis of a 10-question sample suggested
the model lacked specific knowledge on Vietnamese-domain factual questions
(history, law, Hồ Chí Minh thought, geography). Hypothesis: injecting relevant
Vietnamese Wikipedia passages as context would raise accuracy on the ~250
knowledge-dependent questions.

## 2. Setup

| Component | Choice |
|---|---|
| Corpus | Vietnamese Wikipedia (`wikimedia/wikipedia` 20231101.vi), 1,288,680 articles |
| Chunking | paragraph-boundary, ~1500 chars → **1,155,009 chunks** |
| Embedding | BGE-m3 dense, 1024-dim, L2-normalized, `max_length=512` |
| Index | FAISS `IndexFlatIP` (exact cosine), fp32, ~4.7 GB |
| Retrieval | top-3 chunks injected into the prompt as "Ngữ cảnh liên quan" |
| Gating | questions already containing a passage (`"Đoạn thông tin:"` or >500 chars) skip retrieval — 107/463 |
| Index build | RTX 3090, ~187 chunks/s, ~50 min |

No reranker in this iteration (top-3 by raw BGE-m3 cosine).

## 3. Evaluation method

Ground-truth labels are withheld by the organizers, so we use **GPT-5.5-thinking
predictions as a proxy reference** (estimated 85–90% accurate). The metric is
*agreement rate* = fraction of answers matching the proxy. This is a relative
signal, not absolute accuracy, but it lets us compare versions without spending
leaderboard submissions.

We also classify each changed answer as fixed / regressed / unclear relative to
the proxy.

## 4. Results

### 4.1 Headline

| Version | Agreement vs proxy | Leaderboard |
|---|---|---|
| v1 — no RAG | **70.41%** (326/463) | 66.52% (measured) |
| v3 — RAG (top-3, always on) | **65.01%** (301/463) | ~61% (estimated) |
| **Δ** | **−5.40 pp** | worse |

### 4.2 What RAG changed

RAG altered 99 of 463 answers:

- **22** fixed (proxy-correct after RAG)
- **47** regressed (proxy-correct before RAG, broken by it)
- **30** unclear (both differ from proxy)

Net effect strongly negative: regressions outnumber fixes 2-to-1.

### 4.3 Threshold sweep (the decisive test)

Rather than re-running the LLM, we logged the top-1 retrieval similarity per
question and simulated a relevance gate offline:
`answer = v3 if top_score ≥ T else v1`, sweeping T.

Non-passage similarity distribution (356 questions): min 0.484, p25 0.600,
p50 0.632, p75 0.673, p90 0.712, max 0.815.

| Threshold T | Agreement | vs v1 |
|---|---|---|
| 0.50 | 65.87% | −21 |
| 0.60 | 67.17% | −15 |
| 0.64 | 68.03% | −11 |
| 0.66 | 69.33% | −5 |
| 0.72 | 69.55% | −4 |
| 0.80+ | 70.41% | **±0** (RAG effectively off) |

**No threshold beats the no-RAG baseline.** The best achievable outcome is to
raise T high enough that RAG never fires, recovering exactly the v1 score. Every
time retrieval actually injects context — even at high similarity (0.7–0.8) — it
is net-harmful.

## 5. Why RAG hurt

The regressions span both factual-VN and math/calc questions. Root causes:

1. **The corpus rarely contains the answer.** Many questions target Vietnamese
   curriculum and current law (e.g. *Luật Bảo vệ môi trường 2020* article counts,
   Party regulations, Tư tưởng Hồ Chí Minh specifics) that general Wikipedia
   summarizes loosely or omits. BGE-m3 still returns its "most similar" chunk,
   which is topically close but answer-free.
2. **Distraction beats the prior.** Qwen2.5-7B already answers many of these
   correctly from parametric memory (66.52% with zero context). An off-target
   "relevant" passage pulls the model away from a correct guess.
3. **Retrieval is the wrong tool for reasoning/math.** Calculation and
   multi-step logic questions (a large share of the set) gain nothing from
   document lookup — they need computation, not context.

## 6. Decision

**Drop RAG for this dataset.** The measurement is unambiguous and threshold
tuning cannot rescue it. We pivot to **tool augmentation** (calculator + code
interpreter) for the calculation-heavy questions, which directly addresses the
observed error mode rather than layering on retrieval noise.

The viwiki index and the BGE-m3 / FAISS pipeline remain in the codebase (behind
the default-off `--rag` flag) so the experiment is reproducible and the decision
is auditable.

## 7. Artifacts

- `results/pred-v3-rag.csv` — RAG predictions (463)
- `results/scores.csv` — per-question top-1 retrieval similarity
- `src/rag/` — embedder, retriever, index builder (reproducible)
- Baseline `pred` (66.52%) is the current submission of record.
