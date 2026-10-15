# Arabic-Aware Support Ticket Classifier

**Author:** Bandar · **Course:** SDAIA Academy — AI Engineering Track, Applied NLP Engineering · **Use case:** 1 — Intelligent Support Ticket Assistant (intent classification + routing)
**Live demo:** [arabic-ticket-classifier-web.vercel.app](https://arabic-ticket-classifier-web.vercel.app/) · **Notebook:** [`notebooks/capstone_final.ipynb`](notebooks/capstone_final.ipynb)

## Problem
A support team receives customer messages in English, Modern Standard Arabic and Gulf dialect, and needs each one routed to the right queue. The course baseline works well on English but **fails on Arabic**: trained on English only, its confidence collapses on Arabic text and it abstains on most tickets.

**The system takes** a customer message **and produces** one of 27 intents, a confidence score, a route (auto / human review), a category and a priority, **measured by** macro-F1 on held-out English and Arabic test sets, sliced by language and dialect.

## Data
| Split | Size | Languages / slices | Source |
|---|---|---|---|
| train | 5,400 (+149 synthetic Arabic) | English (+ MSA, Gulf) | Course Bitext tickets; synthetic Arabic written with an LLM |
| validation | 1,080 EN + 54 AR dev | English; Arabic MSA + Gulf | Course splits; AR dev = 1 MSA + 1 Gulf ticket per intent |
| test | 1,080 EN + 54 AR | English; Arabic 27 MSA + 27 Gulf | Course splits; AR test = the other half of the course Arabic set |

- **Labels:** 27 intents (e.g. `track_order`, `payment_issue`, `delete_account`), mapped to 11 categories.
- **Known limitations:** English data is templated (near-duplicates between train and test inflate English scores); the Arabic test set is small (54 tickets → wide confidence intervals); the Arabic training data is synthetic.
- **Leakage prevention:** τ is chosen on validation/AR-dev only, never on test. Every synthetic Arabic ticket was compared with every Arabic dev/test ticket (e5 cosine similarity); **13 near-duplicates (cosine > 0.95) were removed** before training ([`data/synthetic_arabic_train.csv`](data/synthetic_arabic_train.csv), column `kept_after_leakage_check`).
- **Privacy:** no personal data — public Bitext data, the course Arabic set and synthetic tickets only.

## Approach
```
customer message
   └─► multilingual-e5-small (frozen encoder, 384-d embedding, "query: " prefix)
         └─► logistic regression (27 intents) ─► confidence ≥ τ ? auto-route : human review
               └─► business rules (course): category per intent, priority per intent + urgency words
```
Four versions, same encoder and classifier, compared on the **same** test sets:

| Version | What changes | τ |
|---|---|---|
| M0 · Baseline | Course baseline: English training data | 0.50 |
| M1 · Tuned τ | Same model, threshold chosen on validation | 0.25 |
| M2 · + Arabic | + 162 synthetic Arabic training tickets (MSA + Gulf) | 0.05 |
| **M3 · + Arabic (clean)** | + 149 synthetic Arabic tickets after removing near-duplicates | 0.05 |

**Why these models:** e5-small is multilingual (covers Arabic), small enough for CPU (~31 ms/ticket) and trained for sentence similarity. Logistic regression is fast, transparent and gives calibrated-enough probabilities for a confidence threshold. An encoder is the right tool for a decision task; an LLM would add latency and cost without being needed for classification.

## Results (held-out test set)
Macro-F1 with bootstrap 95% CIs (500 resamples) — full table: [`results/results_comparison.csv`](results/results_comparison.csv)

| System | EN (n=1,080) | AR (n=54) | AR · MSA | AR · Gulf | Arabic abstained |
|---|---|---|---|---|---|
| M0 · Baseline | 0.958 | 0.262 [0.122, 0.325] | 0.357 | 0.036 | 78% |
| M1 · Tuned τ | 0.996 | 0.398 [0.243, 0.470] | 0.423 | 0.274 | 37% |
| M2 · + Arabic | 0.996 | 0.711 [0.523, 0.790] | 0.753 | 0.568 | 0% |
| **M3 · + Arabic (clean)** | **0.996** | **0.733 [0.576, 0.802]** | **0.753** | **0.648** | 0% |

- Arabic macro-F1 **0.26 → 0.73** (Gulf 0.04 → 0.65) with no loss on English.
- Removing the 13 near-duplicates did **not** lower the score (M2 0.711 → M3 0.733, within CI): the gain comes from learning Arabic, not from memorizing test-like sentences.
- Even the lower CI bound of M3 (0.576) is far above the baseline (0.262).

### Model selection: which encoder?
Same training data as M3, same classifier, no τ; latency = encode + classify one Arabic ticket on Colab CPU — [`results/model_comparison.csv`](results/model_comparison.csv)

| Encoder | EN | AR [95% CI] | ms / ticket |
|---|---|---|---|
| TF-IDF char 2–4 grams (no embeddings) | 0.996 | 0.662 [0.482, 0.742] | 1.0 |
| paraphrase-multilingual-MiniLM-L12 | 0.994 | 0.752 [0.588, 0.829] | 30.3 |
| **multilingual-e5-small (ours)** | 0.996 | 0.733 [0.576, 0.802] | 31.3 |
| multilingual-e5-base (~2× larger) | 0.997 | 0.760 [0.586, 0.836] | 112.9 |

- English does not separate the encoders (templated data); **Arabic is the real test**.
- Embeddings beat TF-IDF on Arabic by 7–10 points.
- The three embedding encoders overlap within their CIs on 54 tickets; e5-base is ~3.6× slower for +0.027.
- We keep **e5-small**: chosen before seeing test results, comparable accuracy, CPU-friendly. Picking whichever encoder wins on the test set would be selecting on test data.

**Fine-tuning:** not part of this cohort's requirements (waived by the instructor). The bottleneck was missing Arabic training data, not model capacity: adding data with the same frozen encoder tripled Arabic macro-F1.

## Error analysis
All 17 M3 test errors were reviewed (13 Arabic of 54, 4 English of 1,080), using the cause list from Lab D3_4 — [`results/error_analysis_M3.csv`](results/error_analysis_M3.csv)

| Cause | Count (of 17) | Example | What we'd do |
|---|---|---|---|
| Missing coverage (dialect, typos, brands) | 7 | «انسحبت الفلوس مرتين على نفس الطلب عاجل» → `cancel_order` (Gulf verb unseen) | Collect real Gulf tickets; typo augmentation |
| Ambiguous | 6 | "i cannot find bill … can u send it" → `check_invoice` vs `get_invoice` | Merge or clarify near-duplicate intents; ask the customer |
| Label noise | 2 | «كيف أعيد تعيين كلمة المرور؟» → `edit_account` (a synthetic training label mentions passwords) | Review synthetic labels |
| Model error | 2 | «ما هي حالة طلب الاسترداد الخاص بي؟» → `check_cancellation_fee` | Stronger encoder / fine-tuning |
| Out of scope | 0 | — | — |

- All Arabic errors have low confidence (≤ 0.51): a higher τ would send them to a human instead of mislabeling them.
- Two English errors are **confident mistakes** (0.73, 0.78) that no threshold would catch.
- Errors were found on the test set, so they are **not** fixed by editing training data (that would tune on test data); they motivate the next steps below.

## Demo
**Web interface** in [`web/`](web/) — a static page (deployable on Vercel) where the encoder runs **in the visitor's browser** with Transformers.js (`Xenova/multilingual-e5-small`, 8-bit ONNX) and the trained logistic-regression weights are loaded from `web/model.json`. Type a message in English, MSA or Gulf dialect and get the top-3 intents, the route, the category and the priority; a slider changes τ live to show the automation-vs-safety trade-off. Because the browser uses a quantized encoder, scores can differ slightly from the evaluated Python model.

A Python/Gradio version of the same interface is in [`app/`](app/).

## Reproduce
1. Open [`notebooks/capstone_final.ipynb`](notebooks/capstone_final.ipynb) in Google Colab → **Runtime → Run all** (CPU; ~15 min). It clones the course repo, prepares the data, trains M0–M3, runs the encoder comparison and the error analysis, and saves `results/` plus `intent_clf_M3.joblib`.
2. Web demo: run `python web/export_model.py intent_clf_M3.joblib` to write `web/model.json` and serve `web/` as a static site (e.g. Vercel, root directory `web`).
3. Gradio demo: copy `intent_clf_M3.joblib` into `app/`, then
   ```bash
   cd app
   pip install -r requirements.txt
   python app.py
   ```
Encoder: `intfloat/multilingual-e5-small` (Hugging Face). Seed: 42. Course repo: [SDAIAAcademy/applied-nlp](https://github.com/SDAIAAcademy/applied-nlp).

## Limitations and next steps
- **Synthetic Arabic training data** and a **54-ticket Arabic test set**: results need confirming on real customer messages.
- With τ = 0.05 the model almost never abstains; in production τ should be raised to trade some automation for safety.
- Next with two more weeks: collect and label real Gulf-dialect tickets, add typo/dialect augmentation, merge ambiguous intent pairs, and try fine-tuning a multilingual encoder once real Arabic data exists.
