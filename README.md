# Provenance Guard

A backend API that analyzes text-based creative work and estimates whether it shows
stronger signals of **AI generation** or **human authorship** — while being explicit
that the result is *probabilistic evidence, not proof*.

> **This system does not verify identity, copyright ownership, or the complete
> creation process.** It evaluates the submitted text only. It should never be used
> as the sole basis for punishment, removal, grading, or copyright decisions.

---

## 1. Project Overview

Provenance Guard is a Flask service that takes a piece of writing, runs it through two
independent detection signals, combines them conservatively, and returns one of three
outcomes — `likely_ai`, `likely_human`, or `uncertain` — together with a plain-language
transparency label and a structured audit-log record.

**The problem.** As AI writing tools become ubiquitous, platforms, educators, and
publishers increasingly want to know whether a piece of text was machine-generated.
But there is no reliable, deterministic way to *prove* authorship from text alone.

**Why detection is probabilistic.** Both of the signals used here measure *correlates*
of machine generation, not generation itself. A large language model can recognize
stylistic patterns that often appear in AI output; structural heuristics can measure
regularity that AI text often exhibits. Neither observes the actual writing process.
The same surface patterns can be produced by a careful human writer, and AI text can be
edited until those patterns disappear. Detection therefore yields a *likelihood*, never
a fact.

**Why I focus on transparency and appeals.** Because the underlying judgment is
uncertain, the system is designed to *communicate that uncertainty honestly* rather than
to manufacture confidence. Every result ships with a label that states the assessment
may be wrong and that the creator can appeal. The appeal workflow gives a human a path
to contest an automated result.

**Why false positives are especially harmful.** Labeling genuine human work as
AI-generated can damage a creator's reputation, grade, or livelihood, and it accuses
someone of something they did not do. A false negative (missing AI text) is comparatively
low-cost. I treat a false positive as the more harmful error, so the architecture is
deliberately conservative: the `uncertain` band is wide, both signals must agree
*strongly* before a confident verdict is issued, and ambiguous cases default to
`uncertain`.

---

## 2. Features

All of the following are implemented in the repository:

| Feature | Where |
| --- | --- |
| `POST /submit` — validate, detect, score, label, persist, respond | `app.py` |
| Two-signal detection pipeline (Groq LLM + stylometric heuristics) | `detector.py`, `stylometry.py` |
| Reliability-adjusted confidence scoring with conservative rules | `scoring.py` |
| Three exact transparency labels | `labels.py` |
| `POST /appeal` — record a creator appeal, move content to `under_review` | `app.py`, `database.py` |
| Rate limiting (per-IP) on `/submit` and `/appeal` | `app.py` (Flask-Limiter) |
| Structured SQLite audit logging (submissions **and** appeals) | `database.py` |
| `GET /log` — recent audit entries, newest first, JSON-decoded | `app.py`, `database.py` |
| `GET /content/<content_id>` — current state of a submission | `app.py`, `database.py` |
| `GET /health` — liveness check | `app.py` |

---

## 3. Architecture

### Submission flow

1. **Request validation** — body must be a JSON object; `text` and `creator_id` are
   required; empty/whitespace text is rejected; `content_type` defaults to `"other"`.
2. **Rate limiting** — Flask-Limiter enforces the per-IP `/submit` limits *before* the
   handler runs.
3. **Content ID generation** — a UUID4 `content_id` is created.
4. **Groq signal** — `run_groq_signal(text, content_type)` (Signal 1).
5. **Stylometric signal** — `run_stylometric_signal(text, content_type)` (Signal 2).
6. **Reliability-adjusted combination** — `combine_signals(...)` weights each signal by
   its reliability.
7. **Uncertainty rules** — signal-disagreement, short-text, and missing-signal rules can
   force `uncertain` and cap confidence.
8. **Transparency-label selection** — `get_transparency_label(attribution, confidence)`.
9. **Content-record storage** — `save_content_record(...)` persists current state for
   `/content` and `/appeal`.
10. **Audit logging** — `log_submission(...)` writes a full structured event.
11. **JSON response** — attribution, likelihood, confidence, label, both signals, and
    uncertainty reasons.

```text
client ──POST /submit──> [validate] ──> [rate limit] ──> [content_id]
                                                              │
                       ┌──────────────────────────────────────┴───────────┐
                       ▼                                                    ▼
              Signal 1: Groq LLM                                 Signal 2: stylometry
              (ai_score, reliability, flags)                     (ai_score, reliability,
                       │                                          features, components)
                       └───────────────────┬────────────────────────────┘
                                            ▼
                          combine_signals()  ── reliability weighting
                                            ── signal-disagreement rule
                                            ── short-text rule
                                            ── missing-signal rule
                                            ▼
                          attribution + ai_likelihood + confidence
                                            ▼
                          get_transparency_label()
                                            ▼
                 save_content_record()  +  log_submission()
                                            ▼
                                     JSON response
```

### Appeal flow

1. A creator submits a `content_id` and `creator_reasoning` to `POST /appeal`.
2. The system retrieves the original content record (`get_content_record`).
3. It validates the appeal (existence, optional creator match, no duplicate).
4. It stores the creator's reasoning and optional evidence description (`create_appeal`).
5. It changes the content status to `under_review` (`update_content_status`).
6. It **preserves** the original attribution and confidence (no overwrite, no auto-reverse).
7. It logs the appeal event to the audit log (`log_event`).
8. It returns an HTTP 201 confirmation.

> The complete architecture diagrams (submission and appeal) live in
> [`planning.md`](planning.md) under **Architecture**; this README includes the compact
> version above and does not replace them.

---

## 4. Detection Signals

The system uses two signals that examine fundamentally different properties of the text,
so that each can act as a check on the other.

### Signal 1: Groq LLM assessment (`detector.py`)

**Model.** `llama-3.3-70b-versatile`, accessed through the Groq API. The API key is read
from `GROQ_API_KEY` in `.env` via `python-dotenv` and is never logged or returned.

**What it measures.** The model reads the whole text and judges it *holistically* —
coherence, voice consistency, generic or repetitive transitions, predictable
organization, overly balanced phrasing, personal specificity, and whether the writing
feels templated. The prompt includes the `content_type` so a poem is not judged with the
same expectations as a blog post.

**Output schema** (the exact dict `run_groq_signal` returns):

```python
{
    "ai_score": float,      # 0.0-1.0, clamped (0.0 = strongly human, 1.0 = strongly AI)
    "reliability": float,   # 0.0-1.0, clamped
    "flags": list[str],     # short observations from the model
    "signal": "groq",
}
```

The model is required to return strict JSON `{"ai_score","reliability","flags"}`; the code
strips Markdown code fences, recovers a JSON object from surrounding prose if needed, and
clamps both scores to `0.0-1.0`. If the API key is missing or the call/parse fails, it
raises a controlled `GroqSignalError` — it **never substitutes a fabricated score**.

**Why it is useful.** A capable LLM can interpret meaning, genre, and voice in ways that
pure statistics cannot. That is why it carries the larger weight in scoring.

**What it cannot reliably detect, and why it is not proof.** The model recognizes
*patterns associated with* AI text; it does not observe how the text was actually
written. Documented blind spots (see `planning.md` → *Signal 1 → Blind spots*):

- **Polished human writing** — formal, well-edited prose can read as "templated."
- **Heavily edited AI writing** — revision removes the tells the model looks for.
- **Non-native English writing** — unusual grammar or repeated structures reflect
  language background, not generation.
- **Short content** — too little signal for a confident judgment.
- **Mixed human/AI authorship** — one document-level score cannot separate sections.
- **Genre effects** — the model may lean on topic or genre stereotypes.

### Signal 2: Stylometric heuristics (`stylometry.py`)

This signal uses only the Python standard library — no second model, no external API. It
measures *structure*, not meaning, which is exactly why it is independent of Signal 1.

**Features actually computed** (`run_stylometric_signal` → `features`):

- `word_count`
- `sentence_count`
- `average_sentence_length`
- `sentence_length_variance`
- `sentence_length_coefficient_variation`
- `type_token_ratio`
- `punctuation_density`
- `paragraph_length_variance`
- `repeated_opener_rate`

**From features to a score.** Features are reduced to five component scores
(`0.0-1.0`, higher = more AI-like regularity): `sentence_uniformity`,
`paragraph_uniformity`, `repetition_score`, `punctuation_regularity`,
`lexical_uniformity`. These are combined with the fixed weights from `planning.md`:

```text
ai_score = 0.30 × sentence_uniformity
         + 0.20 × paragraph_uniformity
         + 0.20 × repetition_score
         + 0.15 × punctuation_regularity
         + 0.15 × lexical_uniformity
```

The core intuition: machine text tends to be *regular* — similar sentence lengths,
even paragraphs, reused openers, steady punctuation. Sentence uniformity, for example, is
derived from the coefficient of variation of sentence lengths: very low variation → high
uniformity → more AI-like. For `content_type == "poem"`, the repetition component is
damped (multiplied by `0.5`) because deliberate repetition is a legitimate human poetic
technique rather than evidence of generation.

**Reliability depends on text length** (short texts are structurally unstable):

| Word count | Reliability |
| --- | ---: |
| fewer than 50 | 0.20 |
| 50–99 | 0.50 |
| 100–249 | 0.75 |
| 250 or more | 1.00 |

**Actual stylometry output** (real result from `run_stylometric_signal` on a deliberately
uniform 72-word passage — no API involved):

```text
ai_score = 0.8561   reliability = 0.50   word_count = 72
components: sentence_uniformity=1.0, paragraph_uniformity=0.5,
            repetition_score=0.874, punctuation_regularity=1.0,
            lexical_uniformity=0.875
```

**Why poetry and short text mislead it.** A poem with repeated lines and limited
vocabulary looks structurally identical to highly regular AI text, so the heuristics can
read it as AI even though the uniformity is intentional. Short text simply does not
contain enough sentences/paragraphs for stable statistics, which is why its reliability is
only `0.20` and the scoring layer forces `uncertain` below 50 words.

**Why the two signals complement each other.** Signal 1 understands *what the text means
and how it reads*; Signal 2 measures *how the text is structured* without any semantic
understanding. They fail in different situations — the LLM can be fooled by genre or
polish, stylometry by repetition or length — so requiring them to *agree* before issuing a
confident verdict is much safer than trusting either alone. When they disagree strongly,
the system treats that as a reason for `uncertain`.

---

## 5. Confidence Scoring and Uncertainty

All scoring lives in `scoring.combine_signals(groq_result, stylometric_result)`.

**Base weights** (the LLM is weighted more because it interprets meaning):

```text
GROQ_BASE_WEIGHT  = 0.65
STYLE_BASE_WEIGHT = 0.35
```

**Reliability adjustment** — each weight is scaled by that signal's reliability:

```text
effective_groq_weight  = 0.65 × groq_reliability
effective_style_weight = 0.35 × stylometric_reliability
```

**Combined AI likelihood** (weighted, reliability-adjusted average, clamped `0.0-1.0`):

```text
ai_likelihood =
    (groq_score × effective_groq_weight + style_score × effective_style_weight)
    / (effective_groq_weight + effective_style_weight)
```

**Confidence** — how strongly the likelihood leans away from the midpoint:

```python
confidence = max(ai_likelihood, 1 - ai_likelihood)
```

**Conservative rules:**

- **Signal-disagreement rule** — if `abs(groq_score - style_score) >= 0.35`, force
  `uncertain` and record `signal_disagreement`.
- **Short-text rule** — if `word_count < 50`, force `uncertain` and cap confidence at
  `0.69`, recording `insufficient_length`.
- **Missing-signal behavior** — if exactly one signal is available, force `uncertain` and
  cap confidence at `0.60` (`missing_signal`); the surviving signal's likelihood is still
  surfaced, and no score is fabricated for the missing one. If **both** signals are
  missing, `combine_signals` raises a controlled `ScoringError` and `/submit` returns
  HTTP 503.

**Exact attribution thresholds** (from `scoring.py`, matching `planning.md`):

| Attribution | Condition |
| --- | --- |
| `likely_ai` | `ai_likelihood >= 0.85` **and** `signal_gap < 0.35` **and** `word_count >= 50` **and** both signals succeeded |
| `likely_human` | `ai_likelihood <= 0.20` **and** `signal_gap < 0.35` **and** `word_count >= 50` **and** both signals succeeded |
| `uncertain` | everything else |

**What a score near 0.60 means to a non-technical user.** It is a *weak lean*, not a
verdict. A combined likelihood of ~0.60 sits firmly inside the uncertain band: the system
is saying "there is a slight tilt in one direction, but not nearly enough to make a
claim." It should be read as "we don't know," not as "probably AI."

**Why the uncertain range is deliberately wide.** Confident `likely_ai` requires
`>= 0.85` and `likely_human` requires `<= 0.20`, leaving the entire `0.20–0.85` middle as
`uncertain`. That width is intentional: because falsely labeling human work as AI is the
costlier error, I would rather return "uncertain" too often than accuse a real creator on
thin evidence.

### Actual score examples

The two examples below are the deterministic scoring cases exercised in
`tests/test_scoring.py`. The **input** Groq and stylometric scores are the fixtures used
in that test file (both with reliability `1.0`, `word_count = 200`); the **combined AI
likelihood, confidence, and attribution** are the *actual outputs* produced by running
`scoring.combine_signals` on those inputs. (A live end-to-end Groq submission has not been
captured — see **Remaining Runtime Evidence**.)

| Example | Groq score | Stylometric score | Combined AI likelihood | Confidence | Attribution |
| --- | ---: | ---: | ---: | ---: | --- |
| High-confidence example | 0.94 | 0.89 | 0.9225 | 0.9225 | `likely_ai` |
| Lower-confidence example | 0.67 | 0.54 | 0.6245 | 0.6245 | `uncertain` |

**Why the scores differ.** In the high-confidence example both signals point strongly to
AI and agree closely (`signal_gap = 0.05`); the reliability-adjusted average lands at
`0.9225`, clearing the `0.85` threshold, so the result is `likely_ai` with high confidence.
Groq contributes most because it carries the larger weight (0.65 vs 0.35), but stylometry
*reinforces* rather than contradicts it. In the lower-confidence example both signals are
only moderately AI-leaning and still agree directionally (`signal_gap = 0.13`), but the
combined `0.6245` falls inside the wide uncertain band, so despite no rule being triggered
the system declines to make a confident claim — exactly the conservative behavior the
thresholds are designed to produce.

---

## 6. Transparency Labels

The exact text is centralized in `labels.py` and returned verbatim in the `/submit` and
`/content` responses (in both the `transparency_label` and backward-compatible `label`
fields). The label is chosen by `get_transparency_label(attribution, confidence)`.

| Result | Required condition | Exact text shown to users |
| --- | --- | --- |
| High-confidence AI | `attribution == "likely_ai"` **and** `confidence >= 0.85` | This content shows strong signals of AI generation. This automated assessment is not proof of authorship and may be incorrect. The creator can appeal this result and provide additional context or evidence. |
| High-confidence human | `attribution == "likely_human"` **and** `confidence >= 0.80` | This content shows strong signals of human authorship. This automated assessment is probabilistic and does not verify the creator’s identity, ownership, or full writing process. |
| Uncertain | `attribution == "uncertain"`, inconsistent attribution/confidence combinations, unknown attribution values, missing-signal cases, short-text forced-uncertain cases, and signal-disagreement cases | The system could not confidently determine whether this content was human-written or AI-generated. No definitive attribution is being made, and the content should not be penalized based on this result. |

**UX goals.** The wording is chosen to *avoid presenting detection as certainty*, to state
plainly that *the automated assessment may be wrong*, to *make uncertainty understandable*
in non-technical language, to *avoid implying automatic punishment*, and to *make the
appeal route visible* (the AI label explicitly mentions appeals). The function always
returns exactly one of these three strings.

---

## 7. Appeals Workflow

**Who can appeal (in this course implementation).** Authorization is represented by the
`creator_id` from the original submission. If a `creator_id` is supplied with the appeal,
it must match the original record; a production system would require authenticated
accounts. (See `planning.md` → *Appeals Workflow*.)

**Required request fields:** `content_id`, `creator_reasoning`.
**Optional fields:** `creator_id`, `evidence_description`.

**Validation rules (`POST /appeal`):**

1. Body must be a JSON object (else `400`).
2. `content_id` must be present and non-empty (else `400`).
3. `creator_reasoning` must be present and non-empty (else `400`).
4. `creator_reasoning` must be `<= 5000` characters (else `400`).
5. The content record must exist (else `404`).
6. If `creator_id` is provided, it must match the original record (else `403`).
7. If an appeal already exists for the content, or the content is already
   `under_review`, the request is rejected (else `409`).

**On success:** a UUID4 `appeal_id` is generated; the reasoning and optional evidence
description are stored; the **original attribution and confidence are preserved** on the
appeal record; the content status changes to `under_review`; a structured appeal event is
written to the audit log; and the endpoint returns HTTP 201.

**Appeals do not trigger automatic reclassification.** The original decision is never
overwritten or auto-reversed. Moving to `under_review` simply flags the item for a human.

**What a human reviewer would inspect** (per `planning.md`): the original attribution, AI
likelihood, confidence, both signal scores, signal disagreement, stylometric features, the
transparency label shown, the creator's reasoning, the evidence description, and current
status.

**Example request** (use a real `content_id` returned by `/submit`):

```json
{
  "content_id": "<content_id-from-submit>",
  "creator_reasoning": "I wrote this myself from personal experience.",
  "creator_id": "<original-creator-id>",
  "evidence_description": "I can provide dated drafts and version history."
}
```

**Success response (HTTP 201):**

```json
{
  "appeal_id": "<generated-uuid>",
  "content_id": "<content_id>",
  "status": "under_review",
  "message": "Your appeal has been recorded and the content is now under review."
}
```

---

## 8. Rate Limiting

Configured with Flask-Limiter, `storage_uri="memory://"`, keyed by client IP via
`get_remote_address`. The values below are verified against `app.py`:

| Endpoint | Limit (decorator) |
| --- | --- |
| `POST /submit` | `10 per minute;100 per day` |
| `POST /appeal` | `3 per hour;10 per day` |
| `GET /health`, `GET /log`, `GET /content/<id>` | no limit (no restrictive default is set) |

Exceeding a limit returns HTTP **429** via a JSON error handler — never an HTML page or a
500.

**Why these limits are realistic.** A legitimate creator is very unlikely to submit more
than ten works in a minute, so the per-minute cap mostly slows automated flooding while
leaving room for normal testing. Rapid repeated requests are far more consistent with
abuse than with genuine use. Appeals should be much rarer than submissions, so they get
stricter limits (3/hour), which also discourages duplicate or spammy appeal attempts on
the same content.

### Rate-Limit Test Evidence

Actual status codes observed from sending 12 consecutive `POST /submit` requests with the
limiter enabled (detection mocked, so no API calls; captured by running the application's
test client):

```text
12x POST /submit status codes: [200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 429, 429]
```

The first ten succeed; the 11th and 12th are throttled with HTTP 429.

---

## 9. Audit Log

**Why structured logging matters.** Every automated decision that can affect a creator
should be inspectable after the fact — for debugging, for accountability, and so a human
reviewing an appeal can see exactly what the system decided and why.

**Submission event fields** (current schema, `database.AUDIT_COLUMNS`):
`event_type`, `content_id`, `creator_id`, `timestamp`, `content_hash`, `content_type`,
`groq_ai_score`, `groq_reliability`, `groq_flags`, `stylometric_ai_score`,
`stylometric_reliability`, `stylometric_features`, `stylometric_component_scores`,
`signal_gap`, `combined_ai_score`, `confidence`, `attribution`, `transparency_label`,
`status`, `uncertainty_reasons`.

**Appeal event fields** (same table; appeal-specific columns):
`event_type` (`"appeal_submitted"`), `appeal_id`, `content_id`, `creator_id`,
`creator_reasoning`, `evidence_description`, `original_attribution`,
`original_confidence`, `status` (`"under_review"`), `timestamp`.

**Privacy.** Raw submitted creative text is **never** stored or exposed through `/log`. A
SHA-256 `content_hash` is stored instead, which supports traceability (you can confirm two
submissions are identical) without retaining the work itself. `GET /log` decodes JSON
columns (`groq_flags`, `stylometric_features`, `stylometric_component_scores`,
`uncertainty_reasons`) back into real arrays/objects and returns entries newest-first.

> **Production note:** `/log` is intentionally open for the course demo. In production it
> would require administrator authentication.

### Audit-Log Sample

The three entries below are the **actual rows currently stored in
`provenance_guard.db`**, shown as `GET /log` would return them (signal flags decoded).
They are genuine live-Groq runs from 2026-06-30. **Note:** these predate the
Milestone 4/5 two-signal schema migration — they were written by the earlier
single-signal endpoint, so they use `llm_score`/`llm_reliability`/`signal_flags` and do
not contain stylometric or appeal fields.

```json
{
  "entries": [
    {
      "content_id": "3b9f6057-48bd-4da0-9e83-8ea552140273",
      "creator_id": "test-user-3",
      "timestamp": "2026-06-30T13:41:57.776055+00:00",
      "content_hash": "7feaf6eff76bf941b30ba7a86cc1adbcd24bd79f6b77ccfd95e3078abdb31951",
      "content_type": "blog_post",
      "attribution": "likely_ai",
      "confidence": 0.7,
      "llm_score": 0.7,
      "llm_reliability": 0.6,
      "signal_flags": ["generic_transitions", "predictable_organization", "overly_balanced_phrasing", "lack_of_personal_specificity"],
      "status": "classified"
    },
    {
      "content_id": "a5e9a77f-4dde-4628-9d72-d70195509fc3",
      "creator_id": "test-user-2",
      "timestamp": "2026-06-30T13:41:39.030408+00:00",
      "content_hash": "df2d95149fa0281fc29ee7d7a3fe845061e8f6be084eb1ab317bf8c18e696c82",
      "content_type": "poem",
      "attribution": "likely_human",
      "confidence": 0.8,
      "llm_score": 0.2,
      "llm_reliability": 0.6,
      "signal_flags": ["coherent imagery", "personal specificity", "unpredictable transition"],
      "status": "classified"
    },
    {
      "content_id": "22bb67c9-496d-4194-af9c-3cfd02dbd9d1",
      "creator_id": "test-user-1",
      "timestamp": "2026-06-30T13:41:18.148134+00:00",
      "content_hash": "00b12bf56ad6135dc90d831b8c8aa0c344d377f1b371fd2fa767eebceaa9a4fa",
      "content_type": "blog_post",
      "attribution": "likely_human",
      "confidence": 0.8,
      "llm_score": 0.2,
      "llm_reliability": 0.6,
      "signal_flags": ["coherent_description", "personal_perspective", "sensory_details", "no_repetitive_transitions"],
      "status": "classified"
    }
  ]
}
```

The sample above covers three real attribution decisions but contains **no appeal event**,
because no appeal has been run against this database. A current-schema capture that
includes a two-signal submission plus an `appeal_submitted` event still needs to be
recorded:

```text
[PASTE AT LEAST 1 ACTUAL appeal_submitted GET /log ENTRY (CURRENT M5 SCHEMA) HERE]
```

---

## 10. API Reference

All responses are JSON. Status codes below are the ones actually implemented in `app.py`.

### `GET /health`
Liveness check. **200** → `{"status": "ok"}`.

### `POST /submit`
**Required:** `text` (string), `creator_id` (string).
**Optional:** `content_type` (one of `poem`, `short_story`, `blog_post`, `other`;
defaults to `"other"`).

**Success — 200:**
```json
{
  "content_id": "...",
  "attribution": "likely_ai | likely_human | uncertain",
  "ai_likelihood": 0.0,
  "confidence": 0.0,
  "transparency_label": "...",
  "label": "... (same exact text as transparency_label)",
  "status": "classified",
  "signals": {
    "groq": { "ai_score": 0.0, "reliability": 0.0, "flags": [] },
    "stylometry": { "ai_score": 0.0, "reliability": 0.0, "features": {}, "component_scores": {} },
    "signal_gap": 0.0
  },
  "uncertainty": { "forced": false, "reasons": [] }
}
```

**Errors:** `400` (body not JSON, missing/empty/non-string `text`, missing/empty
`creator_id`); `413` (text longer than 20,000 characters); `503` (both detection signals
failed — returns `status: "detection_error"`, no attribution); `429` (rate limit
exceeded).

### `POST /appeal`
**Required:** `content_id`, `creator_reasoning`. **Optional:** `creator_id`,
`evidence_description`.

**Success — 201:**
```json
{
  "appeal_id": "...",
  "content_id": "...",
  "status": "under_review",
  "message": "Your appeal has been recorded and the content is now under review."
}
```

**Errors:** `400` (body not JSON, missing `content_id`, missing/empty `creator_reasoning`,
reasoning longer than 5,000 characters); `404` (unknown `content_id`); `403`
(`creator_id` provided but does not match); `409` (appeal already exists / already
`under_review`); `429` (rate limit exceeded).

### `GET /content/<content_id>`
Current state of a submission.

**Success — 200:**
```json
{
  "content_id": "...",
  "attribution": "...",
  "ai_likelihood": 0.0,
  "confidence": 0.0,
  "transparency_label": "...",
  "status": "classified | under_review",
  "has_appeal": true
}
```
**Error:** `404` for an unknown content ID.

### `GET /log`
Returns `{"entries": [...]}` — recent audit events, newest first, JSON fields decoded.
**Security note:** open in this demo; would require administrator authentication in
production. Raw text and the API key are never present.

---

## 11. Testing

**Tests never consume Groq API calls.** Every test that exercises a detection path mocks
`run_groq_signal` (and, for deterministic scoring, `run_stylometric_signal`) at the `app`
module level, and uses a temporary SQLite database (`PROVENANCE_DB_PATH`) so the
development database is never touched.

**Coverage (by test file):**

- **Input validation** — `tests/test_milestone3.py`, `tests/test_milestone4.py` (400/413,
  missing/empty fields).
- **Groq handling** — `tests/test_milestone3.py`, `tests/test_milestone4.py` (controlled
  failure, no fabricated scores, JSON/fence parsing covered by the detector's own logic).
- **Stylometric calculations** — `tests/test_stylometry.py` (empty input, short/long/
  uniform/irregular text, poem repetition damping, reliability-by-word-count, no
  divide-by-zero, sentence parsing).
- **Scoring thresholds** — `tests/test_scoring.py` (likely_ai / likely_human / uncertain
  cases, confidence formula, reliability weighting).
- **Signal disagreement** — `tests/test_scoring.py`, `tests/test_milestone4.py`.
- **Short-text uncertainty** — `tests/test_scoring.py`, `tests/test_milestone4.py`.
- **Transparency labels** — `tests/test_labels.py` (all three variants, thresholds,
  inconsistent combos, verbatim text).
- **Appeals** — `tests/test_appeals.py` (valid appeal, 400/403/404/409, status change,
  appeal event logged, original decision preserved).
- **Rate limiting** — `tests/test_rate_limits.py` (429 JSON for `/submit` and `/appeal`,
  GET endpoints unthrottled).
- **Audit logging** — `tests/test_milestone3.py`, `tests/test_milestone4.py`,
  `tests/test_appeals.py` (both signal scores, combined score, no raw text/key).

**Run the suite:**

```powershell
python -m pytest -v
```

Actual result observed in this repository (`python -m pytest -q`):

```text
75 passed in 1.86s
```

---

## 12. Known Limitations

Each limitation follows directly from how the signals work.

**Repetitive human poetry.** A human poem may use repeated lines, limited vocabulary, and
consistent rhythm. Stylometry can read that uniformity as AI-like even though it is an
intentional creative technique. (Mitigated, not solved, by damping the repetition
component for `content_type="poem"` and forcing `uncertain` on short text.)

**Highly polished human prose.** Professional or academic writing can have uniform
paragraphs and smooth transitions. Both the LLM ("templated") and stylometry ("regular")
may associate that polish with AI generation.

**Heavily edited AI text.** Human revision removes many of the structural patterns
associated with generated text, so both signals can lean human or uncertain. The system
evaluates the submitted text, not its hidden creation history.

**Short submissions.** Very short text lacks enough sentences/paragraphs for stable
stylometric statistics and gives the LLM too little to judge — which is why text under 50
words is forced to `uncertain`.

**Non-native English writing.** Unusual grammar or repeated sentence structures may
reflect a writer's language background rather than AI generation, and can be
misinterpreted by both signals.

**Mixed human and AI authorship.** A single document-level score cannot represent which
sections were written or edited by different sources; such documents usually land in
`uncertain`.

**AI detection remains an unsolved problem.** Provenance Guard should not be used as the
sole basis for punishment, removal, grading, or copyright decisions. It is decision
*support* with explicit uncertainty and an appeal path, not a verdict.

---

## 13. What I Would Change for Production

This project is **not production-ready**. Concrete improvements I would make:

- **Authenticated users** instead of trusting a free-text `creator_id`.
- **Administrator protection for `/log`** (it currently exposes audit data to anyone).
- **Persistent, shared rate-limit storage** such as Redis (the current `memory://` store
  is per-process and resets on restart).
- **Encrypted storage** for audit data and stronger privacy controls around hashes and
  metadata.
- **A real human-review queue** with an **appeal-resolution endpoint**
  (`appeal_approved` / `appeal_denied` / `needs_more_information`).
- **Score calibration on a representative labeled dataset** — the current thresholds and
  feature normalizations are reasoned defaults, not empirically calibrated. Real
  calibration requires labeled evaluation data, not intuition alone.
- **Versioned models and prompts**, plus **monitoring for drift** as model behavior
  changes over time.
- **Multiple languages** and **paragraph-level analysis** to handle mixed authorship.
- **Better provenance evidence** — revision history, editor telemetry, or cryptographic
  content credentials — which would be far stronger than text-only heuristics.

---

## 14. Spec Reflection

**How the spec helped.** Writing `planning.md` first — with exact function signatures,
output schemas, score ranges, thresholds, and verbatim label text — meant every module
was built against a fixed contract. For example, `run_groq_signal` and
`run_stylometric_signal` were specified to return dicts with `ai_score`/`reliability`
before either was written, so when `scoring.combine_signals` was implemented it consumed
those fields directly with no schema mismatch. Defining the three label strings verbatim
in the spec also let `tests/test_labels.py` assert exact text rather than paraphrase.

**An actual divergence.** The **missing-signal failure behavior changed between
milestones**. In the Milestone 3 design, a failed Groq call returned HTTP 503. After the
second signal was integrated (Milestone 4), I changed this: if *one* signal fails but the
other succeeds, `/submit` now returns a graceful `uncertain` result (confidence capped at
0.60) instead of a 503, and only returns 503 when **both** signals fail. This diverged
from the original single-signal behavior because, with two independent signals, a working
signal still carries useful (if limited) information, and silently throwing it away to
return an error would be worse for the user than an honest "uncertain." A second, related
divergence: the audit-log schema grew across milestones (the persistent `content_records`
and `appeals` tables and the appeal columns did not exist in the Milestone 3 design), so
`database.py` gained an additive `PRAGMA table_info` + `ALTER TABLE` migration to upgrade
older databases without dropping data — which is exactly why the three real Milestone 3
rows above still survive in `provenance_guard.db`.

---

## 15. AI Usage

I used an AI coding assistant throughout, always against the spec in `planning.md`, and
reviewed/verified its output. Specific instances that actually occurred:

**Instance 1 — Architecture and specification.** I provided the project requirements and
had the assistant help draft the structure that became `planning.md`. I manually set and
kept ownership of the conservative design choices: the `0.85`/`0.20` thresholds, the
`>= 0.35` signal-disagreement rule, the wide uncertain band, the exact transparency-label
wording, and the principle that false positives against humans are the costlier error.

**Instance 2 — Milestone 3 implementation.** I gave the assistant the *Signal 1*, *API
Contract*, *Architecture (submission)*, *Edge Cases*, and *Audit-Log* sections. It
generated the Flask skeleton, the `run_groq_signal` function, and initial tests. I
manually verified request-field consistency (`text` not `content`), the JSON/Markdown-
fence parsing, controlled API-failure handling, and — importantly — that it **never
substitutes a fake fallback score** when Groq fails.

**Instance 3 — Milestone 4 scoring.** The assistant generated `stylometry.py` and
`scoring.py`. I verified the stylometric weighting and reliability tiers against
`planning.md`, and checked the combination formula, the disagreement/short-text/missing-
signal rules, and the exact thresholds by running deterministic unit tests with hardcoded
signal dictionaries (the same cases shown in §5).

**Instance 4 — Milestone 5 production layer.** The assistant generated `labels.py`, the
`/appeal` endpoint, the rate-limit configuration, and the database migration plus the
`content_records`/`appeals` tables. I manually verified the exact label text (including the
typographic apostrophe in "creator’s"), the `under_review` status change, the preservation
of the original attribution/confidence, and that the migration preserves existing rows.
Runtime evidence in this README was captured by actually running code — not invented.

---

## 16. Repository Structure

```text
ai201-project4-provenance-guard/
├── app.py                 # Flask app: endpoints, validation, rate limits, error handlers
├── detector.py            # Signal 1: Groq LLM assessment (run_groq_signal)
├── stylometry.py          # Signal 2: stylometric heuristics (run_stylometric_signal)
├── scoring.py             # combine_signals: reliability weighting + uncertainty rules
├── labels.py              # get_transparency_label + the three exact label strings
├── database.py            # SQLite: audit_log, content_records, appeals + migration
├── planning.md            # Full specification and architecture diagrams
├── README.md              # This report
├── requirements.txt       # flask, flask-limiter, groq, python-dotenv, pytest
├── provenance_guard.db    # Dev SQLite database (gitignored; holds real demo rows)
├── .env                   # GROQ_API_KEY (gitignored — never committed)
├── .gitignore
└── tests/
    ├── test_milestone3.py     # /submit validation, audit log, /health, rate-limit
    ├── test_milestone4.py     # two-signal integration on /submit
    ├── test_stylometry.py     # stylometric feature/score tests
    ├── test_scoring.py        # scoring thresholds and rules
    ├── test_labels.py         # exact transparency-label tests
    ├── test_appeals.py        # appeal workflow + content records
    └── test_rate_limits.py    # 429 behavior
```

---

## 17. Setup and Reproduction (Windows PowerShell)

```powershell
# 1. Clone the repository
git clone <your-repo-url>
cd ai201-project4-provenance-guard

# 2. Create a virtual environment
python -m venv .venv

# 3. Activate it
.\.venv\Scripts\Activate.ps1

# 4. Install requirements
pip install -r requirements.txt

# 5-6. Create .env and add your Groq API key (never commit this file)
"GROQ_API_KEY=your-real-key-here" | Out-File -Encoding utf8 .env

# 7. Run the tests (no API calls — detection is mocked)
python -m pytest -v

# 8. Start the Flask server
python app.py
```

```powershell
# 9. Submit content (second terminal)
curl.exe -X POST http://127.0.0.1:5000/submit -H "Content-Type: application/json" -d '{\"text\":\"Your creative text here...\",\"creator_id\":\"creator-123\",\"content_type\":\"blog_post\"}'

# 10. Inspect the audit log
curl.exe http://127.0.0.1:5000/log
```

> Never include or commit the actual API key. `.env` and `*.db` are gitignored.

---

## 18. Portfolio Walkthrough

A suggested 2–3 minute walkthrough outline (this is an outline only — no recording is
claimed or included):

1. Show the repository layout and the architecture diagrams in `planning.md`.
2. Start the Flask server (`python app.py`).
3. Send one `POST /submit` request with a substantial passage.
4. Point out the returned `attribution`, `confidence`, both signal scores under
   `signals`, and the `transparency_label`.
5. Send a second submission that is short or ambiguous to produce a lower-confidence /
   `uncertain` result, and contrast the two.
6. File an appeal with `POST /appeal` using the `content_id` returned in step 3.
7. Call `GET /content/<content_id>` and show the status has changed to `under_review`
   with `has_appeal: true`.
8. Call `GET /log` and show both the submission events and the appeal event, noting that
   only a content hash is stored — never the raw text.
9. Briefly explain the rate limits (`/submit` 10/min, `/appeal` 3/hour) and why they
   differ.
10. Close with one limitation (e.g. repetitive poetry can mislead stylometry) and one
    future improvement (e.g. calibration on a labeled dataset).

---

## Remaining Runtime Evidence

The following placeholders still require actual runtime output. Everything else in this
README was taken from the repository or captured by running code locally.

1. **Live end-to-end Milestone 4/5 `/submit` scores (§5 score table).** The score table
   currently uses the deterministic fixtures from `tests/test_scoring.py` with
   combined/confidence computed by `scoring.combine_signals`. A real two-signal submission
   against the live Groq API has not been captured.
2. **Current-schema audit-log entries including an appeal (§9).** The sample shows three
   real Milestone 3-era rows; no `appeal_submitted` event and no current two-signal
   submission row have been recorded yet. Placeholder:
   `[PASTE AT LEAST 1 ACTUAL appeal_submitted GET /log ENTRY (CURRENT M5 SCHEMA) HERE]`.

### Commands to obtain the missing evidence

These make **real Groq API calls** (they consume your key). Run with the server started
(`python app.py`) from a second PowerShell terminal:

```powershell
# (1) A real two-signal submission — record the JSON for the §5 table and note the content_id
curl.exe -X POST http://127.0.0.1:5000/submit -H "Content-Type: application/json" -d '{\"text\":\"<paste a 60+ word passage>\",\"creator_id\":\"demo-1\",\"content_type\":\"blog_post\"}'

# (2) File an appeal against that content_id (no API call), then capture the log
curl.exe -X POST http://127.0.0.1:5000/appeal -H "Content-Type: application/json" -d '{\"content_id\":\"<content_id-from-step-1>\",\"creator_reasoning\":\"I wrote this myself.\",\"evidence_description\":\"Dated drafts available.\"}'

# (3) Capture current-schema /log output (submission + appeal events) for §9
curl.exe http://127.0.0.1:5000/log
```

To regenerate the rate-limit evidence in §8 without any API calls, the
`tests/test_rate_limits.py` suite reproduces the 12-request sequence under
`python -m pytest tests/test_rate_limits.py -v`.
