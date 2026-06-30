# Provenance Guard — Planning Specification

## 1. Project Overview

Provenance Guard is a backend API that analyzes text-based creative work and estimates whether it shows stronger signals of AI generation or human authorship.

The system does not claim to prove who wrote a piece of content. AI-content detection is probabilistic and may fail on short, edited, translated, highly polished, or experimental writing. Provenance Guard therefore uses:

* Two independent detection signals
* Conservative decision thresholds
* An explicit uncertain result
* Plain-language transparency labels
* Structured audit logging
* A creator appeal process

A false positive—labeling human work as likely AI-generated—is treated as more harmful than a false negative. The architecture is intentionally conservative and sends ambiguous cases to the uncertain category.

---

## 2. Detection Signals

The system uses two distinct signals that evaluate different properties of the submitted text.

## Signal 1: Groq LLM Classification

### Purpose

The Groq signal evaluates the text holistically using `llama-3.3-70b-versatile`.

It considers semantic and stylistic patterns such as:

* Overall coherence
* Voice consistency
* Repetitive or generic transitions
* Predictable organization
* Excessively balanced phrasing
* Personal specificity
* Abrupt or unusual creative choices
* Repetition of ideas
* Whether the writing feels templated

### Input

```python
text: str
content_type: str
```

`content_type` may be:

* `poem`
* `short_story`
* `blog_post`
* `other`

The content type is provided so the LLM does not judge poetry using the same expectations as a blog post.

### Output

The function will return a dictionary:

```python
{
    "ai_score": 0.0,       # float from 0.0 to 1.0
    "reliability": 0.0,    # float from 0.0 to 1.0
    "flags": [],           # list of short internal observations
    "signal": "groq"
}
```

Example:

```python
{
    "ai_score": 0.82,
    "reliability": 0.76,
    "flags": [
        "uniform paragraph structure",
        "generic transitions",
        "highly consistent tone"
    ],
    "signal": "groq"
}
```

### Interpretation

* `0.0` means the text strongly resembles human-authored writing.
* `0.5` means the model cannot distinguish confidently.
* `1.0` means the text strongly resembles AI-generated writing.

The score is not proof of authorship.

### Blind spots

The Groq signal may perform poorly when:

* A human writes in a formal or highly polished style
* AI-generated text has been heavily edited
* The content is very short
* The work intentionally uses repetition
* The author is a non-native English speaker
* The content contains both human and AI contributions
* The model relies on topic or genre stereotypes
* The work uses experimental structure

---

## Signal 2: Stylometric Heuristics

### Purpose

The stylometric signal measures structural properties of the text without interpreting its meaning.

This gives the system an independent signal that does not depend on an LLM’s semantic judgment.

### Input

```python
text: str
content_type: str
```

### Features

The initial implementation will calculate:

1. Word count
2. Sentence count
3. Average sentence length
4. Sentence-length variance
5. Sentence-length coefficient of variation
6. Type-token ratio
7. Punctuation density
8. Paragraph-length variance
9. Repeated sentence-opening rate
10. Repeated phrase rate

### Initial feature scoring

Each feature will be normalized into a value from `0.0` to `1.0`, where higher values represent stronger AI-like regularity.

Example component scores:

```python
{
    "sentence_uniformity": 0.80,
    "paragraph_uniformity": 0.70,
    "repetition_score": 0.65,
    "punctuation_regularity": 0.55,
    "lexical_uniformity": 0.60
}
```

The initial stylometric score will use:

```text
stylometric_ai_score =
    0.30 × sentence_uniformity
  + 0.20 × paragraph_uniformity
  + 0.20 × repetition_score
  + 0.15 × punctuation_regularity
  + 0.15 × lexical_uniformity
```

### Output

```python
{
    "ai_score": 0.0,
    "reliability": 0.0,
    "features": {
        "word_count": 0,
        "average_sentence_length": 0.0,
        "sentence_length_variance": 0.0,
        "type_token_ratio": 0.0,
        "punctuation_density": 0.0,
        "paragraph_length_variance": 0.0,
        "repeated_opener_rate": 0.0
    },
    "component_scores": {
        "sentence_uniformity": 0.0,
        "paragraph_uniformity": 0.0,
        "repetition_score": 0.0,
        "punctuation_regularity": 0.0,
        "lexical_uniformity": 0.0
    },
    "signal": "stylometry"
}
```

### Reliability

Stylometric reliability depends on text length.

```text
Fewer than 50 words:
reliability = 0.20

50–99 words:
reliability = 0.50

100–249 words:
reliability = 0.75

250 words or more:
reliability = 1.00
```

Short texts do not provide enough data for stable structural measurements.

### Blind spots

Stylometry may perform poorly when:

* The text is very short
* A poem intentionally repeats words or line structures
* A human writer uses highly controlled prose
* AI is prompted to write irregularly
* Human editing changes AI-generated structure
* A writer has a naturally uniform style
* The text includes dialogue or lists
* The text is translated
* Punctuation conventions differ by dialect or language

---

## 3. Combining the Signals

The system first receives:

```python
groq_ai_score
stylometric_ai_score
groq_reliability
stylometric_reliability
```

### Base weights

The initial weights will be:

```text
Groq signal:       0.65
Stylometric signal: 0.35
```

The LLM receives more weight because it can interpret meaning, genre, and voice. Stylometry remains important as an independent structural check.

### Reliability-adjusted weighting

The effective weight of each signal will be:

```text
effective_groq_weight =
    0.65 × groq_reliability

effective_style_weight =
    0.35 × stylometric_reliability
```

The combined AI score will be:

```text
combined_ai_score =
    (
        groq_ai_score × effective_groq_weight
        +
        stylometric_ai_score × effective_style_weight
    )
    /
    (
        effective_groq_weight
        +
        effective_style_weight
    )
```

The final score is clamped to the range `0.0–1.0`.

### Signal-disagreement rule

The system will calculate:

```text
signal_gap =
    absolute value of
    groq_ai_score - stylometric_ai_score
```

If:

```text
signal_gap >= 0.35
```

the signals are considered to strongly disagree.

Strong disagreement forces the final attribution to:

```text
uncertain
```

even if the weighted average would otherwise cross a confident threshold.

### Missing-signal rule

If either signal fails:

* The system does not claim likely AI or likely human.
* Attribution becomes `uncertain`.
* Confidence is capped at `0.60`.
* The failure is written to the audit log.
* The API may still return a useful response instead of crashing.

If both signals fail, the API returns a service error and logs the failed attempt.

---

## 4. Uncertainty Representation

The system distinguishes between:

```text
ai_likelihood
```

and:

```text
confidence
```

### AI likelihood

`ai_likelihood` is the combined AI score from `0.0–1.0`.

Examples:

```text
0.05 → strongly human-leaning
0.50 → no clear direction
0.60 → slightly AI-leaning, but uncertain
0.95 → strongly AI-leaning
```

### Confidence

Confidence represents how strongly the score supports its direction.

It will be calculated as:

```text
confidence =
    maximum of:
    combined_ai_score
    and
    1 - combined_ai_score
```

Examples:

| AI likelihood | Direction | Confidence |
| ------------: | --------- | ---------: |
|          0.05 | Human     |       0.95 |
|          0.20 | Human     |       0.80 |
|          0.50 | Neither   |       0.50 |
|          0.60 | AI        |       0.60 |
|          0.85 | AI        |       0.85 |
|          0.95 | AI        |       0.95 |

A confidence score of `0.60` means the system has only a mild lean toward one attribution. It is not strong enough for a high-confidence transparency label.

### Attribution thresholds

The initial attribution rules will be:

```text
combined_ai_score >= 0.85
AND no strong signal disagreement
AND sufficient text length
→ likely_ai
```

```text
combined_ai_score <= 0.20
AND no strong signal disagreement
AND sufficient text length
→ likely_human
```

```text
combined_ai_score between 0.20 and 0.85
→ uncertain
```

The uncertain range is deliberately wide because falsely labeling human creative work as AI-generated may harm the creator.

### Short-text rule

If the text contains fewer than 50 words:

```text
attribution = uncertain
confidence is capped at 0.69
```

This rule applies even when one detector reports a very high score.

### Confidence examples

#### Example A

```text
Groq score: 0.94
Stylometric score: 0.89
Combined AI score: 0.92
Signal gap: 0.05
```

Result:

```text
attribution = likely_ai
confidence = 0.92
```

#### Example B

```text
Groq score: 0.67
Stylometric score: 0.54
Combined AI score: 0.62
Signal gap: 0.13
```

Result:

```text
attribution = uncertain
confidence = 0.62
```

#### Example C

```text
Groq score: 0.12
Stylometric score: 0.19
Combined AI score: 0.15
Signal gap: 0.07
```

Result:

```text
attribution = likely_human
confidence = 0.85
```

#### Example D

```text
Groq score: 0.91
Stylometric score: 0.31
Signal gap: 0.60
```

Result:

```text
attribution = uncertain
```

The signals disagree too strongly for a confident result.

---

## 5. Transparency Label Design

The exact label text will be centralized in `labels.py`. The API and README must use the same wording.

## High-confidence AI label

> “This content shows strong signals of AI generation. This automated assessment is not proof of authorship and may be incorrect. The creator can appeal this result and provide additional context or evidence.”

## High-confidence human label

> “This content shows strong signals of human authorship. This automated assessment is probabilistic and does not verify the creator’s identity, ownership, or full writing process.”

## Uncertain label

> “The system could not confidently determine whether this content was human-written or AI-generated. No definitive attribution is being made, and the content should not be penalized based on this result.”

### Design goals

The labels must:

* Avoid claiming certainty
* Avoid accusing the creator
* Explain that automated detection can be wrong
* Make uncertainty visible
* Tell creators that appeals are available
* Avoid technical language such as model logits or feature weights
* Avoid automatic punishment

---

## 6. Appeals Workflow

## Who may appeal

An appeal may be submitted by:

* The original creator identified by `creator_id`
* A platform account authorized to act for the creator
* An administrator during testing

For the course project, authorization will be represented through the same `creator_id` used during submission.

A production system would require authenticated accounts and access control.

## Appeal endpoint

```text
POST /appeal
```

### Required request fields

```json
{
  "content_id": "content_abc123",
  "creator_id": "creator_123",
  "creator_reasoning": "I wrote this poem myself and can provide dated drafts."
}
```

### Optional fields

```json
{
  "evidence_description": "I have Google Docs version history and handwritten notes."
}
```

The project will not upload private evidence files. It will record a description of available evidence.

## Validation

The system checks:

1. The content record exists.
2. The creator ID matches the original submission.
3. The reasoning is not empty.
4. The reasoning is within the length limit.
5. The content is not already under review.
6. The appeal rate limit has not been exceeded.

## Status change

Before appeal:

```text
status = completed
```

After a valid appeal:

```text
status = under_review
```

The original attribution and score are preserved. They are not deleted or automatically reversed.

## Appeal audit entry

The log will store:

```python
{
    "event_type": "appeal_submitted",
    "appeal_id": "appeal_xyz789",
    "content_id": "content_abc123",
    "creator_id": "creator_123",
    "creator_reasoning": "...",
    "evidence_description": "...",
    "original_attribution": "likely_ai",
    "original_confidence": 0.91,
    "new_status": "under_review",
    "timestamp": "..."
}
```

## Human-review queue

A human reviewer should see:

* Appeal ID
* Content ID
* Submission timestamp
* Content type
* Original attribution
* Original AI likelihood
* Confidence
* Groq signal score
* Stylometric signal score
* Signal disagreement
* Stylometric features
* Transparency label used
* Creator reasoning
* Evidence description
* Current status

The reviewer may later update the status to:

```text
appeal_approved
appeal_denied
needs_more_information
```

Automated reclassification is not required for this project.

---

## 7. Anticipated Edge Cases

## Edge Case 1: Repetitive poetry

A human-written poem may use repeated lines, simple vocabulary, and consistent rhythm.

Stylometry may interpret this as:

* Low sentence-length variance
* High repetition
* High structural uniformity

This may produce a false AI signal.

### Handling

* Pass `content_type="poem"` into both detectors.
* Reduce the importance of repetition for poetry.
* Use conservative thresholds.
* Force uncertain for short poems.
* Preserve an appeal path.

---

## Edge Case 2: Highly polished human prose

A professional writer may produce:

* Consistent paragraphs
* Smooth transitions
* Grammatically uniform sentences
* Balanced structure

Both Groq and stylometry may incorrectly treat this polish as AI-like.

### Handling

* Do not use polish alone as evidence of AI generation.
* Require both signals to agree strongly.
* Keep the likely-AI threshold at `0.85`.
* Use transparency language that avoids proof claims.
* Allow appeal with revision history or drafts.

---

## Edge Case 3: Heavily edited AI text

A creator may generate text with AI and then edit:

* Sentence lengths
* Vocabulary
* Punctuation
* Voice
* Structure

Stylometry may identify the result as human-like even though AI contributed substantially.

### Handling

The system may produce likely human or uncertain. This is an accepted limitation because the system evaluates the submitted text, not its hidden creation history.

The transparency label must not claim verified human creation.

---

## Edge Case 4: Very short text

A short poem, title, caption, or paragraph provides too little information for meaningful detection.

### Handling

* Content under 50 words receives `uncertain`.
* Confidence is capped.
* The audit log records `insufficient_length`.
* The API explains that no definitive attribution was made.

---

## Edge Case 5: Non-native English writing

A human author may use unusual grammar, repeated structures, or limited vocabulary.

An LLM may interpret these patterns incorrectly.

### Handling

* Do not treat grammar quality as proof.
* Include limitations in the README.
* Use conservative attribution thresholds.
* Avoid automatic penalties.
* Support appeals.

---

## Edge Case 6: Mixed human and AI authorship

A creator may write part of a story and use AI to revise another section.

The system only returns one score for the entire submitted text.

### Handling

The likely result should usually be uncertain because different sections may produce conflicting patterns.

A future version could score paragraphs independently.

---

## Edge Case 7: Translated text

Human-written work translated by software may have smooth, uniform wording that appears AI-generated.

### Handling

The current system may perform poorly because it does not know the translation history.

The creator can include this information in an appeal.

---

## 8. API Contract

## `POST /submit`

### Request

```json
{
  "content": "Text to analyze...",
  "content_type": "short_story",
  "creator_id": "creator_123"
}
```

### Success response

```json
{
  "content_id": "content_abc123",
  "attribution": "uncertain",
  "ai_likelihood": 0.61,
  "confidence": 0.61,
  "status": "completed",
  "transparency_label": "The system could not confidently determine whether this content was human-written or AI-generated. No definitive attribution is being made, and the content should not be penalized based on this result.",
  "signals": {
    "groq": {
      "ai_score": 0.66,
      "reliability": 0.80
    },
    "stylometry": {
      "ai_score": 0.52,
      "reliability": 0.75
    },
    "signal_gap": 0.14
  }
}
```

## `POST /appeal`

### Request

```json
{
  "content_id": "content_abc123",
  "creator_id": "creator_123",
  "creator_reasoning": "I wrote this myself and can provide dated drafts.",
  "evidence_description": "Google Docs version history is available."
}
```

### Response

```json
{
  "appeal_id": "appeal_xyz789",
  "content_id": "content_abc123",
  "status": "under_review",
  "message": "Your appeal has been recorded and the content is now under review."
}
```

## `GET /log`

Returns structured audit-log entries.

For the course demo, at least three attribution entries will be visible.

In a production system this endpoint would require administrator authentication.

## `GET /content/<content_id>`

Returns:

* Current attribution
* Confidence
* Status
* Appeal status

## `GET /health`

Returns:

```json
{
  "status": "ok"
}
```

---

## 9. Rate-Limit Plan

The initial rate limits will be:

```text
POST /submit:
10 requests per minute per IP
100 requests per day per IP
```

```text
POST /appeal:
3 appeals per hour per IP
10 appeals per day per IP
```

### Reasoning

A normal creator is unlikely to submit more than ten works in one minute. The per-minute limit slows automated flooding while still allowing legitimate testing.

Appeals should be less frequent than submissions. A lower appeal rate prevents spam and repeated appeal creation for the same content.

The system will also reject duplicate appeals when content already has status:

```text
under_review
```

---

## 10. Audit-Log Plan

SQLite will store attribution and appeal events.

## Attribution record fields

```text
id
event_type
content_id
creator_id
content_hash
content_type
word_count
groq_ai_score
groq_reliability
stylometric_ai_score
stylometric_reliability
signal_gap
combined_ai_score
confidence
attribution
transparency_label
status
created_at
```

## Appeal fields

```text
appeal_id
content_id
creator_id
creator_reasoning
evidence_description
original_attribution
original_confidence
status
created_at
```

Raw submitted text will not be returned through `GET /log`. The system will store a content hash for traceability.

---

## 11. Architecture

### Narrative

The submission flow begins when a client sends raw text to `POST /submit`. Flask validates and rate-limits the request, then passes the text to the Groq and stylometric signals. The scoring component combines both outputs, applies disagreement and uncertainty rules, selects one of the three transparency labels, writes the full decision to the audit log, and returns a structured JSON response.

The appeal flow begins when the creator sends a content ID and reasoning to `POST /appeal`. The API verifies the content and creator, stores the appeal, changes the content status to `under_review`, writes an appeal event to the audit log, and returns confirmation.

### Submission flow

```text
Client / Creative Platform
        │
        │ POST /submit
        │ raw text + content type + creator ID
        ▼
┌──────────────────────────────┐
│ Flask Submission Endpoint    │
│ Validate + rate limit        │
└──────────────┬───────────────┘
               │ validated text
               ▼
┌──────────────────────────────┐
│ Detection Orchestrator       │
└──────────┬────────────┬──────┘
           │            │
           │ raw text   │ raw text
           ▼            ▼
┌──────────────────┐  ┌─────────────────────┐
│ Groq Signal      │  │ Stylometric Signal  │
│ AI score         │  │ AI score            │
│ reliability      │  │ reliability         │
│ flags            │  │ features            │
└────────┬─────────┘  └──────────┬──────────┘
         │                        │
         │ signal output          │ signal output
         └────────────┬───────────┘
                      ▼
          ┌──────────────────────────┐
          │ Confidence Scoring       │
          │ Reliability weighting    │
          │ Disagreement rule        │
          │ Short-text rule          │
          └────────────┬─────────────┘
                       │ combined score
                       ▼
          ┌──────────────────────────┐
          │ Attribution Decision     │
          │ likely_ai                │
          │ likely_human             │
          │ uncertain                │
          └────────────┬─────────────┘
                       │ attribution + confidence
                       ▼
          ┌──────────────────────────┐
          │ Transparency Label       │
          │ Exact reader-facing text │
          └────────────┬─────────────┘
                       │ complete decision
                       ▼
          ┌──────────────────────────┐
          │ SQLite Audit Log         │
          │ Scores, features, status │
          └────────────┬─────────────┘
                       │ saved record
                       ▼
          ┌──────────────────────────┐
          │ JSON Response            │
          └──────────────────────────┘
```

### Appeal flow

```text
Creator
   │
   │ POST /appeal
   │ content ID + creator ID + reasoning
   ▼
┌──────────────────────────────┐
│ Appeal Endpoint              │
│ Validate + rate limit        │
└──────────────┬───────────────┘
               │ content ID
               ▼
┌──────────────────────────────┐
│ Content Lookup               │
│ Verify creator and status    │
└──────────────┬───────────────┘
               │ original decision
               ▼
┌──────────────────────────────┐
│ Appeal Service               │
│ Store reasoning and evidence │
│ Generate appeal ID           │
└──────────────┬───────────────┘
               │ appeal record
               ▼
┌──────────────────────────────┐
│ Status Update                │
│ status = under_review        │
└──────────────┬───────────────┘
               │ updated record
               ▼
┌──────────────────────────────┐
│ SQLite Audit Log             │
│ Original result + appeal     │
└──────────────┬───────────────┘
               │ saved event
               ▼
┌──────────────────────────────┐
│ JSON Confirmation Response   │
└──────────────────────────────┘
```

---

## 12. Planned File Structure

```text
ai201-project4-provenance-guard/
├── app.py
├── detector.py
├── stylometry.py
├── scoring.py
├── labels.py
├── database.py
├── planning.md
├── README.md
├── requirements.txt
├── .env
├── .gitignore
└── tests/
    ├── test_submit.py
    ├── test_groq_signal.py
    ├── test_stylometry.py
    ├── test_scoring.py
    ├── test_labels.py
    ├── test_appeals.py
    └── test_rate_limits.py
```

---

## 13. AI Tool Plan

## Milestone 3 — Submission Endpoint and First Signal

### AI tool

Claude

### Spec sections to provide

* Detection Signal 1: Groq LLM Classification
* API Contract
* Architecture submission diagram
* Anticipated Edge Cases
* Audit-Log Plan

### Request

Ask the AI tool to generate:

* Flask application skeleton
* `POST /submit` route
* Request validation
* Content ID generation
* Groq client initialization
* Groq signal function
* Structured JSON parsing
* Graceful Groq API failure handling
* Initial audit-log record structure
* Tests for valid and invalid submissions

### Verification

Before connecting the signal to the endpoint, test the Groq function directly with:

1. A polished AI-style blog paragraph
2. A personal human-style paragraph
3. A short poem
4. Empty input
5. Missing API key
6. Invalid Groq response

Check that:

* Output is always a dictionary
* `ai_score` stays between `0.0` and `1.0`
* Reliability is present
* Failures return a controlled error
* The API key is never exposed

---

## Milestone 4 — Stylometry and Confidence Scoring

### AI tool

Claude

### Spec sections to provide

* Detection Signal 2: Stylometric Heuristics
* Combining the Signals
* Uncertainty Representation
* Anticipated Edge Cases
* Architecture submission diagram

### Request

Ask the AI tool to generate:

* `stylometry.py`
* Sentence and paragraph parsing
* Feature calculations
* Feature normalization
* Stylometric AI score
* Reliability based on word count
* `scoring.py`
* Reliability-adjusted weighted combination
* Signal-disagreement rule
* Short-text rule
* Final attribution and confidence logic
* Unit tests for different score ranges

### Verification

Test:

1. Long uniform AI-style text
2. Irregular personal writing
3. Repetitive human poetry
4. Very short content
5. Two signals that agree strongly
6. Two signals that disagree strongly
7. Scores near each threshold

Confirm:

* Scores vary meaningfully across texts
* Short text becomes uncertain
* A score near `0.60` is uncertain
* A score above `0.85` can become likely AI
* A score below `0.20` can become likely human
* Strong signal disagreement forces uncertainty
* Scores remain within `0.0–1.0`

---

## Milestone 5 — Production Layer

### AI tool

Claude

### Spec sections to provide

* Transparency Label Design
* Appeals Workflow
* Rate-Limit Plan
* Audit-Log Plan
* API Contract
* Architecture appeal diagram

### Request

Ask the AI tool to generate:

* `labels.py`
* Exact transparency-label logic
* SQLite database initialization
* Attribution audit logging
* Appeal storage
* `POST /appeal`
* Status update to `under_review`
* `GET /log`
* `GET /content/<content_id>`
* Flask-Limiter configuration
* Tests for all three labels
* Tests for successful and invalid appeals
* Tests for rate-limit responses

### Verification

Confirm that:

1. All three transparency-label variants are reachable.
2. The exact label text matches this document.
3. A valid appeal stores creator reasoning.
4. An appeal links to the original attribution.
5. Content status changes to `under_review`.
6. Duplicate appeals return `409`.
7. Unknown content IDs return `404`.
8. Unauthorized creator IDs are rejected.
9. Audit logs include signals, confidence, attribution, and appeals.
10. At least three attribution records can be displayed through `GET /log`.
11. Submission and appeal rate limits return `429` when exceeded.

---

## 14. Stretch-Feature Rule

No stretch feature will be implemented until this document is updated with:

* The new component
* Its API contract
* Its data flow
* Its failure cases
* Its audit-log fields
* Its testing plan
