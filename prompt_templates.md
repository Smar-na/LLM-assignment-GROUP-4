# Prompt Templates — Toxic Comment Classification

This document contains all prompt templates used across the four GPT-4.1 classification pipelines. Each template is versioned and annotated with its design rationale. Templates are referenced in the report (Section 2.3) and implemented in the corresponding notebooks.

**Model:** GPT-4.1 (`temperature=0`, `seed=42`, `max_tokens=150`)  
**Output format:** M1 returns binary 0/1 JSON (rescaled to 0/100 before threshold application); M2–M4 return JSON confidence scores (0–100 scale) converted to binary predictions via per-label thresholds  
**Labels:** `toxic`, `severe_toxic`, `obscene`, `threat`, `insult`, `identity_hate`

---

## Table of Contents

1. [Method 1 — Zero-Shot Prompt](#1-method-1--zero-shot-prompt)
2. [Method 2 — Few-Shot Static Prompt](#2-method-2--few-shot-static-prompt)
3. [Method 3 — RAG Dynamic Prompt](#3-method-3--rag-dynamic-prompt)
4. [Method 4 — Label-Aware RAG Prompt (Main)](#4-method-4--label-aware-rag-prompt-main)
5. [Method 4 — Severity Check Prompt (Second Pass)](#5-method-4--severity-check-prompt-second-pass)
6. [Method 4 — Identity Hate Check Prompt (Second Pass)](#6-method-4--identity-hate-check-prompt-second-pass)
7. [Design Rationale and Iteration Log](#7-design-rationale-and-iteration-log)

---

## 1. Method 1 — Zero-Shot Prompt

**File:** `Finalmethod1_zero_shot.ipynb`  
**Strategy:** Zero-shot with role assignment, label definitions, decision guidelines, and strict JSON output enforcement  
**System message:** `You are a strict multi-label text classifier specialising in toxic content detection for online comments.`

```
SYSTEM ROLE: You are a strict multi-label text classifier specialising in toxic content detection for online comments.

TASK: Classify the given comment into six toxicity categories using multi-label classification.
Each category must be assigned a binary value:
• 1 = label is present
• 0 = label is not present

IMPORTANT:
• Labels are NOT mutually exclusive.
• A comment can have multiple labels = 1.
• If the comment contains no toxic content, all labels must be 0.
• Do NOT provide explanations.
• Do NOT infer beyond the text.
• Be conservative and evidence-based, especially for rare labels.

LABEL DEFINITIONS:
1. toxic: General rude, disrespectful, or offensive language.
2. severe_toxic: Extremely aggressive, abusive, or highly hostile language.
3. obscene: Profanity, vulgar, or sexually explicit language.
4. threat: Statements that imply violence, harm, or intimidation.
5. insult: Direct personal attacks, humiliation, or degrading remarks toward a person or group.
6. identity_hate: Hate speech targeting protected characteristics (e.g., race, religion, gender, nationality, ethnicity, sexual orientation).

DECISION GUIDELINES (CRITICAL FOR CONSISTENCY):
• Assign "threat" = 1 only if there is a clear indication of harm or violence.
• Assign "identity_hate" = 1 only if hatred targets an identity group, not just an individual.
• Profanity alone → usually "obscene" and/or "toxic", not automatically "severe_toxic".
• Strong personal attack → "insult" (and possibly "toxic").
• Extremely abusive language → may include "severe_toxic" in addition to other labels.
• Absence of offensive cues → all zeros.

OUTPUT FORMAT (STRICT — JSON ONLY):
Return ONLY a valid JSON object with EXACTLY these keys:
{
"toxic": 0,
"severe_toxic": 0,
"obscene": 0,
"threat": 0,
"insult": 0,
"identity_hate": 0
}

NO extra text. NO explanations. NO additional fields.

Comment: "{comment}"
```

**Output format note:** Method 1 requests binary 0/1 labels directly. The `extract_scores()` function detects binary output and rescales it to 0 or 100 before applying thresholds, so thresholds are applied on the rescaled 0–100 values. In practice, for binary output, the threshold values below only control which label (0 or 100) triggers a positive prediction — any threshold below 100 will pass a positive through.

**Thresholds applied (to rescaled 0–100 scores):** `toxic=25`, `severe_toxic=75`, `obscene=40`, `threat=40`, `insult=50`, `identity_hate=55`

---

## 2. Method 2 — Few-Shot Static Prompt

**File:** `Finalmethod2_few_shot_simple.ipynb`  
**Strategy:** Static few-shot with 41 curated real examples covering all label combinations in the Jigsaw dataset  
**System message:** `You are a precise toxicity classification assistant.`

> **Note:** The `{FEW_SHOT_BLOCK}` placeholder is replaced at runtime with 41 formatted examples drawn from `few_shot_pool_fixed_3.csv`. Each example includes the comment text and its confidence scores (binary labels scaled to 0 or 100). Examples cover all 41 unique label combinations observed in the dataset.

```
You are an expert multi-label toxicity classifier.

Labels and definitions:
- toxic: rude, hateful, or aggressive language
- severe_toxic: extreme abuse or dehumanising hostility (STRICTER than toxic; dehumanisation, calls for violence)
- obscene: explicit profanity or vulgar sexual language
- threat: real, specific implication of harm toward a person (NOT general anger or hyperbole)
- insult: direct personal attack on the person addressed
- identity_hate: attack targeting a protected GROUP identity (race, religion, gender, sexuality)

Decision rules:
1. severe_toxic is a strict subset of toxic.
2. threat requires clear intent — not just angry language.
3. identity_hate must target GROUP identity, not just insult an individual.
4. Multi-label output is allowed.

Reference examples (use these to calibrate your confidence scores):
{FEW_SHOT_BLOCK}

Classify this comment:
{comment}

Output ONLY this JSON with confidence scores 0-100 for each label and nothing else:
{"scores": {"toxic": 0, "severe_toxic": 0, "obscene": 0, "threat": 0, "insult": 0, "identity_hate": 0}}
```

**Thresholds applied:** `toxic=25`, `severe_toxic=75`, `obscene=40`, `threat=40`, `insult=50`, `identity_hate=55`

---

## 3. Method 3 — RAG Dynamic Prompt

**File:** `FINAL_METHOD3_RAGDYNAMIC.ipynb`  
**Strategy:** Per-query dynamic retrieval via `text-embedding-3-small` cosine similarity; top-7 examples injected per comment  
**System message:** `You are a precise multi-label toxicity classification assistant. Return valid JSON only.`

> **Note:** `{retrieved_block}` is replaced at runtime with the top-K most semantically similar examples retrieved from the pool for each individual comment, with at least 3 of the 7 slots reserved for hard-minority-label positives (`severe_toxic`, `threat`, `identity_hate`). Each example shows the comment, its similarity score, and its label scores. `{n_retrieved}` is the actual number of examples retrieved (up to 7).

```
You are an expert multi-label toxicity classifier for online comments.

Your task is to assign a confidence score from 0 to 100 for each toxicity label.

LABELS:
- toxic: rude, aggressive, hostile, or generally offensive language
- severe_toxic: extreme abuse, dehumanising hostility, or very intense toxic language
- obscene: explicit profanity, vulgarity, or sexually explicit/offensive wording
- threat: statement implying physical harm, violence, intimidation, or a clear wish to injure
- insult: direct personal attack, humiliation, ridicule, or degrading statement toward a person
- identity_hate: attack, contempt, or hate directed at a protected group identity
  (e.g. race, religion, ethnicity, nationality, gender, sexual orientation)

IMPORTANT DECISION RULES:
1. Multi-label classification is allowed. More than one label may be present.
2. severe_toxic is a strict subset of toxic.
   - If severe_toxic is high, toxic should also be high.
3. threat requires a genuine implication of harm or violence.
   - Mere anger, profanity, or "I hate you" is not enough.
4. identity_hate requires hostility toward a protected group identity.
   - Attacking one individual without group-based targeting is not identity_hate.
5. Profanity alone may indicate obscene and/or toxic, but not automatically severe_toxic.
6. insult requires a direct degrading or attacking expression toward a person or target.
7. Be conservative for rare labels, especially threat and identity_hate.
8. Base the decision only on the text provided. Do not infer hidden intent beyond the wording.

HOW TO USE THE REFERENCE EXAMPLES:
- The {n_retrieved} examples below were retrieved specifically for this comment
  using semantic similarity (text-embedding-3-small cosine search).
- Similarity scores are shown — higher means more semantically related.
- Use them to calibrate label thresholds, not to copy scores mechanically.

RETRIEVED EXAMPLES (ordered by similarity):
{retrieved_block}

Now classify this comment:
"""{comment}"""

SCORING GUIDANCE:
- 0-10: label clearly absent
- 11-39: weak or ambiguous evidence
- 40-59: borderline / uncertain presence
- 60-79: reasonably clear presence
- 80-100: very clear presence

OUTPUT REQUIREMENTS:
- Return ONLY one valid JSON object.
- Use EXACTLY the keys shown below.
- Do not include explanations, markdown, or extra text.

{
  "scores": {
    "toxic": 0,
    "severe_toxic": 0,
    "obscene": 0,
    "threat": 0,
    "insult": 0,
    "identity_hate": 0
  }
}
```

**Thresholds applied:** `toxic=25`, `severe_toxic=75`, `obscene=40`, `threat=40`, `insult=50`, `identity_hate=55`

---

## 4. Method 4 — Label-Aware RAG Prompt (Main)

**File:** `FINAL_METHOD4.ipynb`  
**Strategy:** Label-aware retrieval with purity-ranked examples, 4-point severity rubric, boundary examples for each label, and explicit failure-mode warnings  
**System message:** `You are a precise toxicity classification assistant. Output only valid JSON.`

> **Note:** `{examples_block}` is replaced at runtime with label-aware retrieved examples formatted with label strings and similarity scores. `{comment}` is the target comment. On self-consistency passes, the intro line changes to `"Re-evaluate this comment carefully, paying extra attention to severity and context:"`.

```
You are an expert multi-label toxicity classifier.

Labels and definitions (read carefully — common mistakes noted):

- toxic: rude, hateful, or aggressive language that could make someone want to leave a conversation.

- severe_toxic: SIGNIFICANTLY more intense than ordinary toxic. Use this severity scale:
  * Mild toxic (score 10-30): single profanity, casual rudeness, mild name-calling
  * Moderate toxic (score 30-50): repeated profanity, aggressive hostility, personal attacks
  * Severe toxic (score 50-80): sustained extreme abuse, dehumanising language, violent hostility,
    intense degrading attacks, or repeated extreme obscene abuse directed at a person
  * Extreme toxic (score 80-100): explicit calls for harm combined with dehumanisation
  COMMON MISTAKE: Do not require genocide-level language for severe_toxic. Intense, sustained
  personal abuse with extreme profanity IS severe_toxic even without extermination language.
  ALSO: A single profane word alone is NOT severe_toxic — severity requires intensity OR persistence.

- obscene: explicit profanity, vulgar sexual language, or graphic crude language.

- threat: a CREDIBLE, SPECIFIC indication of intent to cause physical harm to a person.
  REQUIRES: (a) an identifiable target AND (b) a stated harmful action.
  IS a threat: "I will find you and kill you", "you're going to get shot"
  IS NOT a threat: "I could kill for a pizza" (hyperbole), "fuck off and die" (anger expression),
  "someone should teach them a lesson" (vague, no specific target or action),
  "I hope you get hit by a bus" (wish, not stated intent to act).
  COMMON MISTAKE: Angry language is NOT a threat unless it contains specific intent to harm.

- insult: a direct personal attack on someone's character, intelligence, or worth.
  Must target a specific person or addressee, not a general statement.

- identity_hate: attacks targeting a person BECAUSE OF their membership in a protected group
  (race, religion, ethnicity, gender, sexual orientation, nationality, disability).
  IS identity_hate: "all Muslims are terrorists", "go back to your country", slurs targeting a group
  IS NOT identity_hate: "you're an idiot" (personal insult, not group-based),
  "I disagree with Islam" (critique of religion, not attack on people)
  COMMON MISTAKE: Missing coded/indirect hate speech like "these people don't belong here"
  when referring to an ethnic or religious group.

Decision rules:
1. Multi-label output is expected — most toxic comments have 2-4 labels active.
2. If severe_toxic is positive, toxic must also be positive.
3. Score each label independently based on its definition above.
4. Pay attention to the INTENSITY and PERSISTENCE of language, not just the presence of bad words.

Reference examples (study the severity gradients):
{examples_block}

Classify this comment:
{comment}

Output ONLY this JSON with confidence scores 0-100 and nothing else:
{
  "scores": {
    "toxic": 0,
    "severe_toxic": 0,
    "obscene": 0,
    "threat": 0,
    "insult": 0,
    "identity_hate": 0
  }
}
```

**Thresholds applied (dev-tuned):** Optimised per label on held-out dev set with recall floors (`severe_toxic` ≥ 0.50, `threat` ≥ 0.50, `identity_hate` ≥ 0.60). Saved to `threshold_from_dev_v5_cl1.json`.

---

## 5. Method 4 — Severity Check Prompt (Second Pass)

**File:** `FINAL_METHOD4.ipynb`  
**Trigger:** Called when `toxic ≥ 40` AND `20 ≤ severe_toxic ≤ 65` in the main prompt output  
**Purpose:** Resolves borderline `severe_toxic` cases using a focused 4-point severity rubric  
**Merge:** Second-pass score weighted 0.5 × original + 0.5 × severity check score

```
You are evaluating whether a toxic comment reaches the level of SEVERE toxicity.

Severity scale:
- NOT severe: single profanity, mild rudeness, casual insult, disagreement expressed rudely
- BORDERLINE: aggressive hostility, multiple strong profanities, demeaning personal attack
- SEVERE: sustained extreme abuse, dehumanising language, intense degrading attack, violent hostility,
  or repeated extreme obscenity directed at a person

Comment: {comment_text}

Rate the severity as a single integer 0-100 where:
0-30 = not severe, 30-60 = borderline, 60-100 = clearly severe.

Output ONLY this JSON:
{"severe_toxic_score": 0}
```

---

## 6. Method 4 — Identity Hate Check Prompt (Second Pass)

**File:** `FINAL_METHOD4.ipynb`  
**Trigger:** Called when `identity_hate` hint ≥ 0.08 AND `15 ≤ identity_hate score ≤ 55` in the main prompt output  
**Purpose:** Resolves borderline `identity_hate` cases by explicitly probing for group-based targeting  
**Merge:** Second-pass score weighted 0.5 × original + 0.5 × identity check score

```
Does this comment attack, demean, or express hatred toward people BECAUSE OF
their membership in a protected group (race, religion, ethnicity, gender, sexual orientation,
nationality, disability)?

Look for:
- Explicit slurs targeting a group
- Stereotyping or generalising negatively about a group
- Coded language like "these people", "go back to your country"
- Dehumanising comparisons applied to a group

Comment: {comment_text}

Rate the likelihood of identity-based hate as a single integer 0-100.

Output ONLY this JSON:
{"identity_hate_score": 0}
```

---

## 7. Design Rationale and Iteration Log

This section documents the key design decisions made across prompt iterations and explains why each choice was made. This addresses the rubric requirement to explain *why* design choices worked or failed, not just report results.

### 7.1 Zero-Shot → Few-Shot: Why Static Examples Hurt Performance

Method 2 added 41 static examples to the zero-shot prompt. Contrary to expectations, this degraded macro F1 from 0.773 to 0.723. Three failure mechanisms were identified:

- **Context-length dilution:** The 41-example block added ~3,000 tokens per prompt, impairing the model's attention to the target comment
- **Example-anchoring bias:** Only 3 of 41 examples carried `severe_toxic`, biasing the model toward conservatism on that label
- **Static mismatch:** Every comment received identical examples regardless of content, limiting calibration value

**Lesson:** The number of examples is less important than their relevance to the specific comment being classified.

### 7.2 Few-Shot → RAG Dynamic: Addressing the Three Failure Mechanisms

Method 3 replaced static examples with per-query retrieval, directly addressing all three Method 2 failures:
- Shorter context (7 vs 41 examples)
- Semantic relevance per query via `text-embedding-3-small` cosine similarity
- Naturally adapted label distribution per comment

Result: macro F1 recovered to 0.775 and ROC-AUC improved to 0.9295 (+3.45pp vs zero-shot).

### 7.3 RAG Dynamic → Label-Aware RAG: Targeted Error-Driven Fixes

Method 4 addressed three specific weaknesses identified through error analysis on the Method 3 dev set:

| Weakness | Fix Applied | Effect |
|----------|-------------|--------|
| `severe_toxic` under-detection (threshold=75 too strict) | 4-point severity rubric + dedicated second-pass severity check | Recall: 0.529 → 0.702 |
| `threat` over-prediction (anger flagged as threat) | Explicit boundary examples (IS / IS NOT a threat) + target-proximity requirement | FP reduced by 75% vs Methods 1 & 3 |
| `identity_hate` under-detection (coded language missed) | Group-noun detection heuristic + dedicated second-pass identity check | Recall improvement on borderline cases |

### 7.4 Output Format Decision: Confidence Scores vs Binary Labels

Methods 2–4 were designed to return confidence scores (0–100) rather than binary labels (0/1) for two reasons:

1. **Threshold flexibility:** Returning scores allows per-label threshold tuning on a dev set without re-running inference. This was critical for Method 4, where the `severe_toxic` threshold was dropped from 75 to 20 after dev-set analysis.
2. **ROC-AUC computation:** Continuous scores enable ROC-AUC reporting, providing a threshold-independent performance metric that validates whether degradation is real or just a threshold artefact.

Method 1 requests binary 0/1 labels directly. The pipeline rescales these to 0/100 before applying thresholds, so Method 1 cannot benefit from threshold tuning in the same way — a label is either 0 or 100, with no intermediate signal.

### 7.5 Fallback Prompt

All methods include a fallback for cases where the main prompt returns unparseable output:

```
Classify this comment for toxicity. Return ONLY valid JSON with integer scores 0-100:
{comment_text}

{"scores":{"toxic":0,"severe_toxic":0,"obscene":0,"threat":0,"insult":0,"identity_hate":0}}
```

This ensures every row yields a valid prediction. Failed parses are logged to `error_log.csv` for analysis.
