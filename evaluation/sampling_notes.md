# Golden Evaluation Set — Sampling & Labelling Methodology

## Dataset Source
- **106,928** historical Apple Support tweets from `processed/apple_support.db`
- Pre-clustered into **12 intent clusters** via K-Means on `all-MiniLM-L6-v2` 384D embeddings

## Sampling Strategy

### In-Domain Examples (180 total: 12 clusters × 15 each)
| Source Type | Count per Cluster | Selection Method | Purpose |
|---|---|---|---|
| Centroid-Nearest | 10 | Highest cosine similarity to cluster centroid | Most representative, tests easy classification |
| Cluster Boundary | 5 | Lowest similarity to own centroid (below median) | Hardest in-cluster cases, tests classifier robustness |

### Guardrail Edge Cases (10 total)
| Category | Count | Source | Purpose |
|---|---|---|---|
| Prompt Injection | 5 | Handcrafted adversarial prompts | Tests injection defense (heuristic + LLM) |
| Off-Topic | 3 | Handcrafted non-Apple queries | Tests domain validity guardrail |
| Gibberish/Spam | 2 | Handcrafted keyboard spam | Tests gibberish detection |

### Ambiguous Cross-Cluster Examples (10 total)
- Selected as examples with the **smallest gap** between top-1 and top-2 centroid similarity scores
- These are genuinely hard cases where the query plausibly belongs to two different intent clusters
- Tests the classifier's ability to disambiguate overlapping intents

## Labelling Rubric

### `gold_intent` (int)
- For in-domain: the cluster ID assigned by K-Means (verified by centroid proximity)
- For guardrails: `-1` (no valid intent)
- For ambiguous: the K-Means assigned cluster (accepting that alternatives exist)

### `gold_is_dm` (bool)
- `True` if the issue involves: account credentials, billing, hardware damage, serial numbers, personal info, or if the historical Apple reply pattern was a DM redirect
- Cluster 9 (apple_id_login_failed) defaults to `True`
- Otherwise derived from the original `is_dm` field in the database

### `gold_reply_notes` (str)
- Key phrases and troubleshooting steps that a correct reply should reference
- Derived from Apple's actual historical resolution patterns for each cluster

### `difficulty` (str)
- **Easy**: Centroid-nearest (top 5), obvious guardrail triggers
- **Medium**: Centroid-nearest (rank 6-10), moderate guardrails
- **Hard**: Boundary examples, ambiguous cross-cluster, subtle injection attempts

## Known Biases & Limitations
1. **Temporal bias**: Dataset is from 2016-2017 Apple Support Twitter era; iOS 11 issues dominate
2. **Cluster quality**: K-Means labels are unsupervised; some clusters overlap (e.g., clusters 0 and 8 both relate to the letter "i" glitch)
3. **DM labelling**: The `is_dm` ground truth comes from whether Apple's *actual* historical reply was a DM redirect, not whether it *should have been*
4. **Single annotator**: Labels were generated programmatically + verified; no inter-annotator agreement measured
5. **English only**: All examples are English-language tweets
