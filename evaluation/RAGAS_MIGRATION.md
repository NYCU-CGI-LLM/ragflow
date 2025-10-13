# Migration to RAGAS Metrics

## What Changed

### ✅ **Completed Changes**

1. **New RAGAS Metrics Module** (`ragas_metrics.py`)
   - Replaced custom metrics with RAGAS framework
   - Supports ID-based retrieval metrics (fast, no LLM needed)
   - Adds generation quality metrics (LLM-based)

2. **Updated Evaluation Pipeline** (`ragas_evaluation.py`)
   - Uses RAGAS `context_precision` and `context_recall` instead of custom recall/precision
   - Added generation evaluation support
   - Batch processing for efficient RAGAS generation metrics

3. **Updated Configuration** (`config.py`)
   - Added `generation_enabled` flag
   - Added `generation_metrics` list
   - Added chat identifier support (`chat_id` or `chat_name`) for generation evaluation

4. **Updated Config Files**
   - `example_config.yaml`: Shows all RAGAS options including generation
   - Other configs: Need manual update (see below)

---

## RAGAS Retrieval Metrics Mapping

| Old Custom Metric | New RAGAS Metric | Calculation |
|-------------------|------------------|-------------|
| `recall@K` | `context_recall@K` | Same: relevant retrieved / total relevant |
| `precision@K` | `context_precision@K` | Same: relevant retrieved / total retrieved |
| `f1@K` | N/A (can calculate from precision+recall) | Removed |
| `ndcg@K` | N/A | Removed (not in RAGAS ID-based) |
| `mrr` | N/A | Removed (not in RAGAS ID-based) |
| `hit_rate` | N/A | Removed (not in RAGAS ID-based) |

**Note:** RAGAS ID-based metrics focus on the core precision/recall. Other metrics can be added back if needed.

---

## New RAGAS Generation Metrics

| Metric | What It Measures | Requires Ground Truth? | Requires LLM? |
|--------|------------------|----------------------|---------------|
| **faithfulness** | Is answer grounded in retrieved context? | ❌ No | ✅ Yes |
| **answer_relevancy** | Does answer address the question? | ❌ No | ✅ Yes |
| **answer_correctness** | Factual accuracy vs ground truth | ✅ Yes | ✅ Yes |
| **answer_similarity** | Semantic similarity to ground truth | ✅ Yes | ❌ No (embeddings only) |

---

## How to Use

### 1. Install RAGAS

```bash
pip install ragas datasets
```

### 2. For Retrieval-Only Evaluation (No Changes Needed)

Your existing config works! Just run:

```bash
python evaluation/run_evaluation.py --config evaluation/config/minimal_config.yaml
```

**Output will now show:**
- `context_precision@K` instead of `precision@K`
- `context_recall@K` instead of `recall@K`

### 3. For Generation Evaluation (Requires Setup)

#### Step 1: Add ground truth answers to your dataset

```json
[
  {
    "query": "What is the capital of France?",
    "doc_ids": [123, 456],
    "expected_answer": "Paris is the capital of France."
  }
]
```

#### Step 2: Create a chat in RAGFlow and note its ID or name

Via API or UI, create a chat associated with your dataset.

#### Step 3: Update config

Choose **one** of the modes below:

**Retrieval evaluation**
```yaml
retrieval:
  api_key: "your_api_key"
  dataset_id: "your_dataset_id"
  size: 5
  metrics: [recall, precision]
```

**Generation evaluation**
```yaml
generation:
  api_key: "your_api_key"
  chat_name: "your_chat_name"  # Or chat_id: "..."
  dynamic_rerank_limit: true
  metrics: [recall, precision]
```

#### Step 4: Run evaluation

```bash
python evaluation/run_evaluation.py --config your_config.yaml
```

---

## Example Output

### Retrieval Metrics

```
Retrieval Metrics (RAGAS ID-based)
--------------------------------------------------------------------------------

context_recall@5:
  Mean:   0.8500
  Median: 0.9000
  Std:    0.1200
  Min:    0.2000
  Max:    1.0000

context_precision@5:
  Mean:   0.7800
  Median: 0.8000
  Std:    0.1500
  Min:    0.2000
  Max:    1.0000
```

### Generation Metrics (if enabled)

```
Generation Metrics (RAGAS)
--------------------------------------------------------------------------------

faithfulness: 0.8234
answer_relevancy: 0.9123
answer_correctness: 0.7891
```

---

## Migration Checklist

- [x] Install RAGAS: `pip install ragas datasets`
- [ ] Test retrieval-only evaluation with existing config
- [ ] (Optional) Add ground truth answers to dataset
- [ ] (Optional) Create chat and record chat_id or chat_name
- [ ] (Optional) Enable generation metrics in config
- [ ] Update your custom scripts/notebooks to use new metric names

---

## Backward Compatibility

❌ **Breaking Changes:**
- Metric names changed: `recall@K` → `context_recall@K`
- Removed metrics: `f1@K`, `ndcg@K`, `mrr`, `hit_rate`
- Results structure changed: `metrics` → `retrieval_metrics`

If you need the old metrics back, you can:
1. Keep the old `metrics.py` file
2. Use both old and new metrics side-by-side
3. Manually calculate F1 from precision+recall

---

## Troubleshooting

### "Import ragas could not be resolved"

**Solution:** Install RAGAS

```bash
pip install ragas datasets
```

### "Generation evaluation requires a chat identifier"

**Solution:** Add `chat_id` or `chat_name` to your config, or disable generation:

```yaml
generation:
  enabled: false
```

### LLM API errors during generation metrics

**Solution:** RAGAS uses your OpenAI API key from environment:

```bash
export OPENAI_API_KEY="your_openai_api_key"
```

Or configure in RAGAS (see RAGAS documentation).

---

## Next Steps

1. **Test the new system:**
   ```bash
   python evaluation/run_evaluation.py --config evaluation/config/example_config.yaml
   ```

2. **Compare results:** Run same dataset with old and new metrics to verify

3. **Enable generation:** Once retrieval works, try generation evaluation

4. **Optimize:** Adjust generation metrics based on your needs (LLM calls = $$)

---

## Questions?

- RAGAS Documentation: https://docs.ragas.io/
- RAGFlow Evaluation README: `evaluation/README.md`
