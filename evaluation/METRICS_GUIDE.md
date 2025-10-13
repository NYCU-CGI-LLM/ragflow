# RAGAS Metrics Guide

Complete guide to all metrics supported in the RAGFlow evaluation system.

> **Note:** Each evaluation run should provide either a `retrieval` **or** a `generation`
> section in the config (not both). The evaluator automatically computes retrieval metrics
> in both modes; generation mode additionally produces model answers for scoring.

---

## 📊 Retrieval Metrics (ID-based)

These metrics evaluate **retrieval quality** using document IDs. They are:
- ✅ **Fast** - No LLM or API calls needed
- ✅ **Free** - No API costs
- ✅ **Deterministic** - Same results every time

### Available Metrics

| Config Name | RAGAS Name | Formula | What It Measures |
|-------------|------------|---------|------------------|
| `recall` | `context_recall@K` | relevant_retrieved / total_relevant | What % of relevant docs were retrieved? |
| `precision` | `context_precision@K` | relevant_retrieved / total_retrieved | What % of retrieved docs are relevant? |

### Configuration

```yaml
retrieval:
  metrics:
    - recall      # Calculate context_recall@K
    - precision   # Calculate context_precision@K

  k_values:
    - 1   # Top-1 metrics
    - 5   # Top-5 metrics
    - 10  # Top-10 metrics
```

### Example Output

```
context_recall@5:
  Mean:   0.8500  # Retrieved 85% of relevant docs on average
  
context_precision@5:
  Mean:   0.7000  # 70% of retrieved docs were relevant
```

---

## 🤖 Generation Metrics (RAGAS)

These metrics evaluate **generated answer quality**. They require:
- 🔌 **API Access** - OpenAI or compatible LLM API
- 💰 **Costs Money** - LLM inference is not free
- ⏱️ **Slower** - API calls take time

### LLM-Based Metrics (Expensive)

| Metric | Requires Ground Truth? | Requires LLM? | Cost | What It Measures |
|--------|----------------------|---------------|------|------------------|
| **faithfulness** | ❌ No | ✅ Yes | 💰 | Are facts in answer supported by retrieved context? |
| **answer_relevancy** | ❌ No | ✅ Yes | 💰 | Does answer directly address the question? |
| **answer_correctness** | ✅ Yes | ✅ Yes | 💰💰💰 | Is answer factually correct vs ground truth? |

### Embedding-Based Metrics (Cheaper)

| Metric | Requires Ground Truth? | Requires LLM? | Cost | What It Measures |
|--------|----------------------|---------------|------|------------------|
| **answer_similarity** | ✅ Yes | ❌ No (embeddings only) | 💵 | Semantic similarity to ground truth answer |

---

## 🔍 Metric Details

### Faithfulness

**What:** Checks if the generated answer is grounded in the retrieved context.

**How it works:**
1. LLM extracts claims from the generated answer
2. LLM verifies each claim against retrieved context
3. Score = (verified claims) / (total claims)

**Example:**
```
Context: "Paris is the capital of France. Population: 2.1M"
Answer:  "Paris is France's capital with 2.1M people and has the Eiffel Tower"

Claims:
  ✓ Paris is France's capital (supported)
  ✓ 2.1M people (supported)
  ✗ Has Eiffel Tower (NOT in context)

Faithfulness: 2/3 = 0.67
```

**When to use:** Always! Prevents hallucinations.

---

### Answer Relevancy

**What:** Measures if the answer directly addresses the question.

**How it works:**
1. LLM generates questions from the answer
2. Calculates similarity between generated questions and original question
3. Score = average similarity

**Example:**
```
Question: "What is the capital of France?"
Answer:   "Paris is the capital. France is in Europe and known for wine."

LLM generates from answer:
  - "What is France's capital?" (similarity: 0.95) ✓
  - "Where is France?" (similarity: 0.3) ✗
  - "What is France known for?" (similarity: 0.2) ✗

Relevancy: mean([0.95, 0.3, 0.2]) = 0.48 (low, too much irrelevant info)
```

**When to use:** Ensure answers stay on topic.

---

### Answer Correctness

**What:** Evaluates factual accuracy compared to ground truth.

**How it works:**
1. LLM extracts facts from both answers
2. Calculates F1 score of fact overlap
3. Calculates semantic similarity (embeddings)
4. Score = F1_facts × semantic_similarity

**Example:**
```
Ground Truth: "Paris is France's capital, pop. 2.1M"
Generated:    "Paris is the capital with about 2M people"

Facts comparison:
  TP: Paris is capital ✓
  TP: Population ~2M ✓
  F1: 1.0

Semantic similarity: 0.92

Correctness: 1.0 × 0.92 = 0.92
```

**Cost:** ⚠️ **Most expensive** - Uses LLM to extract and compare facts

**When to use:** When you have ground truth and need precise accuracy measurement.

---

### Answer Similarity

**What:** Semantic similarity between generated and ground truth answers.

**How it works:**
1. Generate embeddings for both answers
2. Calculate cosine similarity
3. Score = cosine_similarity(embed(generated), embed(ground_truth))

**Example:**
```
Ground Truth: "The capital of France is Paris"
Generated:    "Paris is France's capital city"

Embedding similarity: 0.95 (very similar meaning)
```

**Cost:** 💵 **Cheapest** - Only embedding API, no LLM needed

**When to use:** Cheaper alternative to answer_correctness when you need semantic comparison.

---

## 📋 Configuration Examples

### Retrieval Only (Free, Fast)

```yaml
retrieval:
  api_key: "..."
  dataset_id: "..."
  size: 10
  metrics:
    - recall
    - precision
  k_values: [1, 5, 10]
```

**Cost:** $0  
**Speed:** Very fast

---

### Generation (Moderate Cost)

```yaml
generation:
  api_key: "..."
  chat_name: "My Assistant"  # Or provide chat_id
  dynamic_rerank_limit: true
  metrics:
    - recall        # Retrieval metrics still computed from chat doc_ids
    - precision
  k_values: [1, 5, 10]

dataset:
  path: "./dataset/your_dataset.json"

output:
  results_path: "results/evaluation_results.json"
```

**Cost:** ~$0.01-0.10 per sample (depends on answer length)  
**Speed:** Moderate (API calls)

---

### Full Generation Evaluation (High Cost)

```yaml
generation:
  api_key: "..."
  chat_name: "My Assistant"
  dynamic_rerank_limit: true
  metrics:
    - recall
    - precision
  k_values: [1, 5, 10]

dataset:
  path: "./dataset/your_dataset.json"

output:
  results_path: "results/evaluation_results.json"
```

**Cost:** ~$0.05-0.50 per sample  
**Speed:** Slow (multiple LLM calls per sample)

**⚠️ Warning:** For 1000 samples, this could cost $50-500!

---

## 💡 Recommendations

### For Development/Iteration
```yaml
retrieval:
  api_key: "..."
  dataset_id: "..."
  metrics: [recall]
```
- **Free and fast**
- Good for quick retrieval optimization

### For Quality Assurance
```yaml
generation:
  api_key: "..."
  chat_name: "Support QA Assistant"
  metrics: [recall, precision]
  k_values: [1, 5, 10]
```
- **Moderate cost**
- Catches hallucinations and off-topic answers

### For Production Validation
```yaml
generation:
  api_key: "..."
  chat_name: "Prod Assistant"
  metrics: [recall, precision]
  k_values: [1, 5, 10]
  # Evaluate retrieval plus lightweight generation checks downstream
```
- **Balanced cost/quality**
- Good compromise

### For Research/Publication
```yaml
generation:
  api_key: "..."
  chat_name: "Research Assistant"
  metrics: [recall, precision]
  k_values: [1, 5, 10]
```
- **Complete evaluation**
- Use sparingly due to cost

---

## 🔧 Setup Requirements

### Retrieval Metrics
- ✅ RAGFlow API key
- ✅ Dataset with ground truth doc IDs
- ✅ No additional setup

### Generation Metrics
- ✅ RAGFlow chat_id or chat_name
- ✅ Dataset with ground truth answers (for answer_correctness, answer_similarity)
- ✅ OpenAI API key (set via environment):
  ```bash
  export OPENAI_API_KEY="sk-..."
  ```
- ✅ Install RAGAS:
  ```bash
  pip install ragas datasets
  ```

---

## 📊 Interpreting Results

### Good Scores

| Metric | Good Score | What It Means |
|--------|-----------|---------------|
| context_recall@5 | > 0.8 | Retrieving most relevant docs |
| context_precision@5 | > 0.6 | Most retrieved docs are useful |
| faithfulness | > 0.8 | Few hallucinations |
| answer_relevancy | > 0.8 | Stays on topic |
| answer_correctness | > 0.7 | Factually accurate |

### Troubleshooting Low Scores

**Low recall?** → Retrieval isn't finding relevant docs
- Increase `size` parameter
- Adjust `vector_similarity_weight`
- Check embeddings quality

**Low precision?** → Retrieving too much noise
- Decrease `size` parameter
- Add reranking
- Improve query understanding

**Low faithfulness?** → Model is hallucinating
- Check if retrieved context is sufficient
- Try different LLM
- Adjust prompt/temperature

**Low relevancy?** → Answer goes off-topic
- Improve system prompt
- Try different LLM
- Check if context is too broad

---

## 📚 References

- [RAGAS Documentation](https://docs.ragas.io/)
- [RAGAS Metrics Overview](https://docs.ragas.io/en/stable/concepts/metrics/)
- RAGFlow Evaluation: `evaluation/README.md`
- Migration Guide: `evaluation/RAGAS_MIGRATION.md`
