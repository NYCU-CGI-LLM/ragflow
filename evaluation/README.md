# RAGAS Evaluation for RAGFlow

This module provides RAGAS-style evaluation capabilities for the RAGFlow retrieval system using the RAGFlow SDK.

## Features

- **SDK-based evaluation**: Uses RAGFlow Python SDK directly (no HTTP calls)
- **Config-driven**: YAML configuration with proper comment support
- **Comprehensive metrics**: Recall@K, Precision@K, F1@K, NDCG@K, MRR, Hit Rate
- **CLI integration**: Run evaluations via command line
- **Batch processing**: Evaluate entire datasets with progress tracking

## Installation

If you haven't installed RAGFlow dependencies yet:

```bash
# Install all RAGFlow dependencies (includes ruamel-yaml for YAML support)
pip install -e .
```

Or if you only want to run evaluations:

```bash
# Install minimal dependencies for evaluation
pip install ruamel-yaml tqdm
```

The evaluation system uses YAML config files which support proper comments and are more readable than JSON. The required `ruamel-yaml` package is already included in RAGFlow's dependencies.

## Quick Start

### 1. Prepare Your Dataset

Create a JSON file with your evaluation data:

```json
[
  {
    "query": "患者临床信息：主诉：反复胸闷憋喘2年余...",
    "doc_ids": [385, 540, 550, 870, 1001]
  },
  {
    "query": "Another query here",
    "doc_ids": [123, 456, 789]
  }
]
```

**Dataset Format Options:**

1. **List format** (recommended):
```json
[
  {"query": "...", "doc_ids": [...]},
  {"query": "...", "doc_ids": [...]}
]
```

2. **Dict with data key**:
```json
{
  "data": [
    {"query": "...", "doc_ids": [...]},
    {"query": "...", "doc_ids": [...]}
  ]
}
```

3. **With metadata** (like TCM dataset):
```json
[
  {
    "user_id": "123",
    "prompt": "...",
    "expected_doc_id": 593,
    "expected_answer": "...",
    "format_type": "..."
  }
]
```

**Supported field names:**
- **Query**: `query`, `question`, `prompt`, or `content`
- **Ground truth**: `doc_ids`, `document_ids`, `expected_doc_id`, or `expected_doc_ids` (can be single value or list)

### 2. Create Configuration File

Copy `config/example_config.yaml` and modify:

> Choose exactly one evaluation mode per config: `retrieval` **or** `generation`.

```yaml
# Example configuration for RAGFlow evaluation
retrieval:
  api_key: "ragflow-{your_api_key}"
  dataset_id: "{your_dataset_id}"
  # OR use dataset_name instead:
  # dataset_name: "My Knowledge Base"
  
  size: 5  # Number of chunks to retrieve
  vector_similarity_weight: 1.0  # Weight for vector similarity (0-1)
  dynamic_rerank_limit: true  # Use dynamic rerank limit
  # Optional: specify which metrics to calculate
  metrics:
    - recall
    - precision
    - f1
  # Optional: specify K values for @K metrics
  k_values:
    - 1
    - 3
    - 5
  
  # Optional: only add if your server runs on a different URL
  # base_url: "http://your-server:8080"

dataset:
  path: "./dataset/your_dataset.json"
  limit: null  # null = evaluate all samples

output:
  results_path: "results/evaluation_results.json"
```

```yaml
# Example configuration for generation-focused evaluation
generation:
  api_key: "ragflow-{your_api_key}"
  chat_name: "My Assistant"  # or chat_id: "{your_chat_id}"
  # size: 8  # Optional: override retrieval chunk count for this evaluation run
  dynamic_rerank_limit: true
  metrics:
    - recall
    - precision
  k_values:
    - 1
    - 3
    - 5

dataset:
  path: "./dataset/your_dataset.json"

output:
  results_path: "results/generation_results.json"
```

### 3. Run Evaluation

#### Via Standalone Script (Recommended):

```bash
python evaluation/run_evaluation.py --config evaluation/config/minimal_config.yaml
```

#### Via Flask CLI:

```bash
# Using Flask CLI
python -m flask --app api.ragflow_server ragas --config path/to/config.yaml

# Or if you have Flask app running:
flask ragas --config path/to/config.yaml
```

#### Via Python:

```python
from evaluation.ragas_evaluation import run_evaluation

results = run_evaluation('path/to/config.yaml')
```

## Configuration Options

### Retrieval Settings

- `api_key`: Your RAGFlow API key **(required)**
- `dataset_id`: Knowledge base/dataset UUID to retrieve from **(required if `dataset_name` not provided)**
- `dataset_name`: Knowledge base/dataset friendly name to retrieve from **(required if `dataset_id` not provided)**
  - **Tip:** Use `dataset_name` for better readability. The system will automatically resolve it to the UUID.
- `base_url`: RAGFlow server base URL (default: `http://localhost:9380`) **(optional)**
- `size`: Number of chunks to retrieve (default: 5). **Use -1 to retrieve all documents** (up to 10000, limited by Elasticsearch KNN) **(optional)**
  - **Internal Behavior:** The system automatically adjusts the candidate pool size to match `size`:
    - `size <= 64`: Optimized small retrieval (candidate pool ~1024)
    - `size > 64`: Candidate pool = `max(size, 1024)`, capped at 10000
  - **Key Fix:** The system now ensures `size <= 10000` always retrieves from the same ES query (page 1), preventing pagination bugs
  - **Consistency Rules:**
    - ✅ `recall@5` should now be consistent across different `size` values (5, 80, 1027, etc.)
    - ✅ Larger `size` values provide more candidates for reranking, potentially improving quality
  - **Best Practice:** For evaluating `recall@K`, use `size` ≥ K to ensure sufficient candidates. Use `size=-1` to retrieve maximum documents (capped at 10000)
- `vector_similarity_weight`: Weight for vector similarity (0-1, default: 1.0) **(optional)**
- `dynamic_rerank_limit`: Whether to use dynamic rerank limit (true/false, default: true) **(optional)**
- `k`: Number of documents to retrieve (top K, default: 5) **(legacy, prefer `size`)**
- `similarity_threshold`: Minimum similarity score for retrieval (0-1, default: 0.2)
- `rerank_id`: Reranker model ID (optional)
- `keyword`: Enable keyword search (default: false)
- `cross_languages`: List of languages for cross-language search (optional)

### Dataset Settings

- `path`: Path to dataset JSON file
- `limit`: Maximum number of samples to evaluate (null = all)

### Output Settings

- `results_path`: Where to save evaluation results

## Metrics Explained

### Recall@K
Proportion of relevant documents that are retrieved in top K results.
- **Formula**: (Relevant Retrieved) / (Total Relevant)
- **Range**: 0.0 to 1.0 (higher is better)

### Precision@K
Proportion of retrieved documents that are relevant in top K results.
- **Formula**: (Relevant Retrieved) / (Total Retrieved)
- **Range**: 0.0 to 1.0 (higher is better)

### F1@K
Harmonic mean of Precision@K and Recall@K.
- **Formula**: 2 × (Precision × Recall) / (Precision + Recall)
- **Range**: 0.0 to 1.0 (higher is better)

### NDCG@K
Normalized Discounted Cumulative Gain - considers ranking order.
- **Range**: 0.0 to 1.0 (higher is better)

### MRR (Mean Reciprocal Rank)
Reciprocal of the rank of the first relevant document.
- **Range**: 0.0 to 1.0 (higher is better)

### Hit Rate
Whether at least one relevant document was retrieved.
- **Range**: 0.0 or 1.0

## Output Format

Results are saved as JSON with:

```json
{
  "config": {
    "dataset_path": "...",
    "dataset_size": 100,
    "top_k": 5,
    ...
  },
  "aggregate_metrics": {
    "recall@5": {
      "mean": 0.85,
      "median": 0.90,
      "std": 0.12,
      "min": 0.20,
      "max": 1.0
    },
    ...
  },
  "detailed_results": [
    {
      "query": "...",
      "ground_truth_ids": [385, 540],
      "retrieved_ids": [385, 123, 540],
      "retrieval_time": 0.234,
      "metrics": {
        "recall@5": 1.0,
        "precision@5": 0.666,
        ...
      }
    }
  ],
  "timestamp": "2025-10-08 12:34:56"
}
```

## Examples

### Basic Evaluation

```bash
flask ragas --config configs/basic_eval.json
```

### Limited Sample Evaluation (Testing)

Modify config to include `limit`:
```json
{
  "dataset": {
    "path": "./dataset/large_dataset.json",
    "limit": 10
  }
}
```

### Evaluation with Reranking

```json
{
  "retrieval": {
    "rerank_id": "your-reranker-id",
    ...
  }
}
```

## Troubleshooting

### "Cannot extract dataset_id from API URL"
Ensure your `api_url` follows the format:
```
http://localhost:9380/api/v1/retrieval_simple_rag/{dataset_id}
```

### "Dataset file not found"
Check that the `path` in config is relative to where you run the command, or use absolute paths.

### "API key authentication failed"
Verify your API key is correct and has proper permissions.

## Integration with CI/CD

Example GitHub Actions workflow:

```yaml
- name: Run RAGAS Evaluation
  run: |
    python -m flask --app api.ragflow_server ragas --config configs/ci_eval.json
    
- name: Check Results
  run: |
    python scripts/check_metrics.py results/evaluation_results.json
```

## Development

### Using RAGAS Metrics

The evaluation system now uses **RAGAS framework metrics** for both retrieval and generation evaluation.

**Retrieval Metrics (ID-based, FREE):**
- `context_precision@K` - What fraction of retrieved docs are relevant?
- `context_recall@K` - What fraction of relevant docs were retrieved?

**Generation Metrics:**
- `ExactMatch` - Case-insensitive string comparison with ground truth

See `RAGAS_MIGRATION.md` for full details and migration guide.

## License

Copyright 2024 The InfiniFlow Authors. Licensed under Apache 2.0.
