# RAGAS Evaluation - Quick Start Guide

This guide will help you run your first RAGAS evaluation in 5 minutes.

## Step 1: Prepare Your Configuration

Edit `config/example_config.yaml` with your actual values:

```yaml
# Evaluation configuration
retrieval:
  api_key: "ragflow-YOUR_API_KEY"
  dataset_id: "YOUR_DATASET_ID"
  size: 5  # Number of chunks to retrieve
  dynamic_rerank_limit: true

dataset:
  path: "./dataset/your_dataset.json"

output:
  results_path: "results/evaluation_results.json"
```

**How to get your values:**

1. **API Key**: From RAGFlow settings → API Keys
2. **Dataset ID**: The knowledge base/dataset ID from RAGFlow
3. **Size**: Number of chunks to retrieve (typically 5-10)

**Note:** The config assumes RAGFlow runs on `http://localhost:9380`. If your setup is different, add:
```yaml
retrieval:
  base_url: "http://your-server:port"
  api_key: "ragflow-YOUR_API_KEY"
  # ... rest of config
```

## Step 2: Prepare Your Dataset

Create a JSON file with your test queries and ground truth document IDs:

**Format 1: Simple format**
```json
[
  {
    "query": "Your test query here",
    "doc_ids": [123, 456, 789]
  },
  {
    "query": "Another test query",
    "doc_ids": [111, 222, 333]
  }
]
```

**Format 2: With additional metadata** (like your TCM dataset)
```json
[
  {
    "user_id": "438612",
    "prompt": "患者临床信息：...",
    "expected_answer": "气虚不摄证",
    "format_type": "reading_comprehension_five",
    "expected_doc_id": 593
  }
]
```

**Supported Field Names:**
- **Query text**: `query`, `question`, `prompt`, or `content`
- **Ground truth doc IDs**: `doc_ids`, `document_ids`, `expected_doc_id`, or `expected_doc_ids`
- Ground truth can be a single integer or a list of integers
- Additional fields (like `user_id`, `expected_answer`, etc.) are preserved but not used in evaluation

## Step 3: Run Evaluation

### Option A: Using Standalone Script (Recommended)

```bash
cd /path/to/ragflow_nv28
python evaluation/run_evaluation.py --config evaluation/config/minimal_config.yaml
```

### Option B: Using Flask CLI

```bash
cd /path/to/ragflow_nv28
python -m flask --app api.ragflow_server ragas --config evaluation/config/example_config.yaml
```

### Option C: From Python Code

```python
from evaluation.ragas_evaluation import run_evaluation

results = run_evaluation('config/minimal_config.yaml')
print(f"Mean Recall@5: {results['aggregate_metrics']['recall@5']['mean']}")
```

## Step 4: View Results

The evaluation will print a summary to the console and save detailed results to the path specified in your config.

### Example Output:

```
================================================================================
RAGAS Evaluation Summary
================================================================================

Dataset: ./dataset/tcm_sd_test_rc_direct.json
Samples evaluated: 100
Top K: 5

--------------------------------------------------------------------------------
Aggregate Metrics
--------------------------------------------------------------------------------

recall@5:
  Mean:   0.8500
  Median: 0.9000
  Std:    0.1200
  Min:    0.2000
  Max:    1.0000

precision@5:
  Mean:   0.7800
  Median: 0.8000
  Std:    0.1500
  Min:    0.2000
  Max:    1.0000
```

## Step 5: Analyze Results

Use the analysis script to dig deeper:

```bash
# Show summary
python analyze_results.py results/evaluation_results.json

# Show 10 worst performing queries
python analyze_results.py results/evaluation_results.json --worst-queries 10

# Show 5 best performing queries
python analyze_results.py results/evaluation_results.json --best-queries 5 --metric recall@5

# Export to CSV for Excel/spreadsheet analysis
python analyze_results.py results/evaluation_results.json --export-csv results.csv

# Compare multiple evaluations
python analyze_results.py results/eval1.json results/eval2.json --compare
```

## Common Issues & Solutions

### Issue: "Missing 'dataset_id' or 'dataset_name' in retrieval config"
**Solution**: Provide either `dataset_id` (UUID) or `dataset_name` (friendly name):

**Option A: Using dataset_id (UUID)**
```yaml
retrieval:
  api_key: "ragflow-YOUR_API_KEY"
  dataset_id: "6ba750d6a34e11f0a29a37d227f1c0af"
```

**Option B: Using dataset_name (recommended for readability)**
```yaml
retrieval:
  api_key: "ragflow-YOUR_API_KEY"
  dataset_name: "My Knowledge Base"
```

### Issue: RAGFlow server not on default port
**Solution**: Add `base_url` to your config:
```yaml
retrieval:
  base_url: "http://localhost:8080"
  api_key: "ragflow-YOUR_API_KEY"
  # ... rest of config
```

### Issue: "Dataset file not found"
**Solution**: Use absolute paths or paths relative to where you run the command:
```yaml
dataset:
  path: "/absolute/path/to/dataset.json"
```

### Issue: Authentication failed
**Solution**: Verify your API key is correct and has proper permissions in RAGFlow

### Issue: No documents retrieved
**Solution**: 
- Check `similarity_threshold` - lower it if too high (try 0.1)
- Verify dataset_id is correct
- Check if documents exist in the knowledge base

## Quick Test with Example Data

Want to test immediately? Use the example dataset:

```bash
# 1. Edit config/minimal_config.yaml with your API key and dataset ID
nano config/minimal_config.yaml

# 2. Run with example dataset
python run_evaluation.py --config config/minimal_config.yaml

# 3. Analyze results
python analyze_results.py results/evaluation_results.json --worst-queries 3
```

## Understanding Metrics

- **Recall@K**: What % of relevant docs did we find? (Higher is better)
- **Precision@K**: What % of retrieved docs are relevant? (Higher is better)
- **F1@K**: Balance between precision and recall (Higher is better)
- **NDCG@K**: How well are relevant docs ranked? (Higher is better)
- **MRR**: How quickly do we find the first relevant doc? (Higher is better)
- **Hit Rate**: Did we find at least one relevant doc? (1.0 = yes, 0.0 = no)

## Next Steps

1. **Start small**: Test with 10-20 samples first (use `"limit": 20` in config)
2. **Iterate**: Adjust retrieval parameters and re-evaluate
3. **Compare**: Save results and compare different configurations
4. **Analyze**: Use the analysis script to find patterns in failures

## Tips for Better Evaluation

1. **Quality over quantity**: Better to have accurate ground truth for 50 queries than questionable data for 500
2. **Representative samples**: Include easy, medium, and hard queries
3. **Document your config**: Save configs with descriptive names like `eval_config_v1_baseline.yaml`
4. **Track changes**: Keep a log of what changed between evaluations
5. **Use version control**: Commit configs and results to git

## Getting Help

- Check the main README.md for detailed documentation
- Review `config/example_config.yaml` and example dataset files for reference
- Check RAGFlow documentation for API details

Happy evaluating! 🚀
