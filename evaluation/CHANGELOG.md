# RAGAS Evaluation - Changelog

## Version 1.2.0 - Hardcoded Default URL

### Changes

**`base_url` is now optional with default value:**
- Default: `http://localhost:9380`
- Only specify if your RAGFlow runs on a different URL
- Simplifies config for standard local setups

**Minimal config example:**
```json
{
  "retrieval": {
    "api_key": "ragflow-xxx",
    "dataset_id": "your-dataset-id"
  },
  "dataset": {
    "path": "./dataset.json"
  },
  "output": {
    "results_path": "results/results.json"
  }
}
```

## Version 1.1.0 - Simplified Configuration

### Breaking Changes

**Config format simplified** - Removed redundant `api_url` field:

**Old format (deprecated):**
```json
{
  "retrieval": {
    "api_url": "http://localhost:9380/api/v1/retrieval_simple_rag/dataset_id",
    "api_key": "ragflow-xxx"
  }
}
```

**New format:**
```json
{
  "retrieval": {
    "base_url": "http://localhost:9380",
    "dataset_id": "dataset_id",
    "api_key": "ragflow-xxx"
  }
}
```

### New Features

**Enhanced dataset format support:**
- Now supports `prompt` field (in addition to `query`, `question`, `content`)
- Now supports `expected_doc_id` (singular) in addition to `doc_ids` (plural)
- Automatically converts single doc_id to list format
- Preserves metadata fields like `user_id`, `expected_answer`, `format_type` for analysis

**Supported field variations:**
- Query fields: `query`, `question`, `prompt`, `content`
- Ground truth fields: `doc_ids`, `document_ids`, `expected_doc_id`, `expected_doc_ids`

### Migration Guide

If you have existing configs using the old format:

1. **Split api_url into base_url + dataset_id:**
   ```python
   # Old
   api_url = "http://localhost:9380/api/v1/retrieval_simple_rag/abc123"
   
   # New
   base_url = "http://localhost:9380"
   dataset_id = "abc123"
   ```

2. **Update your config.json:**
   ```bash
   # Replace
   "api_url": "http://localhost:9380/api/v1/retrieval_simple_rag/abc123"
   
   # With
   "base_url": "http://localhost:9380",
   "dataset_id": "abc123"
   ```

3. **Dataset format is backwards compatible** - no changes needed if you already use:
   - `query` + `doc_ids` fields
   - `question` + `document_ids` fields

### Why This Change?

1. **Simpler**: Uses SDK directly, doesn't need full API endpoint path
2. **Clearer**: Separates server location from dataset identifier
3. **More flexible**: Easy to switch between datasets without changing URLs
4. **Consistent**: Matches RAGFlow SDK initialization pattern

### Version 1.0.0 - Initial Release

- SDK-based retrieval evaluation
- Comprehensive metrics (Recall, Precision, F1, NDCG, MRR, Hit Rate)
- Config-driven evaluation
- CLI integration
- Result analysis tools
