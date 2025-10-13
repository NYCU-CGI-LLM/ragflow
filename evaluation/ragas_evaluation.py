#
#  Copyright 2024 The InfiniFlow Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#

import json
import sys
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
from tqdm import tqdm

# Add SDK path to import
sdk_path = Path(__file__).parent.parent / "sdk" / "python"
if str(sdk_path) not in sys.path:
    sys.path.insert(0, str(sdk_path))

# Patch importlib.metadata to handle missing ragflow_sdk package
import importlib.metadata
original_version = importlib.metadata.version

def patched_version(distribution_name):
    if distribution_name == "ragflow_sdk":
        return "0.1.0-dev"  # Fallback version for local development
    return original_version(distribution_name)

importlib.metadata.version = patched_version

# Now import RAGFlow SDK
from ragflow_sdk import RAGFlow

# Restore original function
importlib.metadata.version = original_version

from .config import EvaluationConfig
from .ragas_metrics import RagasMetrics


class RagasEvaluator:
    """
    RAGAS-style evaluator for RAGFlow retrieval system
    Uses RAGFlow SDK directly for evaluation
    """
    
    def __init__(self, config: EvaluationConfig):
        self.config = config
        self.mode = config.mode
        self.ragflow_client = None
        self.dataset = []
        self.results = []
        self.resolved_dataset_id = None  # Will be set during initialization
        self.ragas_metrics = RagasMetrics()  # Initialize RAGAS metrics evaluator
        self._chat_id: Optional[str] = None
        self.chat = None
        self.chat_dataset_ids: List[str] = []
    
    def _initialize_client(self):
        """Initialize RAGFlow SDK client and resolve dataset ID"""
        base_url = self.config.base_url
        self.ragflow_client = RAGFlow(
            api_key=self.config.api_key,
            base_url=base_url
        )
        print("Initialized RAGFlow client")
        print(f"  Base URL: {base_url}" + (" (default)" if base_url == "http://localhost:9380" else ""))
        
        if self.mode == "retrieval":
            if self.config.dataset_id:
                self.resolved_dataset_id = self.config.dataset_id
                print(f"  Dataset ID: {self.resolved_dataset_id}")
            elif self.config.dataset_name:
                print(f"  Dataset Name: {self.config.dataset_name}")
                print("  Resolving dataset name to ID...")
                try:
                    dataset = self.ragflow_client.get_dataset(self.config.dataset_name)
                    self.resolved_dataset_id = dataset.id
                    print(f"  ✓ Resolved to Dataset ID: {self.resolved_dataset_id}")
                except Exception as exc:
                    raise ValueError(
                        f"Failed to find dataset with name '{self.config.dataset_name}': {exc}"
                    ) from exc
            else:
                raise ValueError("Retrieval config requires either 'dataset_id' or 'dataset_name'.")
        else:
            chat_identifier = self.config.chat_id
            chat_name = self.config.chat_name
            if chat_identifier:
                print(f"  Chat ID: {chat_identifier}")
                chat = self.ragflow_client.get_chat(chat_id=chat_identifier)
            else:
                print(f"  Chat Name: {chat_name}")
                print("  Resolving chat name to ID...")
                try:
                    chat = self.ragflow_client.get_chat(name=chat_name)
                    print(f"  ✓ Resolved to Chat ID: {chat.id}")
                except Exception as exc:
                    raise ValueError(
                        f"Failed to find chat with name '{chat_name}': {exc}"
                    ) from exc
            
            self.chat = chat
            self._chat_id = chat.id
            # Record datasets linked to the chat (if any)
            dataset_ids = getattr(chat, "dataset_ids", []) or []
            self.chat_dataset_ids = list(dataset_ids)
            if self.chat_dataset_ids:
                self.resolved_dataset_id = self.chat_dataset_ids[0]
                print(f"  Linked dataset IDs: {self.chat_dataset_ids}")
    
    def load_dataset(self):
        """Load evaluation dataset from JSON file"""
        if not self.config.dataset_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {self.config.dataset_path}")
        
        with open(self.config.dataset_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Support both list format and dict format
        if isinstance(data, list):
            self.dataset = data
        elif isinstance(data, dict) and 'data' in data:
            self.dataset = data['data']
        else:
            self.dataset = [data]
        
        # Apply limit if specified
        if self.config.dataset_limit:
            self.dataset = self.dataset[:self.config.dataset_limit]
            print(f"Loaded {len(self.dataset)} samples from dataset (limited from total)")
        else:
            print(f"Loaded {len(self.dataset)} samples from dataset")
    
    def _retrieve_for_query(self, query: str) -> List[int]:
        """
        Retrieve documents for a query using RAGFlow SDK
        
        Args:
            query: Query text
        
        Returns:
            List of document IDs
        """
        if self.mode != "retrieval":
            raise RuntimeError("Internal error: _retrieve_for_query should not be called in generation mode.")
        try:
            # Use the new simple_rag_retrieval method which directly returns doc IDs
            doc_ids = self.ragflow_client.simple_rag_retrieval(
                dataset_id=self.resolved_dataset_id,
                question=query,
                size=self.config.size,
                vector_similarity_weight=self.config.vector_similarity_weight,
                dynamic_rerank_limit=self.config.dynamic_rerank_limit
            )
            
            return doc_ids if doc_ids else []
        
        except Exception as e:
            print(f"\n⚠️  Error during retrieval for query '{query[:50]}...': {str(e)}")
            import traceback
            traceback.print_exc()
            return []
    
    def _generate_answer_for_query(
        self,
        query: str,
        sample_idx: Optional[int] = None
    ) -> tuple[str, List[str], List[int]]:
        """
        Generate answer for a query using RAGFlow chat API
        
        Args:
            query: Query text
        
        Returns:
            Tuple of (generated_answer, retrieved_contexts, retrieved_doc_ids)
        """
        try:
            if self.chat is None:
                raise RuntimeError(
                    "Generation evaluation requires a resolved chat. "
                    "Ensure your config provides 'chat_id' or 'chat_name' under the generation section."
                )

            response = self.chat.chat_simple_rag(
                messages=[{"role": "user", "content": query}],
                dynamic_rerank_limit=self.config.dynamic_rerank_limit
            )
            
            generated_answer = response['choices'][0]['message']['content']
            retrieved_doc_ids: List[int] = []
            for doc_id in response.get('doc_ids') or []:
                try:
                    retrieved_doc_ids.append(int(doc_id))
                except (TypeError, ValueError):
                    retrieved_doc_ids.append(doc_id)
            # Preserve order while removing duplicates
            deduped_ids = []
            for doc_id in retrieved_doc_ids:
                if doc_id not in deduped_ids:
                    deduped_ids.append(doc_id)
            retrieved_doc_ids = deduped_ids
            
            # Extract contexts from retrieved docs
            # For now, we'll use placeholder contexts
            # In a full implementation, you'd fetch the actual chunk contents
            contexts = [f"Retrieved document {doc_id}" for doc_id in retrieved_doc_ids]
            
            # Debug: show generated answer for first few samples
            if sample_idx is not None and sample_idx < 3:
                print(f"   LLM answer: {json.dumps(generated_answer)}")
            
            return generated_answer, contexts, retrieved_doc_ids
        
        except Exception as e:
            print(f"\n⚠️  Error during generation: {str(e)}")
            return "", [], []
    
    def evaluate_sample(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate a single sample
        
        Args:
            sample: Sample dict with query and ground truth doc_ids
                Supports multiple field name variations:
                - query/question/prompt/content for the query text
                - doc_ids/document_ids/expected_doc_id for ground truth
        
        Returns:
            Evaluation result with metrics
        """
        # Support multiple query field names
        query = (sample.get('query') or 
                sample.get('question') or 
                sample.get('prompt') or 
                sample.get('content', ''))
        
        # Support multiple ground truth field names
        # Handle both list and single values
        ground_truth_ids = None
        if 'doc_ids' in sample:
            ground_truth_ids = sample['doc_ids']
        elif 'document_ids' in sample:
            ground_truth_ids = sample['document_ids']
        elif 'expected_doc_id' in sample:
            ground_truth_ids = sample['expected_doc_id']
        elif 'expected_doc_ids' in sample:
            ground_truth_ids = sample['expected_doc_ids']
        else:
            ground_truth_ids = []
        
        # Convert to list if single value
        if not isinstance(ground_truth_ids, list):
            ground_truth_ids = [ground_truth_ids]
        
        # Normalize ground_truth_ids - convert to int if possible, otherwise keep as string
        normalized_gt_ids = []
        for doc_id in ground_truth_ids:
            if doc_id is not None:
                try:
                    normalized_gt_ids.append(int(doc_id))
                except (ValueError, TypeError):
                    normalized_gt_ids.append(str(doc_id))
        ground_truth_ids = normalized_gt_ids
        
        sample_idx = len(self.results)
        run_generation = self.config.generation_enabled
        
        retrieved_ids: List[int] = []
        retrieval_time = 0.0
        generated_answer: Optional[str] = None
        contexts: List[str] = []
        
        if run_generation:
            start_time = time.time()
            generated_answer, contexts, retrieved_ids = self._generate_answer_for_query(
                query,
                sample_idx=sample_idx
            )
            retrieval_time = time.time() - start_time
            retrieved_ids = retrieved_ids or []
        else:
            start_time = time.time()
            retrieved_ids = self._retrieve_for_query(query)
            retrieval_time = time.time() - start_time
        
        # Debug: Print first few samples to help diagnose ID mismatch
        if sample_idx < 3:  # Show first 3 samples
            print(f"\n🔍 Debug Sample {sample_idx + 1}:")
            print(f"   Ground truth IDs: {ground_truth_ids} (type: {type(ground_truth_ids[0]) if ground_truth_ids else 'N/A'})")
            # print(f"   Retrieved IDs: {retrieved_ids[:3]}{'...' if len(retrieved_ids) > 3 else ''} (type: {type(retrieved_ids[0]) if retrieved_ids else 'N/A'})")
            print(f"   Retrieved IDs: {retrieved_ids} (type: {type(retrieved_ids[0]) if retrieved_ids else 'N/A'})")
            if ground_truth_ids and retrieved_ids:
                matches = set(ground_truth_ids).intersection(set(retrieved_ids))
                print(f"   Matches: {matches if matches else 'NONE - ID types/values do not match!'}")
        
        # Calculate RAGAS retrieval metrics (ID-based)
        k_values = self.config.k_values or [1, 3, 5, 10]
        
        # Map config metrics to RAGAS metric names
        ragas_metrics = []
        if self.config.metrics:
            for metric in self.config.metrics:
                # Support both old names (recall, precision) and new RAGAS names
                if metric in ['recall', 'context_recall']:
                    ragas_metrics.append('context_recall')
                elif metric in ['precision', 'context_precision']:
                    ragas_metrics.append('context_precision')
        else:
            # Default: calculate both if no metrics specified
            ragas_metrics = ['context_precision', 'context_recall']
        
        # Remove duplicates while preserving order
        ragas_metrics = list(dict.fromkeys(ragas_metrics))
        
        retrieval_metrics = self.ragas_metrics.evaluate_retrieval_at_k(
            question=query,
            retrieved_doc_ids=retrieved_ids,
            ground_truth_doc_ids=ground_truth_ids,
            k_values=k_values,
            metrics=ragas_metrics
        )
        
        result = {
            'query': query,
            'ground_truth_ids': ground_truth_ids,
            'retrieved_ids': retrieved_ids,
            'retrieval_time': retrieval_time,
            'retrieval_metrics': retrieval_metrics
        }
        
        # Add generation evaluation if enabled
        if run_generation:
            if generated_answer is not None:
                result['generated_answer'] = generated_answer
            if contexts:
                result['retrieved_contexts'] = contexts
            result['generation_time'] = retrieval_time
            
            ground_truth_answer = (
                sample.get('expected_answer')
                or sample.get('ground_truth')
                or sample.get('answer')
            )
            
            if ground_truth_answer:
                result['ground_truth_answer'] = ground_truth_answer
                result['generation_data'] = {
                    'question': query,
                    'answer': generated_answer or "",
                    'contexts': contexts,
                    'ground_truth': ground_truth_answer
                }
        
        return result
    
    def evaluate(self) -> Dict[str, Any]:
        """
        Run evaluation on entire dataset
        
        Returns:
            Complete evaluation results with aggregate metrics
        """
        print("Starting RAGAS evaluation...")
        
        # Initialize client and load dataset
        self._initialize_client()
        self.load_dataset()
        
        # Evaluate each sample
        self.results = []
        for sample in tqdm(self.dataset, desc="Evaluating samples"):
            result = self.evaluate_sample(sample)
            self.results.append(result)
        
        # Calculate aggregate retrieval metrics
        all_retrieval_metrics = [r['retrieval_metrics'] for r in self.results]
        aggregate_retrieval_metrics = self.ragas_metrics.calculate_aggregate_metrics(all_retrieval_metrics)
        
        aggregate_metrics = {
            'retrieval': aggregate_retrieval_metrics
        }
        
        # Calculate generation metrics if enabled (ExactMatch only)
        if self.config.generation_enabled:
            generation_data_list = [r['generation_data'] for r in self.results if 'generation_data' in r]
            
            if generation_data_list:
                print("\nCalculating ExactMatch metric...")
                
                # Prepare data for ExactMatch evaluation
                answers = [d['answer'] for d in generation_data_list]
                ground_truths = [d['ground_truth'] for d in generation_data_list]
                
                # Run ExactMatch metric
                generation_results = self.ragas_metrics.evaluate_generation(
                    answers=answers,
                    ground_truths=ground_truths
                )
                
                if 'error' not in generation_results:
                    aggregate_metrics['generation'] = generation_results
                    
                    # Add individual scores to results
                    exact_match_scores = generation_results['exact_match']['scores']
                    for i, result in enumerate(self.results):
                        if 'generation_data' in result and i < len(exact_match_scores):
                            result['generation_metrics'] = {
                                'exact_match': exact_match_scores[i]
                            }
                else:
                    print(f"\n⚠️  Generation metrics calculation failed: {generation_results.get('error', 'Unknown error')}")
        
        # Compile final results
        config_summary = {
            'mode': self.mode,
            'dataset_path': str(self.config.dataset_path),
            'dataset_size': len(self.dataset),
            'base_url': self.config.base_url,
            'dynamic_rerank_limit': self.config.dynamic_rerank_limit,
            'generation_enabled': self.config.generation_enabled,
        }
        if self.resolved_dataset_id:
            config_summary['dataset_id'] = self.resolved_dataset_id
        if self.mode == 'retrieval':
            if self.config.dataset_name:
                config_summary['dataset_name'] = self.config.dataset_name
            config_summary.update({
                'size': self.config.size,
                'top_k': self.config.top_k,
                'similarity_threshold': self.config.similarity_threshold,
                'vector_similarity_weight': self.config.vector_similarity_weight,
            })
        else:
            config_summary.update({
                'chat_id': self._chat_id,
                'chat_name': self.config.chat_name,
                'linked_dataset_ids': self.chat_dataset_ids,
            })
        
        evaluation_results = {
            'config': config_summary,
            'aggregate_metrics': aggregate_metrics,
            'detailed_results': self.results,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        return evaluation_results
    
    def save_results(self, results: Dict[str, Any]):
        """Save evaluation results to file"""
        output_path = self.config.results_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"\nResults saved to: {output_path}")
    
    def print_summary(self, results: Dict[str, Any]):
        """Print summary of evaluation results"""
        print("\n" + "="*80)
        print("RAGAS Evaluation Summary")
        print("="*80)
        
        config = results['config']
        print(f"\nMode: {config.get('mode', 'retrieval').title()}")
        print(f"Dataset: {config['dataset_path']}")
        print(f"Samples evaluated: {config['dataset_size']}")
        if config.get('mode') == 'retrieval':
            print(f"Size: {config.get('size')}")
            print(f"Vector weight: {config.get('vector_similarity_weight')}")
        else:
            print(f"Chat ID: {config.get('chat_id')}")
            print(f"Chat Name: {config.get('chat_name')}")
            print(f"Linked datasets: {config.get('linked_dataset_ids') or '[]'}")
        print(f"Generation enabled: {config.get('generation_enabled', False)}")
        
        aggregate = results['aggregate_metrics']
        
        # Print retrieval metrics
        if 'retrieval' in aggregate:
            print("\n" + "-"*80)
            print("Retrieval Metrics (RAGAS ID-based)")
            print("-"*80)
            
            for metric_name, stats in sorted(aggregate['retrieval'].items()):
                print(f"\n{metric_name}:")
                print(f"  Mean:   {stats['mean']:.4f}")
                print(f"  Median: {stats['median']:.4f}")
                print(f"  Std:    {stats['std']:.4f}")
                print(f"  Min:    {stats['min']:.4f}")
                print(f"  Max:    {stats['max']:.4f}")
        
        # Print generation metrics if available
        if 'generation' in aggregate:
            print("\n" + "-"*80)
            print("Generation Metrics")
            print("-"*80)
            
            gen_metrics = aggregate['generation']
            exact_match = gen_metrics.get('exact_match', {})
            mean_score = exact_match.get('mean', 0)
            num_samples = exact_match.get('num_samples', 0)
            
            print(f"\nExactMatch:")
            print(f"  Mean: {mean_score:.4f}")
            print(f"  Samples: {num_samples}")
        
        print("\n" + "="*80)
    
    def run(self):
        """Main entry point to run complete evaluation"""
        try:
            results = self.evaluate()
            self.print_summary(results)
            self.save_results(results)
            return results
        except Exception as e:
            print(f"Error during evaluation: {str(e)}")
            import traceback
            traceback.print_exc()
            raise


def run_evaluation(config_path: str) -> Dict[str, Any]:
    """
    Convenience function to run evaluation from config file
    
    Args:
        config_path: Path to configuration JSON file
    
    Returns:
        Evaluation results
    """
    config = EvaluationConfig(config_path)
    evaluator = RagasEvaluator(config)
    return evaluator.run()
