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

"""
RAGAS-based metrics for retrieval and generation evaluation
"""

from typing import List, Dict, Any
import statistics
import asyncio
from ragas.dataset_schema import SingleTurnSample
from ragas.metrics import (
    context_precision,
    context_recall,
    ExactMatch,
)


class RagasMetrics:
    """
    Wrapper for RAGAS metrics supporting both retrieval and generation evaluation
    """
    
    def __init__(self):
        """
        Initialize RAGAS metrics
        """
        # Available retrieval metrics
        self.retrieval_metrics = {
            'context_precision': context_precision,
            'context_recall': context_recall,
        }
        
        # Available generation metric (only exact match)
        self.exact_match_scorer = ExactMatch()
    
    def evaluate_retrieval_id_based(
        self,
        question: str,
        retrieved_doc_ids: List[int],
        ground_truth_doc_ids: List[int],
        metrics: List[str] = None
    ) -> Dict[str, float]:
        """
        Evaluate retrieval quality using ID-based metrics (fast, no LLM)
        
        Args:
            question: Query text
            retrieved_doc_ids: List of retrieved document IDs
            ground_truth_doc_ids: List of ground truth document IDs
            metrics: List of metric names to calculate (default: all)
        
        Returns:
            Dictionary with metric scores
        """
        if metrics is None:
            metrics = ['context_precision', 'context_recall']
        
        # Convert to sets for comparison
        retrieved_set = set(retrieved_doc_ids)
        ground_truth_set = set(ground_truth_doc_ids)
        
        results = {}
        
        if 'context_precision' in metrics:
            # Context Precision: What fraction of retrieved docs are relevant?
            # = (relevant retrieved) / (total retrieved)
            if len(retrieved_doc_ids) > 0:
                relevant_retrieved = len(retrieved_set.intersection(ground_truth_set))
                results['context_precision'] = relevant_retrieved / len(retrieved_doc_ids)
            else:
                results['context_precision'] = 0.0
        
        if 'context_recall' in metrics:
            # Context Recall: What fraction of relevant docs were retrieved?
            # = (relevant retrieved) / (total relevant)
            if len(ground_truth_doc_ids) > 0:
                relevant_retrieved = len(retrieved_set.intersection(ground_truth_set))
                results['context_recall'] = relevant_retrieved / len(ground_truth_doc_ids)
            else:
                results['context_recall'] = 0.0
        
        return results
    
    def evaluate_retrieval_at_k(
        self,
        question: str,
        retrieved_doc_ids: List[int],
        ground_truth_doc_ids: List[int],
        k_values: List[int] = None,
        metrics: List[str] = None
    ) -> Dict[str, float]:
        """
        Evaluate retrieval quality at different K values
        
        Args:
            question: Query text
            retrieved_doc_ids: List of retrieved document IDs
            ground_truth_doc_ids: List of ground truth document IDs
            k_values: List of K values to evaluate at (default: [1, 3, 5, 10])
            metrics: List of metric names (default: ['context_precision', 'context_recall'])
        
        Returns:
            Dictionary with metrics at each K value
        """
        if k_values is None:
            k_values = [1, 3, 5, 10]
        
        if metrics is None:
            metrics = ['context_precision', 'context_recall']
        
        results = {}
        
        for k in k_values:
            # Evaluate at top-K
            k_metrics = self.evaluate_retrieval_id_based(
                question,
                retrieved_doc_ids[:k],
                ground_truth_doc_ids,
                metrics
            )
            
            # Add @K suffix to metric names
            for metric_name, value in k_metrics.items():
                results[f'{metric_name}@{k}'] = value
        
        return results
    
    def evaluate_generation(
        self,
        questions: List[str],
        answers: List[str],
        contexts: List[List[str]],
        ground_truths: List[str] = None,
        metrics: List[str] = None
    ) -> Dict[str, Any]:
        """
        Evaluate generation quality using RAGAS metrics
        
        Args:
            questions: List of questions
            answers: List of generated answers
            contexts: List of retrieved contexts (list of chunks for each question)
            ground_truths: List of ground truth answers (optional, needed for answer_correctness)
            metrics: List of metric names to evaluate (default: all available)
        
        Returns:
            Dictionary with aggregate metric scores and per-sample results
        """
        if metrics is None:
            metrics = ['faithfulness', 'answer_relevancy']
            if ground_truths is not None:
                metrics.extend(['answer_correctness', 'answer_similarity'])
        
        # Prepare dataset for RAGAS
        data = {
            'question': questions,
            'answer': answers,
            'contexts': contexts,
        }
        
        if ground_truths is not None and ('answer_correctness' in metrics or 'answer_similarity' in metrics):
            data['ground_truth'] = ground_truths
        
        dataset = Dataset.from_dict(data)
        
        # Select metrics to evaluate
        selected_metrics = []
        for metric_name in metrics:
            if metric_name in self.generation_metrics:
                selected_metrics.append(self.generation_metrics[metric_name])
        
        if not selected_metrics:
            return {'error': 'No valid metrics selected'}
        
        # Run RAGAS evaluation
        try:
            result = evaluate(
                dataset,
                metrics=selected_metrics,
            )
            
            return {
                'aggregate': result,
                'detailed': result.to_pandas().to_dict('records')
            }
        except Exception as e:
            return {'error': str(e)}
    
    async def evaluate_generation_async(
        self,
        answers: List[str],
        ground_truths: List[str]
    ) -> Dict[str, Any]:
        """
        Evaluate generation quality using ExactMatch metric
        
        Args:
            answers: List of generated answers
            ground_truths: List of ground truth answers
        
        Returns:
            Dictionary containing exact match scores
        """
        scores = []
        
        for answer, ground_truth in zip(answers, ground_truths):
            sample = SingleTurnSample(
                response=answer,
                reference=ground_truth
            )
            score = await self.exact_match_scorer.single_turn_ascore(sample)
            scores.append(float(score))
        
        # Calculate statistics
        mean_score = statistics.mean(scores) if scores else 0.0
        
        return {
            'exact_match': {
                'mean': mean_score,
                'scores': scores,
                'num_samples': len(scores)
            }
        }
    
    def evaluate_generation(
        self,
        answers: List[str],
        ground_truths: List[str]
    ) -> Dict[str, Any]:
        """
        Synchronous wrapper for evaluate_generation_async
        
        Args:
            answers: List of generated answers
            ground_truths: List of ground truth answers
        
        Returns:
            Dictionary containing exact match scores
        """
        return asyncio.run(self.evaluate_generation_async(answers, ground_truths))
    
    def calculate_aggregate_metrics(self, all_metrics: List[Dict[str, float]]) -> Dict[str, Any]:
        """
        Calculate aggregate statistics across all queries
        
        Args:
            all_metrics: List of metric dictionaries for each query
        
        Returns:
            Dictionary with mean, median, min, max for each metric
        """
        if not all_metrics:
            return {}
        
        metric_names = set()
        for metrics in all_metrics:
            metric_names.update(metrics.keys())
        
        aggregated = {}
        
        for metric_name in metric_names:
            values = [m[metric_name] for m in all_metrics if metric_name in m]
            if values:
                aggregated[metric_name] = {
                    'mean': statistics.mean(values),
                    'median': statistics.median(values),
                    'min': min(values),
                    'max': max(values),
                    'std': statistics.stdev(values) if len(values) > 1 else 0.0
                }
        
        return aggregated


def evaluate_sample_retrieval(
    question: str,
    retrieved_doc_ids: List[int],
    ground_truth_doc_ids: List[int],
    k_values: List[int] = None,
    metrics: List[str] = None
) -> Dict[str, float]:
    """
    Convenience function to evaluate a single retrieval sample
    
    Args:
        question: Query text
        retrieved_doc_ids: List of retrieved document IDs
        ground_truth_doc_ids: List of ground truth document IDs
        k_values: List of K values (default: [1, 3, 5, 10])
        metrics: List of metrics to calculate (default: all)
    
    Returns:
        Dictionary with metric scores
    """
    evaluator = RagasMetrics()
    return evaluator.evaluate_retrieval_at_k(
        question,
        retrieved_doc_ids,
        ground_truth_doc_ids,
        k_values,
        metrics
    )

