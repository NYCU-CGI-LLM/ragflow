#!/usr/bin/env python3
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
Script to analyze and compare RAGAS evaluation results
Usage: python analyze_results.py results/evaluation_results.json
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Any, List


def load_results(path: str) -> Dict[str, Any]:
    """Load results from JSON file"""
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def print_summary(results: Dict[str, Any]):
    """Print summary of results"""
    print("\n" + "="*80)
    print("EVALUATION SUMMARY")
    print("="*80)
    
    config = results['config']
    print(f"\nDataset: {config['dataset_path']}")
    print(f"Samples: {config['dataset_size']}")
    print(f"Top K: {config['top_k']}")
    print(f"Timestamp: {results['timestamp']}")
    
    print("\n" + "-"*80)
    print("AGGREGATE METRICS")
    print("-"*80)
    
    aggregate = results['aggregate_metrics']
    
    # Group metrics by type
    recall_metrics = {k: v for k, v in aggregate.items() if 'recall' in k}
    precision_metrics = {k: v for k, v in aggregate.items() if 'precision' in k}
    f1_metrics = {k: v for k, v in aggregate.items() if 'f1' in k}
    ndcg_metrics = {k: v for k, v in aggregate.items() if 'ndcg' in k}
    other_metrics = {k: v for k, v in aggregate.items() 
                     if not any(x in k for x in ['recall', 'precision', 'f1', 'ndcg'])}
    
    def print_metrics(metrics, title):
        if metrics:
            print(f"\n{title}:")
            for name, stats in sorted(metrics.items()):
                print(f"  {name:20s} Mean: {stats['mean']:.4f}  "
                      f"Median: {stats['median']:.4f}  "
                      f"Std: {stats['std']:.4f}")
    
    print_metrics(recall_metrics, "Recall Metrics")
    print_metrics(precision_metrics, "Precision Metrics")
    print_metrics(f1_metrics, "F1 Metrics")
    print_metrics(ndcg_metrics, "NDCG Metrics")
    print_metrics(other_metrics, "Other Metrics")


def find_worst_queries(results: Dict[str, Any], metric: str = 'recall@5', top_n: int = 10):
    """Find queries with worst performance"""
    detailed = results['detailed_results']
    
    # Sort by metric
    sorted_results = sorted(
        detailed,
        key=lambda x: x['metrics'].get(metric, 0)
    )
    
    print("\n" + "-"*80)
    print(f"TOP {top_n} WORST PERFORMING QUERIES ({metric})")
    print("-"*80)
    
    for i, result in enumerate(sorted_results[:top_n], 1):
        score = result['metrics'].get(metric, 0)
        query = result['query'][:100] + "..." if len(result['query']) > 100 else result['query']
        print(f"\n{i}. Score: {score:.4f}")
        print(f"   Query: {query}")
        print(f"   Ground Truth IDs: {result['ground_truth_ids']}")
        print(f"   Retrieved IDs: {result['retrieved_ids'][:5]}")


def find_best_queries(results: Dict[str, Any], metric: str = 'recall@5', top_n: int = 10):
    """Find queries with best performance"""
    detailed = results['detailed_results']
    
    # Sort by metric (descending)
    sorted_results = sorted(
        detailed,
        key=lambda x: x['metrics'].get(metric, 0),
        reverse=True
    )
    
    print("\n" + "-"*80)
    print(f"TOP {top_n} BEST PERFORMING QUERIES ({metric})")
    print("-"*80)
    
    for i, result in enumerate(sorted_results[:top_n], 1):
        score = result['metrics'].get(metric, 0)
        query = result['query'][:100] + "..." if len(result['query']) > 100 else result['query']
        print(f"\n{i}. Score: {score:.4f}")
        print(f"   Query: {query}")
        print(f"   Ground Truth IDs: {result['ground_truth_ids']}")
        print(f"   Retrieved IDs: {result['retrieved_ids'][:5]}")


def compare_results(result_files: List[str]):
    """Compare multiple evaluation results"""
    all_results = []
    for file in result_files:
        results = load_results(file)
        all_results.append({
            'file': Path(file).name,
            'results': results
        })
    
    print("\n" + "="*80)
    print("COMPARISON OF MULTIPLE EVALUATIONS")
    print("="*80)
    
    # Get common metrics
    first_metrics = set(all_results[0]['results']['aggregate_metrics'].keys())
    common_metrics = first_metrics
    for r in all_results[1:]:
        common_metrics &= set(r['results']['aggregate_metrics'].keys())
    
    print(f"\nComparing {len(result_files)} evaluations")
    print(f"Common metrics: {len(common_metrics)}")
    
    # Print comparison table
    print("\n" + "-"*80)
    for metric in sorted(common_metrics):
        print(f"\n{metric}:")
        print(f"  {'File':<40} {'Mean':<10} {'Median':<10} {'Std':<10}")
        print(f"  {'-'*70}")
        for r in all_results:
            stats = r['results']['aggregate_metrics'][metric]
            print(f"  {r['file']:<40} "
                  f"{stats['mean']:<10.4f} "
                  f"{stats['median']:<10.4f} "
                  f"{stats['std']:<10.4f}")


def export_csv(results: Dict[str, Any], output_path: str):
    """Export detailed results to CSV"""
    import csv
    
    detailed = results['detailed_results']
    
    if not detailed:
        print("No detailed results to export")
        return
    
    # Get all metric names
    metric_names = sorted(detailed[0]['metrics'].keys())
    
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        fieldnames = ['query', 'ground_truth_ids', 'retrieved_ids', 'retrieval_time'] + metric_names
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        
        writer.writeheader()
        for result in detailed:
            row = {
                'query': result['query'],
                'ground_truth_ids': str(result['ground_truth_ids']),
                'retrieved_ids': str(result['retrieved_ids']),
                'retrieval_time': result['retrieval_time']
            }
            row.update(result['metrics'])
            writer.writerow(row)
    
    print(f"\nExported detailed results to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Analyze RAGAS evaluation results',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python analyze_results.py results/evaluation_results.json
  python analyze_results.py results/eval1.json --worst-queries 5
  python analyze_results.py results/eval1.json --export-csv output.csv
  python analyze_results.py results/eval1.json results/eval2.json --compare
        """
    )
    
    parser.add_argument(
        'results',
        nargs='+',
        help='Path to evaluation results JSON file(s)'
    )
    
    parser.add_argument(
        '--worst-queries',
        type=int,
        default=0,
        metavar='N',
        help='Show N worst performing queries'
    )
    
    parser.add_argument(
        '--best-queries',
        type=int,
        default=0,
        metavar='N',
        help='Show N best performing queries'
    )
    
    parser.add_argument(
        '--metric',
        default='recall@5',
        help='Metric to use for ranking queries (default: recall@5)'
    )
    
    parser.add_argument(
        '--export-csv',
        type=str,
        metavar='PATH',
        help='Export detailed results to CSV file'
    )
    
    parser.add_argument(
        '--compare',
        action='store_true',
        help='Compare multiple result files'
    )
    
    args = parser.parse_args()
    
    try:
        if args.compare and len(args.results) > 1:
            compare_results(args.results)
        else:
            results = load_results(args.results[0])
            print_summary(results)
            
            if args.worst_queries > 0:
                find_worst_queries(results, args.metric, args.worst_queries)
            
            if args.best_queries > 0:
                find_best_queries(results, args.metric, args.best_queries)
            
            if args.export_csv:
                export_csv(results, args.export_csv)
        
        print("\n" + "="*80 + "\n")
        return 0
    
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())

