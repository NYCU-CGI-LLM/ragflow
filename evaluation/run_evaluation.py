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
Standalone script to run RAGAS evaluation
Usage: python run_evaluation.py --config path/to/config.json
"""

import argparse
import sys
from pathlib import Path

# Add parent directory (project root) to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from evaluation.ragas_evaluation import run_evaluation


def main():
    parser = argparse.ArgumentParser(
        description='Run RAGAS evaluation for RAGFlow retrieval system',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_evaluation.py --config config/minimal_config.yaml
  python run_evaluation.py --config config/example_config.yaml
        """
    )
    
    parser.add_argument(
        '--config',
        required=True,
        type=str,
        help='Path to evaluation configuration YAML file'
    )
    
    args = parser.parse_args()
    
    try:
        print("="*80)
        print("RAGFlow RAGAS Evaluation")
        print("="*80)
        print(f"\nConfig file: {args.config}\n")
        
        results = run_evaluation(args.config)
        
        print("\n" + "="*80)
        print("Evaluation completed successfully!")
        print("="*80)
        
        return 0
    
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Error during evaluation: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())

