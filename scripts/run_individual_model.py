#!/usr/bin/env python3
"""
Run Individual Model Script

This script allows you to run specific models individually for easier testing and debugging.
Usage: python run_individual_model.py <model_name>

Available models:
- svar
- cge
- olg
- dsge
- rbc
- soe
- financial
- abm
- behavioral
- game_theory
- hank
- iam
- neg
- qmm
- search_matching
"""

import sys
import os
import logging
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import ModelRunner class from run_all_models.py
from run_all_models import ModelRunner

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Model mapping
MODELS = {
    'svar': 'run_svar_model',
    'cge': 'run_cge_model',
    'olg': 'run_olg_model',
    'dsge': 'run_dsge_model',
    'rbc': 'run_rbc_model',
    'soe': 'run_soe_model',
    'financial': 'run_financial_model',
    'abm': 'run_abm_model',
    'behavioral': 'run_behavioral_model',
    'game_theory': 'run_game_theory_model',
    'hank': 'run_hank_model',
    'iam': 'run_iam_model',
    'neg': 'run_neg_model',
    'qmm': 'run_qmm_model',
    'search_matching': 'run_search_matching_model'
}

def main():
    if len(sys.argv) != 2:
        print("Usage: python run_individual_model.py <model_name>")
        print("\nAvailable models:")
        for model_name in MODELS.keys():
            print(f"  - {model_name}")
        sys.exit(1)
    
    model_name = sys.argv[1].lower()
    
    if model_name not in MODELS:
        print(f"Error: Unknown model '{model_name}'")
        print("\nAvailable models:")
        for available_model in MODELS.keys():
            print(f"  - {available_model}")
        sys.exit(1)
    
    print(f"\n{'='*50}")
    print(f"🚀 RUNNING {model_name.upper()} MODEL")
    print(f"{'='*50}")
    
    try:
        # Initialize the model runner
        runner = ModelRunner()
        
        # Get the method name and call it
        method_name = MODELS[model_name]
        method = getattr(runner, method_name)
        
        # Run the selected model
        method()
        print(f"\n✅ {model_name.upper()} model completed successfully!")
    except Exception as e:
        print(f"\n❌ {model_name.upper()} model failed with error: {str(e)}")
        logging.error(f"Model {model_name} failed", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()