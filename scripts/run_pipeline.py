#!/usr/bin/env python3
"""
Main Pipeline Runner for CTI LLM Project
Runs the complete pipeline or individual steps
"""

import argparse
import logging
import sys
from pathlib import Path

# Add scripts to path
sys.path.append(str(Path(__file__).parent / "scripts"))

from data_collection import main as collect_data
from data_processing import main as process_data
from training import main as train_model
from inference import CTIInferenceEngine

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('cti_llm_pipeline.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def run_complete_pipeline():
    """Run the complete pipeline end-to-end"""
    logger.info("Starting Complete CTI LLM Pipeline...")
    
    try:
        # Step 1: Data Collection
        logger.info("Step 1: Data Collection...")
        collect_data()
        
        # Step 2: Data Processing
        logger.info("Step 2: Data Processing...")
        process_data()
        
        # Step 3: Model Training
        logger.info("Step 3: Model Training...")
        train_model()
        
        # Step 4: Test Inference
        logger.info("Step 4: Testing Inference...")
        engine = CTIInferenceEngine()
        engine.load_model()
        
        test_prompts = [
            "List Indicators of Compromise in APT35",
            "Extract IOCs from FIN7 threat report",
            "What are the indicators for Lazarus Group?"
        ]
        
        print("\n" + "="*60)
        print("TESTING INFERENCE")
        print("="*60)
        
        for prompt in test_prompts:
            print(f"\nPrompt: {prompt}")
            result = engine.generate_iocs(prompt)
            print(f"Response:\n{result}")
            print("-" * 40)
        
        logger.info("Pipeline completed successfully!")
        print("\n🎉 Pipeline completed successfully!")
        print("You can now use the model for inference.")
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        print(f"\n❌ Pipeline failed: {e}")
        raise

def main():
    """Main pipeline function"""
    parser = argparse.ArgumentParser(description="CTI LLM Pipeline Runner")
    parser.add_argument("--step", 
                       choices=["collect", "process", "train", "inference", "all"],
                       default="all",
                       help="Which step to run (default: all)")
    
    args = parser.parse_args()
    
    print("CTI LLM Fine-tuning Pipeline")
    print("=" * 50)
    
    try:
        if args.step == "all":
            run_complete_pipeline()
        elif args.step == "collect":
            collect_data()
        elif args.step == "process":
            process_data()
        elif args.step == "train":
            train_model()
        elif args.step == "inference":
            engine = CTIInferenceEngine()
            engine.interactive_mode()
            
    except Exception as e:
        logger.error(f"Error in pipeline: {e}")
        print(f"Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
