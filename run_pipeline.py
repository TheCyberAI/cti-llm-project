#!/usr/bin/env python3
"""
Main Pipeline Runner for CTI LLM Project
Runs the complete pipeline or individual steps
"""

import argparse
import logging
import sys
import importlib.util
from pathlib import Path

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

def import_module_from_file(module_name, file_path):
    """Import a module from a file path"""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def run_data_collection():
    """Run data collection step"""
    try:
        data_collection_path = Path(__file__).parent / "scripts" / "data_collection.py"
        data_collection = import_module_from_file("data_collection", data_collection_path)
        data_collection.main()
        return True
    except Exception as e:
        logger.error(f"Data collection failed: {e}")
        return False

def run_data_processing():
    """Run data processing step"""
    try:
        data_processing_path = Path(__file__).parent / "scripts" / "data_processing.py"
        data_processing = import_module_from_file("data_processing", data_processing_path)
        data_processing.main()
        return True
    except Exception as e:
        logger.error(f"Data processing failed: {e}")
        return False

def run_training():
    """Run training step"""
    try:
        training_path = Path(__file__).parent / "scripts" / "training.py"
        training = import_module_from_file("training", training_path)
        training.main()
        return True
    except Exception as e:
        logger.error(f"Training failed: {e}")
        return False

def run_inference():
    """Run inference step"""
    try:
        inference_path = Path(__file__).parent / "scripts" / "inference.py"
        inference = import_module_from_file("inference", inference_path)
        inference_engine = inference.CTIInferenceEngine()
        inference_engine.interactive_mode()
        return True
    except Exception as e:
        logger.error(f"Inference failed: {e}")
        return False

def run_complete_pipeline():
    """Run the complete pipeline end-to-end"""
    logger.info("Starting Complete CTI LLM Pipeline...")
    
    steps = [
        ("Data Collection", run_data_collection),
        ("Data Processing", run_data_processing),
        ("Model Training", run_training),
    ]
    
    for step_name, step_function in steps:
        logger.info(f"Step: {step_name}...")
        if not step_function():
            logger.error(f"Pipeline failed at {step_name}")
            return False
    
    # Test inference
    logger.info("Step: Testing Inference...")
    try:
        inference_path = Path(__file__).parent / "scripts" / "inference.py"
        inference = import_module_from_file("inference", inference_path)
        engine = inference.CTIInferenceEngine()
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
        return True
        
    except Exception as e:
        logger.error(f"Inference test failed: {e}")
        return False

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
            success = run_complete_pipeline()
        elif args.step == "collect":
            success = run_data_collection()
        elif args.step == "process":
            success = run_data_processing()
        elif args.step == "train":
            success = run_training()
        elif args.step == "inference":
            success = run_inference()
        else:
            print(f"Unknown step: {args.step}")
            return 1
            
        if not success:
            print(f"\n❌ Step '{args.step}' failed. Check the logs for details.")
            return 1
            
    except Exception as e:
        logger.error(f"Error in pipeline: {e}")
        print(f"Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
