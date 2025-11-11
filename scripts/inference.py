#!/usr/bin/env python3
"""
Inference Script for CTI LLM Project
Uses the fine-tuned model to generate structured IOCs
"""

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    pipeline
)
from peft import PeftModel
import logging
from pathlib import Path
import sys

# Add the parent directory to Python path to import config
sys.path.append(str(Path(__file__).parent.parent))

from config.settings import config

logger = logging.getLogger(__name__)

class CTIInferenceEngine:
    """Inference engine for the fine-tuned CTI model"""
    
    def __init__(self, model_path: str = None):
        self.model_path = model_path or config.training.output_dir
        self.model = None
        self.tokenizer = None
        self.pipeline = None
        
    def load_model(self):
        """Load the fine-tuned model"""
        logger.info("Loading fine-tuned model...")
        
        try:
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load base model
            base_model = AutoModelForCausalLM.from_pretrained(
                config.model.model_name,
                torch_dtype=torch.float32,
            )
            
            # Apply LoRA adapters
            self.model = PeftModel.from_pretrained(base_model, self.model_path)
            
            # Move to appropriate device
            device = config.get_device()
            if str(device) == "mps":
                self.model = self.model.to(device)
            
            # Create pipeline
            self.pipeline = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                torch_dtype=torch.float32,
            )
            
            logger.info("Model loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def generate_iocs(self, prompt: str, max_length: int = 300) -> str:
        """Generate structured IOCs from prompt"""
        if self.pipeline is None:
            self.load_model()
        
        # Format prompt
        formatted_prompt = f"### Instruction:\n{prompt}\n\n### Response:\n"
        
        try:
            # Generate response
            response = self.pipeline(
                formatted_prompt,
                max_new_tokens=max_length,
                temperature=0.3,
                do_sample=True,
                top_p=0.9,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                repetition_penalty=1.1
            )
            
            generated_text = response[0]['generated_text']
            
            # Extract only the response part
            if "### Response:" in generated_text:
                response_text = generated_text.split("### Response:")[1].strip()
            else:
                response_text = generated_text
            
            return self.clean_response(response_text)
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return f"Error: {str(e)}"
    
    def clean_response(self, text: str) -> str:
        """Clean and format the model response"""
        lines = text.split('\n')
        cleaned_lines = []
        
        for line in lines:
            line = line.strip()
            
            # Keep section headers
            if line.startswith('## '):
                cleaned_lines.append(line)
            # Keep bullet points with IOCs
            elif line.startswith('- '):
                # Extract only the IOC (first part after -)
                ioc_part = line[2:].split()[0] if line[2:].split() else ""
                if ioc_part:
                    cleaned_lines.append(f"- {ioc_part}")
            # Skip empty lines and other text
            elif not line:
                continue
        
        return '\n'.join(cleaned_lines)
    
    def interactive_mode(self):
        """Run in interactive mode"""
        self.load_model()
        
        print("\n" + "="*60)
        print("CTI LLM Inference Engine")
        print("="*60)
        print("Enter prompts to generate structured IOCs")
        print("Type 'quit' to exit\n")
        
        while True:
            try:
                prompt = input("Enter prompt: ").strip()
                
                if prompt.lower() in ['quit', 'exit', 'q']:
                    break
                
                if not prompt:
                    continue
                
                print("\nGenerating IOCs...")
                result = self.generate_iocs(prompt)
                print(f"\nResponse:\n{result}\n")
                print("-" * 50)
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"Error: {e}")

def main():
    """Main inference function"""
    engine = CTIInferenceEngine()
    
    # Test with sample prompts
    test_prompts = [
        "List Indicators of Compromise in APT35",
        "Extract IOCs from FIN7 threat report",
        "What are the indicators for Lazarus Group?",
        "Show me IOCs from recent cyber attack"
    ]
    
    print("Testing CTI LLM Inference...")
    print("=" * 50)
    
    engine.load_model()
    
    for prompt in test_prompts:
        print(f"\nPrompt: {prompt}")
        result = engine.generate_iocs(prompt)
        print(f"Response:\n{result}")
        print("-" * 50)
    
    # Start interactive mode
    engine.interactive_mode()

if __name__ == "__main__":
    main()
