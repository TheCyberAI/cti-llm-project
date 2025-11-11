#!/usr/bin/env python3
"""
Model Training Script for CTI LLM Project
Fine-tunes a language model for IOC extraction
"""

import os
import sys
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import Dataset
import json
import logging
from pathlib import Path

# Add the parent directory to Python path to import config
sys.path.append(str(Path(__file__).parent.parent))

from config.settings import config

logger = logging.getLogger(__name__)

class CTIModelTrainer:
    """Fine-tunes LLM for CTI IOC extraction"""
    
    def __init__(self):
        self.output_dir = Path(config.training.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def setup_device(self):
        """Setup device for training"""
        device = config.get_device()
        logger.info(f"Using device: {device}")
        return device
    
    def load_training_data(self) -> Dataset:
        """Load and prepare training data"""
        data_file = Path(config.data.processed_data_dir) / "training_data.json"
        
        if not data_file.exists():
            raise FileNotFoundError(f"Training data not found: {data_file}")
        
        with open(data_file, 'r') as f:
            training_data = json.load(f)
        
        if len(training_data) == 0:
            raise ValueError("No training data available. Run data processing first.")
        
        # Format for instruction tuning
        formatted_data = []
        for example in training_data:
            formatted_text = f"### Instruction:\n{example['prompt']}\n\n### Response:\n{example['completion']}"
            formatted_data.append({"text": formatted_text})
        
        logger.info(f"Loaded {len(formatted_data)} training examples")
        return Dataset.from_list(formatted_data)
    
    def setup_model_and_tokenizer(self):
        """Initialize model and tokenizer"""
        logger.info(f"Loading model: {config.model.model_name}")
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(config.model.model_name)
        
        # Set padding token
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Load model
        device = self.setup_device()
        
        model = AutoModelForCausalLM.from_pretrained(
            config.model.model_name,
            torch_dtype=torch.float32,
        )
        
        if str(device) == "mps":
            model = model.to(device)
        
        return model, tokenizer
    
    def setup_lora_config(self, model):
        """Setup LoRA configuration with compatible target modules"""
        # For GPT-2 based models (distilgpt2, gpt2)
        target_modules = ["c_attn", "c_proj"]
        
        logger.info(f"Using LoRA target modules: {target_modules}")
        
        return LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=config.model.lora_r,
            lora_alpha=config.model.lora_alpha,
            lora_dropout=config.model.lora_dropout,
            target_modules=target_modules,
            bias="none",
        )
    
    def tokenize_function(self, examples, tokenizer):
        """Tokenize training examples"""
        tokenized = tokenizer(
            examples["text"],
            truncation=True,
            padding=True,
            max_length=config.model.max_length,
            return_tensors="pt"
        )
        
        # For causal LM, labels are the same as input_ids
        tokenized["labels"] = tokenized["input_ids"].clone()
        
        return tokenized
    
    def train(self):
        """Execute the training pipeline"""
        logger.info("Starting model training...")
        
        try:
            # Load data
            dataset = self.load_training_data()
            
            # Setup model and tokenizer
            model, tokenizer = self.setup_model_and_tokenizer()
            
            # Apply LoRA
            lora_config = self.setup_lora_config(model)
            model = get_peft_model(model, lora_config)
            model.print_trainable_parameters()
            
            # Tokenize dataset
            tokenized_dataset = dataset.map(
                lambda x: self.tokenize_function(x, tokenizer),
                batched=True,
                remove_columns=dataset.column_names
            )
            
            # Training arguments
            training_args = TrainingArguments(
                output_dir=str(self.output_dir),
                overwrite_output_dir=True,
                per_device_train_batch_size=config.model.batch_size,
                gradient_accumulation_steps=config.model.gradient_accumulation_steps,
                learning_rate=config.model.learning_rate,
                num_train_epochs=config.model.num_epochs,
                logging_dir=str(Path(config.training.logging_dir)),
                logging_steps=config.training.logging_steps,
                save_steps=config.training.save_steps,
                evaluation_strategy="no",
                save_strategy="steps",
                fp16=False,
                warmup_ratio=0.1,
                lr_scheduler_type="cosine",
                report_to=None,
                remove_unused_columns=False,
            )
            
            # Data collator
            data_collator = DataCollatorForLanguageModeling(
                tokenizer=tokenizer,
                mlm=False,
            )
            
            # Initialize trainer
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=tokenized_dataset,
                data_collator=data_collator,
                tokenizer=tokenizer,
            )
            
            # Start training
            logger.info("Training started...")
            trainer.train()
            
            # Save final model
            trainer.save_model()
            tokenizer.save_pretrained(str(self.output_dir))
            
            logger.info(f"Training completed! Model saved to {self.output_dir}")
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise

def main():
    """Main training function"""
    trainer = CTIModelTrainer()
    trainer.train()

if __name__ == "__main__":
    main()
