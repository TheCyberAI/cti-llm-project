import os
from dataclasses import dataclass
from typing import Optional
import torch

@dataclass
class DataConfig:
    """Configuration for data processing - macOS optimized"""
    raw_data_dir: str = "data/raw"
    processed_data_dir: str = "data/processed"
    max_file_size_mb: int = 50
    supported_formats: list = None
    
    def __post_init__(self):
        if self.supported_formats is None:
            self.supported_formats = ['.txt', '.pdf', '.json']

@dataclass
class ModelConfig:
    """Configuration for model training - macOS compatible"""
    # Use smaller, compatible models
    model_name: str = "distilgpt2"  # Most compatible option
    token: Optional[str] = None
    
    # Training parameters optimized for macOS
    batch_size: int = 1
    gradient_accumulation_steps: int = 8
    learning_rate: float = 5e-5
    num_epochs: int = 3
    max_length: int = 512
    
    # LoRA configuration (simplified)
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.1

@dataclass
class TrainingConfig:
    """Training configuration optimized for macOS"""
    output_dir: str = "models/cti-llm"
    logging_dir: str = "logs"
    save_steps: int = 100
    eval_steps: int = 100
    logging_steps: int = 10

class Config:
    """Main configuration class for macOS"""
    def __init__(self):
        self.data = DataConfig()
        self.model = ModelConfig()
        self.training = TrainingConfig()
    
    def detect_apple_silicon(self):
        """Detect if running on Apple Silicon"""
        try:
            import platform
            return platform.processor() == 'arm'
        except:
            return False
    
    def get_device(self):
        """Get the appropriate device for training/inference"""
        if torch.backends.mps.is_available() and self.detect_apple_silicon():
            return "mps"
        elif torch.cuda.is_available():
            return "cuda"
        else:
            return "cpu"

# Global configuration instance
config = Config()
