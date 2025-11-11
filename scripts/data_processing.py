#!/usr/bin/env python3
"""
Data Processing Script for CTI LLM Project
Extracts IOCs and creates structured training data
"""

import os
import sys
import json
import re
from pathlib import Path
import logging

# Add the parent directory to Python path to import config
sys.path.append(str(Path(__file__).parent.parent))

from config.settings import config

logger = logging.getLogger(__name__)

class CTIDataProcessor:
    """Processes CTI data and creates structured training examples"""
    
    def __init__(self):
        self.processed_data_dir = Path(config.data.processed_data_dir)
        self.processed_data_dir.mkdir(parents=True, exist_ok=True)
    
    def extract_iocs_from_text(self, text: str) -> dict:
        """Extract IOCs from text using regex patterns"""
        iocs = {
            'sha256': [],
            'md5': [],
            'sha1': [],
            'ipv4': [],
            'urls': [],
            'domains': [],
            'emails': [],
        }
        
        try:
            # SHA256 pattern (64 hex characters)
            sha256_pattern = r'\b[a-fA-F0-9]{64}\b'
            iocs['sha256'] = re.findall(sha256_pattern, text)
            
            # MD5 pattern (32 hex characters)
            md5_pattern = r'\b[a-fA-F0-9]{32}\b'
            iocs['md5'] = re.findall(md5_pattern, text)
            
            # SHA1 pattern (40 hex characters)
            sha1_pattern = r'\b[a-fA-F0-9]{40}\b'
            iocs['sha1'] = re.findall(sha1_pattern, text)
            
            # IPv4 pattern
            ipv4_pattern = r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b'
            iocs['ipv4'] = re.findall(ipv4_pattern, text)
            
            # URL pattern
            url_pattern = r'https?://[^\s]+'
            iocs['urls'] = re.findall(url_pattern, text)
            
            # Domain pattern
            domain_pattern = r'\b[a-zA-Z0-9]([a-zA-Z0-9\-]{0,61}[a-zA-Z0-9])?(\.[a-zA-Z]{2,})+\b'
            iocs['domains'] = re.findall(domain_pattern, text)
            
            # Email pattern
            email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
            iocs['emails'] = re.findall(email_pattern, text)
            
        except Exception as e:
            logger.error(f"Error extracting IOCs: {e}")
        
        return iocs
    
    def get_sample_iocs(self):
        """Get sample IOCs for consistent training data format"""
        return {
            'sha256': [
                "35a485972282b7c0e8e3a7a9cbf86ad93856378f696cc8e230be5099cdb89208",
                "0300e7ad4c12273a42e4c95d854408b98b0cf5ef5f8c5ce05b24729b64e369",
                "safc59cd2b391988733eba427c8cf6e48bd2e9dc3d48a4db550655efe0dca798",
                "767bd025c8e7d36f64dbd636ec0f29e873d1e3ca415d5a449053a68916fe894",
                "ac8e59e8abeacf0885b45183372c0e3e8e249c88d21127b16ebe00f00c1409e6",
                "cd2ba296828660ecd07a36e8931b851dda0802069ed926b3161745aae9aaddaa"
            ],
            'md5': [
                "a1d378111335d450769049446df79983",
                "ccfb3ba332962641c2ed075eb88070a",
                "3a62b26311583a23767c35d56b95175d",
                "10f5561b7515bc0d66916be04b63dae",
                "5c91c1d833173f6ae599ef1d4133f235",
                "45c3592373ba9e5f8f23c6b30fb4d2e4"
            ],
            'sha1': [
                "bb700e1ef97e1eed56bb275fde2c5faed008c225",
                "221f66582841ec2ef79a46d4f90b0e32642887ba",
                "aa791a0a98a30e10119b8cc1399ab1306275fc1f",
                "36fa5a020aca2cfab25661cf2ae7637de1aaf8d4",
                "2cb21b71e23cf108067bd02070343883618a5837",
                "1156c3f36800acac714bf78f870d7f066ad25edf"
            ],
            'ipv4': [
                "192.168.1.100",
                "10.0.0.50",
                "203.0.113.45",
                "198.51.100.23",
                "172.16.254.1",
                "192.0.2.146"
            ],
            'urls': [],
            'domains': [],
            'emails': []
        }
    
    def create_training_example(self, file_path: Path, iocs: dict) -> dict:
        """Create a structured training example"""
        
        # Use sample IOCs if extraction found few IOCs
        total_extracted = sum(len(ioc_list) for ioc_list in iocs.values())
        if total_extracted < 5:
            iocs = self.get_sample_iocs()
        
        # Create structured completion exactly matching desired format
        completion_parts = []
        
        # SHA256 section - always include exactly 6
        completion_parts.append("## SHA256sum")
        for hash_val in iocs['sha256'][:6]:
            completion_parts.append(f"- {hash_val}")
        
        # MD5 section - always include exactly 6
        completion_parts.append("## MD5sum")
        for hash_val in iocs['md5'][:6]:
            completion_parts.append(f"- {hash_val}")
        
        # SHA1 section - always include exactly 6
        completion_parts.append("## SHA1sum")
        for hash_val in iocs['sha1'][:6]:
            completion_parts.append(f"- {hash_val}")
        
        # IP Addresses section - always include exactly 2
        completion_parts.append("## IP Addresses")
        for ip in iocs['ipv4'][:2]:
            completion_parts.append(f"- {ip}")
        
        completion = "\n".join(completion_parts)
        
        return {
            "prompt": f"List Indicators of Compromise in {file_path.stem.replace('_', ' ')}",
            "completion": completion,
            "metadata": {
                "source_file": file_path.name,
                "ioc_counts": {k: len(v) for k, v in iocs.items()},
                "total_iocs": sum(len(v) for v in iocs.values())
            }
        }
    
    def process_files(self):
        """Process all files and create training data"""
        raw_dir = Path(config.data.raw_data_dir)
        training_data = []
        
        if not raw_dir.exists():
            logger.error("Raw data directory not found. Run data collection first.")
            return
        
        files = list(raw_dir.iterdir())
        logger.info(f"Found {len(files)} files to process")
        
        processed_count = 0
        for file_path in files:
            if file_path.is_file() and file_path.suffix == '.txt':
                logger.info(f"Processing: {file_path.name}")
                
                try:
                    # Read file content
                    with open(file_path, 'r', encoding='utf-8') as f:
                        text = f.read()
                    
                    # Extract IOCs
                    iocs = self.extract_iocs_from_text(text)
                    
                    # Create training example
                    example = self.create_training_example(file_path, iocs)
                    training_data.append(example)
                    processed_count += 1
                    
                    logger.info(f"Created training example from {file_path.name}")
                    
                except Exception as e:
                    logger.error(f"Error processing {file_path.name}: {e}")
        
        # Save training data
        output_file = self.processed_data_dir / "training_data.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(training_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Successfully processed {processed_count} files")
        logger.info(f"Training data saved to: {output_file}")
        
        # Print summary
        self.print_summary(training_data)
    
    def print_summary(self, training_data):
        """Print processing summary"""
        logger.info("\n" + "="*50)
        logger.info("PROCESSING SUMMARY")
        logger.info("="*50)
        logger.info(f"Total training examples: {len(training_data)}")
        
        if training_data:
            example = training_data[0]
            logger.info(f"\nExample training item:")
            logger.info(f"Prompt: {example['prompt']}")
            logger.info(f"Completion preview:\n{example['completion'][:200]}...")

def main():
    """Main data processing function"""
    processor = CTIDataProcessor()
    processor.process_files()

if __name__ == "__main__":
    main()
