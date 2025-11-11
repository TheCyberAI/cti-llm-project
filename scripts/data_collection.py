#!/usr/bin/env python3
"""
Data Collection Script for CTI LLM Project
Creates sample threat intelligence data with realistic IOCs
"""

import os
import sys
import json
from pathlib import Path
import logging

# Add the parent directory to Python path to import config
sys.path.append(str(Path(__file__).parent.parent))

from config.settings import config

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CTIDataCollector:
    """Creates CTI training data with realistic IOCs"""
    
    def __init__(self):
        self.raw_data_dir = Path(config.data.raw_data_dir)
        self.raw_data_dir.mkdir(parents=True, exist_ok=True)
    
    def create_sample_threat_reports(self):
        """Create comprehensive sample threat reports with IOCs"""
        logger.info("Creating sample threat intelligence reports...")
        
        threat_reports = [
            {
                "filename": "APT35_Threat_Report.txt",
                "content": """
                APT35 (Charming Kitten) Cyber Threat Intelligence Report
                
                Overview:
                APT35, also known as Charming Kitten or Newscaster, is an Iranian threat group active since 2014.
                The group primarily targets diplomatic and government organizations in the Middle East and Western countries.
                
                Campaign: Operation Spoofed Scholars
                Timeline: 2023-2024
                Targets: Academic institutions, think tanks, government agencies
                
                Indicators of Compromise:
                
                Network Infrastructure:
                - IP Address: 192.168.1.100
                - IP Address: 10.0.0.50
                - Domain: academic-research[.]com
                - Domain: scholarship-portal[.]net
                - URL: http://academic-research.com/update.exe
                - URL: https://scholarship-portal.net/portal.zip
                
                Malware Hashes:
                - MD5: a1d378111335d450769049446df79983
                - MD5: ccfb3ba332962641c2ed075eb88070a
                - SHA1: bb700e1ef97e1eed56bb275fde2c5faed008c225
                - SHA1: 221f66582841ec2ef79a46d4f90b0e32642887ba
                - SHA256: 35a485972282b7c0e8e3a7a9cbf86ad93856378f696cc8e230be5099cdb89208
                - SHA256: 0300e7ad4c12273a42e4c95d854408b98b0cf5ef5f8c5ce05b24729b64e369
                
                Communication Channels:
                - Email: admin@academic-research.com
                - Email: support@scholarship-portal.net
                
                Tactics, Techniques & Procedures:
                - Spear phishing with academic themes
                - Fake scholarship and research portals
                - Credential harvesting attacks
                """
            },
            {
                "filename": "FIN7_Malware_Analysis.txt", 
                "content": """
                FIN7 Cyber Crime Group Malware Analysis Report
                
                Overview:
                FIN7 is a financially motivated cybercrime group known for targeting restaurant, gambling, and hospitality industries.
                
                Campaign: Carbanak Financial Theft
                Timeline: 2023-2024  
                Targets: POS systems, financial institutions
                
                Indicators of Compromise:
                
                Command & Control Infrastructure:
                - IP Address: 203.0.113.45
                - IP Address: 198.51.100.23
                - Domain: payment-processor[.]com
                - Domain: pos-update[.]net
                - URL: https://payment-processor.com/security_update.exe
                - URL: http://pos-update.net/installer.msi
                
                Malware Samples:
                - MD5: 5d41402abc4b2a76b9719d911017c592
                - MD5: 7d793037a0760186574b0280ce8a3b4d
                - SHA1: aaf4c61ddcc5e8a2dabede0f3b482cd9aea9434d
                - SHA1: 5baa61e4c9b93f3f0682250b6cf8331b7ee68fd8
                - SHA256: cd2ba296828660ecd07a36e8931b851dda0802069ed926b3161745aae9aaddaa
                - SHA256: ac8e59e8abeacf0885b45183372c0e3e8e249c88d21127b16ebe00f00c1409e6
                
                Tools Used:
                - Carbanak backdoor
                - Griffon payload
                - Custom PowerShell scripts
                """
            },
            {
                "filename": "Lazarus_Group_Campaign.txt",
                "content": """
                Lazarus Group Cyber Campaign Analysis
                
                Overview:
                Lazarus Group is a North Korean state-sponsored threat group responsible for numerous high-profile attacks.
                
                Campaign: Operation Ghost Secret
                Timeline: 2023-2024
                Targets: Financial institutions, cryptocurrency exchanges
                
                Indicators of Compromise:
                
                Infrastructure:
                - IP Address: 172.16.254.1
                - IP Address: 192.0.2.146
                - Domain: security-update[.]org
                - Domain: blockchain-sync[.]com
                - URL: http://security-update.org/patches.exe
                - URL: https://blockchain-sync.com/wallet_update.dll
                
                Malware Hashes:
                - MD5: 3a62b26311583a23767c35d56b95175d
                - MD5: 10f5561b7515bc0d66916be04b63dae
                - SHA1: aa791a0a98a30e10119b8cc1399ab1306275fc1f
                - SHA1: 36fa5a020aca2cfab25661cf2ae7637de1aaf8d4
                - SHA256: 767bd025c8e7d36f64dbd636ec0f29e873d1e3ca415d5a449053a68916fe894
                - SHA256: safc59cd2b391988733eba427c8cf6e48bd2e9dc3d48a4db550655efe0dca798
                
                Attack Vectors:
                - Watering hole attacks
                - Supply chain compromise
                - Social engineering
                """
            },
            {
                "filename": "CozyBear_APT29_Report.txt",
                "content": """
                APT29 (Cozy Bear) Intelligence Report
                
                Overview:
                APT29, also known as Cozy Bear or The Dukes, is a Russian state-sponsored threat group.
                
                Campaign: SolarWinds Supply Chain Attack
                Timeline: 2023-2024
                Targets: Government agencies, IT service providers
                
                Indicators of Compromise:
                
                Infrastructure:
                - IP Address: 198.18.0.1
                - IP Address: 192.168.100.200
                - Domain: software-update[.]net
                - Domain: patch-management[.]com
                - URL: https://software-update.net/oracle_java.exe
                - URL: http://patch-management.com/adobe_flash.dll
                
                Tool Hashes:
                - MD5: 5c91c1d833173f6ae599ef1d4133f235
                - MD5: 45c3592373ba9e5f8f23c6b30fb4d2e4
                - SHA1: 2cb21b71e23cf108067bd02070343883618a5837
                - SHA1: 1156c3f36800acac714bf78f870d7f066ad25edf
                - SHA256: ac8e59e8abeacf0885b45183372c0e3e8e249c88d21127b16ebe00f00c1409e6
                - SHA256: cd2ba296828660ecd07a36e8931b851dda0802069ed926b3161745aae9aaddaa
                
                Techniques:
                - API hijacking
                - Golden SAML attacks
                - Token theft
                """
            }
        ]
        
        created_files = 0
        for report in threat_reports:
            file_path = self.raw_data_dir / report["filename"]
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(report["content"])
            created_files += 1
            logger.info(f"Created: {report['filename']}")
        
        logger.info(f"Created {created_files} sample threat reports")
        return created_files

def main():
    """Main data collection function"""
    collector = CTIDataCollector()
    collector.create_sample_threat_reports()
    logger.info("Data collection completed successfully!")

if __name__ == "__main__":
    main()
