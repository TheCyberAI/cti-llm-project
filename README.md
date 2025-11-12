# Create virtual environment
python -m venv cti_llm_env

# Activate on Linux/Mac
source cti_llm_env/bin/activate

# Activate on Windows
cti_llm_env\Scripts\activate


#Install Dependencies
pip install -r requirements.txt

#Install System Dependencies (Ubuntu/Debian)
# For PDF processing and system utilities
sudo apt update
sudo apt install -y libmagic1 poppler-utils tesseract-ocr

# Make the script executable
chmod +x run_pipeline.py

# Run complete pipeline
python run_pipeline.py --step all

# Or run individual steps
python run_pipeline.py --step collect
python run_pipeline.py --step process
python run_pipeline.py --step train
python run_pipeline.py --step inference
