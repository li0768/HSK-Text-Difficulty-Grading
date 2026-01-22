# Model Training Guide

## 📋 Prerequisites
- Python 3.8 or higher
- Required packages (install via pip):

```bash
pip install torch transformers numpy pandas scikit-learn
```

## 🚀 Quick Start

### Step 1: Prepare Dataset
Download the `Dataset.zip` file and extract it to your local directory. We recommend placing it on your desktop for easier access.

### Step 2: Place Training Script
Locate the `HSK Level model.py` file in the `Training Model Files` folder and copy it to your desktop (in the same directory as the extracted Dataset folder).

### Step 3: Run Training Script
Open Command Prompt (cmd) and navigate to your desktop:

```bash
cd Desktop
python "HSK Level model.py"
```

### Step 4: Expected Output
After successful execution, the following files will be generated on your desktop:
- `best_optimized_model.pth` (model weights)
- `best_optimized_model_tokenizer.pkl` (tokenizer)

## 📁 Project Structure (Recommended)
```
Desktop/
├── Dataset/
│   ├── train.csv
│   ├── test.csv
│   └── ...
├── HSK Level model.py
├── best_optimized_model.pth (generated)
└── best_optimized_model_tokenizer.pkl (generated)
```

## ⚠️ Notes
- Ensure you have sufficient disk space for model files
- Training time may vary depending on your hardware
- For GPU acceleration, ensure CUDA-compatible PyTorch is installed

## ❓ Troubleshooting
If you encounter issues:
1. Verify all dependencies are installed correctly
2. Check that the Dataset folder contains the required CSV files
3. Ensure you have write permissions on the desktop directory

---

This format follows standard GitHub README conventions with clear section headers, code blocks, and structured information.# HSK-Text-Difficulty-Grading
