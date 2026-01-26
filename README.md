# HSK Level Classification Model

## 📋 Prerequisites
- **Python 3.8+** (Recommended: Python 3.8-3.11)
- **Required Packages**:
  ```bash
  pip install torch torchvision torchaudio transformers
  pip install numpy pandas scikit-learn tqdm
  pip install matplotlib seaborn jupyter
  ```

## 🚀 Quick Start

### Step 1: Download Files
Download the following files to your **Desktop**:
1. **Supplementary File (Dataset).zip** - Contains the Chinese textbook dataset
2. **Supplementary File (Training Model Files).zip** - Contains training scripts and model files

### Step 2: Extract Files
Extract both zip files directly on your Desktop. The extracted structure should look like:
```
Desktop/
├── Supplementary File (Dataset)/
│   └── 李思卫五套教材（分级后教材_已过滤）/
│       └── 所属难度范围（初级（1-3级），中级（4-6级），高级（7-9级））/
│           ├── 初级（1-3级）/
│           │   ├── 1级/
│           │   ├── 2级/
│           │   └── 3级/
│           │       └── 新标准中文 初级中文3/
│           │           ├── lesson1.txt
│           │           └── ...
│           ├── 中级（4-6级）/
│           └── 高级（7-9级）/
└── Supplementary File (Training Model Files)/
    ├── HSK Level model.py
    ├── *.pth
    └── *.ptl
```

### Step 3: Prepare Training Script
Move `HSK Level model.py` from `Supplementary File (Training Model Files)` to your **Desktop**.

### Step 4: Run Training
1. **Open Command Prompt (cmd)** as Administrator
2. **Navigate to Desktop**:
   ```bash
   cd %USERPROFILE%\Desktop
   ```
3. **Execute Training Script**:
   ```bash
   python "HSK Level model.py"
   ```

## 📁 Project Structure
```
Desktop/
├── Supplementary File (Dataset)/
│   └── 李思卫五套教材（分级后教材_已过滤）/
│       └── 所属难度范围（初级（1-3级），中级（4-6级），高级（7-9级））/
│           ├── 初级（1-3级）/      # Beginner level (HSK 1-3)
│           │   ├── 1级/            # HSK Level 1
│           │   ├── 2级/            # HSK Level 2
│           │   └── 3级/            # HSK Level 3
│           ├── 中级（4-6级）/      # Intermediate level (HSK 4-6)
│           └── 高级（7-9级）/      # Advanced level (HSK 7-9)
├── HSK Level model.py              # Main training script
├── Supplementary File (Training Model Files)/
│   ├── HSK Level model.py
│   ├── *.pth                       # Pre-trained model weights
│   └── *.ptl                       # Model configuration files
├── best_optimized_model.pth        # Generated after training
└── best_optimized_model_tokenizer.pkl  # Generated after training
```

## ⚙️ Configuration (If Needed)
If the default paths don't work, modify these variables in `HSK Level model.py`:

```python
# Update these paths according to your system
dataset_path = "C:/Users/YourUsername/Desktop/Supplementary File (Dataset)/李思卫五套教材（分级后教材_已过滤）"
output_path = "C:/Users/YourUsername/Desktop/"
```

## 📊 Expected Training Output
The script will display:
- **Epoch progress** with loss and accuracy
- **Validation results** after each epoch
- **Training completion** message

**Generated Files** (on Desktop after training):
- `best_optimized_model.pth` - Best model weights
- `best_optimized_model_tokenizer.pkl` - Tokenizer for inference
- Training logs and metrics

## ⚠️ Important Notes
1. **Chinese Path Names**: Keep the original Chinese folder names or modify the script accordingly
2. **File Encoding**: All `.txt` files should be UTF-8 encoded
3. **Memory Requirements**: 
   - Minimum 8GB RAM recommended
   - GPU with 4GB+ VRAM for faster training
4. **Training Time**: Approximately 30-60 minutes depending on hardware

## 🔍 Troubleshooting Guide

| Problem | Solution |
|---------|----------|
| **ModuleNotFoundError** | Install missing packages: `pip install [package-name]` |
| **FileNotFoundError** | Check if the dataset path in script matches your actual path |
| **Memory Error** | Reduce batch size in the training script |
| **Encoding Error** | Ensure all text files are saved as UTF-8 |
| **Permission Denied** | Run Command Prompt as Administrator |
| **Slow Training** | Ensure you have GPU support enabled (CUDA) |

## 💡 Pro Tips
1. **Backup First**: Copy the original training script before making changes
2. **Monitor Progress**: The script shows real-time training progress with loss values
3. **Check Output**: Verify generated files exist after training completes
4. **Path Validation**: Use this command to check your current directory:
   ```bash
   dir
   ```

## 📈 Performance Metrics
After training completes, you should see:
- **Training Accuracy**: > 85%
- **Validation Accuracy**: > 80%
- **Model Size**: Approximately 100-200MB

## 🆘 Need Help?
1. Check the **Troubleshooting Guide** above
2. Verify all file paths are correct
3. Ensure Python and packages are properly installed
4. If issues persist, check the script for any path modifications needed

## 📝 Example Successful Run
```bash
C:\Users\YourName>cd Desktop
C:\Users\YourName\Desktop>python "HSK Level model.py"
[INFO] Loading dataset...
[INFO] Found 1500 training samples
[INFO] Epoch 1/10 - Loss: 1.2345 - Accuracy: 45.67%
[INFO] Validation Accuracy: 50.12%
...
[INFO] Training completed!
[INFO] Best model saved as: best_optimized_model.pth
[INFO] Tokenizer saved as: best_optimized_model_tokenizer.pkl
```

---
