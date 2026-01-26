# HSK Level Classification Model

## 📋 Prerequisites
- **Python 3.8+** (Recommended: Python 3.8-3.11)
- **Required Packages**:
  ```bash
  pip install torch torchvision torchaudio
  pip install numpy pandas scikit-learn
  pip install matplotlib tqdm
  ```

## 🚀 Quick Start

### Step 1: Download Files
Download the following two zip files directly to your **Desktop**:
1. **数据集（8：1：1划分训练集，测试集，验证集）.zip** - Pre-split dataset (8:1:1 train/test/validation ratio)
2. **Supplementary File (Training Model Files).zip** - Training scripts and model templates

### Step 2: Extract Files
Extract both zip files on your Desktop. The structure should be:
```
Desktop/
├── 数据集（8：1：1划分训练集，测试集，验证集）/
│   ├── train/          # Training data (80%)
│   ├── test/           # Testing data (10%)
│   └── validation/     # Validation data (10%)
└── Supplementary File (Training Model Files)/
    ├── HSK Level model.py
    ├── *.pth           # Model template files
    └── *.ptl           # Configuration files
```

### Step 3: Prepare Training Script
Locate `HSK Level model.py` inside the `Supplementary File (Training Model Files)` folder and **move it to your Desktop**.

### Step 4: Start Training
1. **Open Command Prompt (cmd)**
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
├── 数据集（8：1：1划分训练集，测试集，验证集）/  # Extracted dataset
│   ├── train/              # 80% of total data
│   │   ├── class1/
│   │   ├── class2/
│   │   └── ...
│   ├── test/               # 10% of total data
│   └── validation/         # 10% of total data
├── Supplementary File (Training Model Files)/  # Original files (optional)
│   ├── HSK Level model.py
│   ├── *.pth
│   └── *.ptl
├── HSK Level model.py      # Moved here for execution
├── best_model.pth          # Generated after training
└── model_checkpoint.ptl    # Generated after training
```

## ⚙️ Automatic Configuration
The training script is pre-configured to:
- **Automatically detect** the dataset folder on Desktop
- **Load data** with 8:1:1 split already applied
- **Save best model** to Desktop after training
- **Generate training logs** in console

## 📊 Expected Training Process
When you run the script, you will see:
```
[INFO] Loading dataset from: C:\Users\YourName\Desktop\数据集（8：1：1划分训练集，测试集，验证集）
[INFO] Found X training samples, Y validation samples, Z test samples
[INFO] Starting training...
[Epoch 1/10] Loss: X.XXXX, Accuracy: XX.XX%
[Epoch 2/10] Loss: X.XXXX, Accuracy: XX.XX%
...
[INFO] Training completed!
[INFO] Best model saved to Desktop
```

## ⚠️ Important Notes
1. **No Path Configuration Needed**: The script automatically searches for dataset on Desktop
2. **Model Files**: 
   - `.pth` files contain trained model weights
   - `.ptl` files contain model architecture and configurations
3. **Training Time**: Depends on dataset size, typically 10-30 minutes
4. **Storage**: Ensure Desktop has at least 2GB free space for generated files

## 🔍 Troubleshooting Guide

| Problem | Solution |
|---------|----------|
| **File Not Found** | Ensure both zip files are extracted directly on Desktop |
| **Python Not Recognized** | Add Python to PATH or use full path: `C:\Python39\python.exe "HSK Level model.py"` |
| **Missing Packages** | Run: `pip install torch numpy pandas scikit-learn` |
| **Permission Error** | Run Command Prompt as Administrator |
| **Out of Memory** | Reduce batch size in script (if configurable) |

## 📈 Output Files Explanation
After successful training, these files will appear on your Desktop:

1. **best_model.pth** 
   - Contains the trained model weights
   - Can be loaded for inference or further training
   - File size: ~100-500MB

2. **model_checkpoint.ptl**
   - Contains model architecture and training configuration
   - Includes optimizer state and hyperparameters
   - Essential for resuming training

3. **Training Logs** (in console)
   - Epoch-wise loss and accuracy
   - Validation performance
   - Final test set evaluation

## 💡 Pro Tips
1. **Monitor Training**: Watch the loss values decreasing each epoch
2. **Verify Output**: Check that `.pth` and `.ptl` files are created after training
3. **Backup Models**: Copy generated files to a safe location
4. **Clean Up**: Remove original zip files after extraction to save space

## ✅ Quick Verification
To verify everything is set up correctly:

1. **Check Desktop contents**:
   ```
   Desktop/
   ├── 数据集（8：1：1划分训练集，测试集，验证集）/ (folder)
   ├── Supplementary File (Training Model Files)/ (folder)
   └── HSK Level model.py (file)
   ```

2. **Run quick test**:
   ```bash
   python --version  # Should show Python 3.8+
   python -c "import torch; print(torch.__version__)"  # Should print version
   ```

## 🆘 Need Help?
If the script doesn't run:
1. **Check Step 3**: Ensure `HSK Level model.py` is on Desktop (not inside any folder)
2. **Check Extraction**: Both zip files must be extracted (not just copied)
3. **Check Python**: Type `python` in cmd to verify it's installed
4. **Check Dependencies**: Install all required packages from Prerequisites section

---
**Model Version**: 1.0  
**Last Updated**: January 2024  
**Dataset Split**: 8:1:1 (Train:Test:Validation)  
**Compatibility**: Windows 10/11, Python 3.8+

*For issues, ensure you've followed all steps exactly as described above.*
