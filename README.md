# Model Training Guide

## 📋 Prerequisites
- Python 3.8 or higher
- Required packages (install via pip):

```bash
pip install torch transformers numpy pandas scikit-learn
```

## 🚀 Quick Start

### Step 1: Prepare Dataset
1. Download the `Supplementary File (Dataset).zip` file
2. Extract the archive to a convenient location (e.g., Desktop)
3. The extracted folder structure will be:
   ```
   Supplementary File (Dataset).zip\
   └── 李思卫五套教材（分级后教材_已过滤）\
       └── 所属难度范围（初级（1-3级），中级（4-6级），高级（7-9级））\
           ├── 初级（1-3级）\
           │   ├── 1级\
           │   ├── 2级\
           │   └── 3级\
           │       └── 新标准中文 初级中文3\
           │           ├── lesson1.txt
           │           ├── lesson2.txt
           │           └── ...
           ├── 中级（4-6级）\
           └── 高级（7-9级）\
   ```

### Step 2: Place Training Script
Locate the `HSK Level model.py` file in the `Training Model Files` folder and copy it to your working directory (e.g., Desktop).

### Step 3: Configure Paths (Important!)
Before running the script, you may need to update the dataset path inside `HSK Level model.py` to match your extraction location. For example:

```python
# Update this line in the script if necessary
dataset_path = "C:/Users/YourUsername/Desktop/Supplementary File (Dataset)/李思卫五套教材（分级后教材_已过滤）"
```

### Step 4: Run Training Script
Open Command Prompt (cmd) and navigate to your working directory:

```bash
cd Desktop
python "HSK Level model.py"
```

### Step 5: Expected Output
After successful execution, the following files will be generated in your working directory:
- `best_optimized_model.pth` (model weights)
- `best_optimized_model_tokenizer.pkl` (tokenizer)

## 📁 Recommended Project Structure
```
Desktop/
├── Supplementary File (Dataset)/
│   └── 李思卫五套教材（分级后教材_已过滤）/
│       └── 所属难度范围（初级（1-3级），中级（4-6级），高级（7-9级））/
│           ├── 初级（1-3级）/
│           ├── 中级（4-6级）/
│           └── 高级（7-9级）/
├── HSK Level model.py
├── best_optimized_model.pth (generated after training)
└── best_optimized_model_tokenizer.pkl (generated after training)
```

## ⚠️ Important Notes
1. **Path Considerations**: The script expects Chinese folder names. If you encounter encoding issues:
   - Keep original Chinese folder names
   - Or modify the script's path variables accordingly

2. **Dataset Structure**: The training script should automatically traverse through:
   - Difficulty levels (Beginner/Intermediate/Advanced)
   - Specific HSK levels (1-9)
   - Textbook folders
   - Individual lesson `.txt` files

3. **File Encoding**: Ensure text files are UTF-8 encoded to avoid reading errors

## ❓ Troubleshooting
| Issue | Solution |
|-------|----------|
| FileNotFoundError | Verify the dataset path in the script matches your extraction location |
| Encoding errors | Check that all `.txt` files are UTF-8 encoded |
| ModuleNotFoundError | Install required packages using `pip install -r requirements.txt` (if available) |
| Memory issues | Reduce batch size in the training script if using limited RAM |

## 📝 Example Path
For a typical installation on Windows:
```
C:\Users\[YourUsername]\Desktop\Supplementary File (Dataset)\李思卫五套教材（分级后教材_已过滤）\所属难度范围（初级（1-3级），中级（4-6级），高级（7-9级））\初级（1-3级）\3级\新标准中文 初级中文3\lesson1.txt
```

---

**Note**: The training script should handle the nested folder structure automatically. If you encounter path-related errors, check the directory structure matches the example above.
