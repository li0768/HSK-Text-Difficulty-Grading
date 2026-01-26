HSK Level Classification Model
This repository contains the code and resources for training a HSK (Hanyu Shuiping Kaoshi) level classification model.

Prerequisites
Python 3.7+

PyTorch 1.8+

Required Python packages (install via pip install -r requirements.txt if available)

Quick Start Guide
Step 1: Download Files
Download the following two zip files to your Desktop:

数据集（8：1：1划分训练集，测试集，验证集）.zip - Contains pre-split dataset (8:1:1 ratio for train/test/validation)

Supplementary File (Training Model Files).zip - Contains model training files

Step 2: Extract Files
Extract both zip files on your Desktop.

Step 3: Prepare Training Script
Move HSK Level model.py from the extracted Supplementary File (Training Model Files) folder to your Desktop.

Step 4: Start Training
Open Command Prompt (cmd)

Navigate to your Desktop:

bash
cd %USERPROFILE%\Desktop
Run the training script:

bash
python "HSK Level model.py"
File Structure
text
Desktop/
├── 数据集（8：1：1划分训练集，测试集，验证集）/
│   ├── train/
│   ├── test/
│   └── validation/
├── Supplementary File (Training Model Files)/
│   ├── HSK Level model.py (move to Desktop)
│   ├── *.pth (trained model weights)
│   └── *.ptl (model-related files)
└── HSK Level model.py
Model Files
The Supplementary File (Training Model Files).zip contains:

HSK Level model.py - Main training script

.pth files - PyTorch model weights (best model after training)

.ptl files - Additional model configuration files

Training Process
When you run HSK Level model.py, the script will:

Load the dataset from the extracted folder

Initialize the model architecture

Train using the 8:1:1 split dataset

Save the best model as .pth and .ptl files

Output training progress and evaluation metrics

Expected Output
After successful training, you should see:

Training loss and accuracy metrics

Validation/test set performance

Model files saved in the current directory

Troubleshooting
File not found errors: Ensure all files are extracted to the correct locations on your Desktop

Python/pip not recognized: Add Python to your system PATH or use absolute path to Python executable

Missing dependencies: Install required packages manually:

bash
pip install torch torchvision
Notes
The .pth and .ptl files are generated automatically after training completes

Training time may vary depending on your hardware configuration

For best results, ensure you have sufficient disk space and memory

Support
For issues or questions, please check the repository's issue tracker or contact the maintainer.
