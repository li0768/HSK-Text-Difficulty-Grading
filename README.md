# HSK Text Grading Based on Deep Hybrid Neural Network and RoBERTa

This repository contains the official PyTorch implementation for the paper: **"Research on HSK Chinese Text Grading Based on Deep Hybrid Neural Network Models"**. 

The proposed model is a Deep Hybrid Neural Network that classifies Chinese texts into seven HSK difficulty levels. It integrates character-level local features (Multi-scale CNN + BiLSTM + Multi-head Attention) with contextual semantic features (fine-tuned Chinese RoBERTa-wwm-ext). These heterogeneous features are concatenated (3,840 dimensions) and mapped into a unified manifold for final classification.

## 📊 Main Results

Under a strictly source-stratified split (Train/Val/Test = 3,649/454/463), the proposed hybrid model achieves:
*   **Test Accuracy:** 92.87% (Main Seed 123)
*   **Weighted F1-score:** 92.87% (Main Seed 123)
*   **Three-Seed Stability:** 92.22% ± 0.65% (Seeds 42, 123, 2024)

## 📂 Repository Structure

```text
├── experiments/            # Training, baseline, and ablation scripts
│   ├── run_main.py             # Main training for the DHNN components
│   ├── run_roberta.py          # Fine-tuning script for RoBERTa
│   ├── run_bert_baseline.py    # BERT baseline training
│   ├── run_kgmnn.py            # KGMNN baseline training
│   └── run_ablation.py         # Ablation studies (-CNN, -BiLSTM, etc.)
├── models/                 # PyTorch model architectures (DHNN, etc.)
├── utils/                  # Data loaders (split_by_source), metrics, and helpers
└── README.md
🛠️ Environment Setup
All experiments were conducted on a CPU-only environment (AMD Ryzen 7 4800U, 32GB RAM). GPU acceleration is supported and will significantly speed up the RoBERTa fine-tuning.

Clone this repository:

bash
git clone https://github.com/li0768/HSK-Text-Difficulty-Grading.git
cd HSK-Text-Difficulty-Grading
Install dependencies:

bash
pip install -r requirements.txt
🚀 Reproduction Guide
Due to copyright restrictions on the raw textbook content, the raw texts are not included. However, the complete data split logic (split_by_source), preprocessing, and training scripts are provided.

Before running the scripts, ensure the BASE_PATH in the experiment files points to your local dataset directory.

Step 1: Train the main DHNN components
bash
python experiments/run_main.py
Step 2: Fine-tune the RoBERTa encoder
bash
python experiments/run_roberta.py
(Note: If you are in mainland China, set the mirror before running: set HF_ENDPOINT=https://hf-mirror.com)

Step 3: Feature Fusion & Evaluation
Ensure you have the prediction JSON files from Step 1 and Step 2. You can use the provided fusion script (if available) or perform probability averaging to reproduce the final 92.87% metric.

Step 4: Run Baselines & Ablation Studies
bash
python experiments/run_bert_baseline.py
python experiments/run_kgmnn.py
python experiments/run_ablation.py
📝 Citation
If you find this code useful in your research, please consider citing our paper:

bibtex
@article{li2024hsktxt,
  title={Research on HSK Chinese Text Grading Based on Deep Hybrid Neural Network Models},
  author={Li, Siwei and Liu, Hua},
  journal={[Journal Name]},
  year={2024}
}
📄 License
This project is licensed under the MIT License. The raw copyrighted textbook content is not covered by this license and must be obtained through legal channels.