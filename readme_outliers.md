# Rogue Dimensions Are Good For You: Code Pipeline

This repository contains the code used in our paper:

**"Not a nuisance but a useful heuristic: Outlier dimensions favor frequent tokens in language models"**  
*Authors: Iuri Macocco and Nora Graichen and Gemma Boleda and Marco Baroni*  
Accepted at BlackBoxNLP @ EMNLP 2025 
[https://arxiv.org/abs/2503.21718]

## 🧠 Overview

This project implements the full pipeline used in our study, including:
- Data loading and preprocessing
- Model run and extraction of latent representations
- Evaluation and result generation

The code is modular and designed for reproducibility and ease of extension.

## 📁 Repository Structure

```
project-name/
├── README.md              # Project overview
├── environment.yml        # Dependencies
├── main.py                # Entry point for running the pipeline
├── data/                  # Input data
├── notebooks/             # Jupyter notebooks for demo/exploration
├── results/               # Main output, model specific
├── src/
│   ├── preprocess.py      # Data preprocessing functions
│   ├── running.py         # Extract the representations
│   ├── evaluate.py        # Find ODs and perform ablations
│   └── analysis.py        # Run the final analysis and generate plots
└── LICENSE
```

## ⚙️ Installation

Clone the repo and install dependencies (using conda):

```bash
git clone https://github.com/your-username/project-name.git
cd project-name
conda env create -f environment.yml
conda activate your-env-name
```

## 🚀 Usage

To run the full pipeline:

```bash
python main.py --config config.yaml

%Or run modules individually:

python src/preprocess.py
python src/train.py
python src/evaluate.py
```

For a quick example set ```run.model: fast``` in the config.yaml, the code will run on pythia-70m on 500 sentences in less than 30 seconds.

## 📊 Results

Output results (e.g., model checkpoints, plots, metrics) are saved in the results/model directory by default. See the paper for detailed results and analysis.

## 📄 Citation

If you use this code, please cite our paper:
```
@misc{macocco:etal:2025,
      title={Outlier dimensions favor frequent tokens in language models}, 
      author={Iuri Macocco and Nora Graichen and Gemma Boleda and Marco Baroni},
      year={2025},
      eprint={2503.21718},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2503.21718}, 
}
```
## 🛠 License

This project is licensed under the MIT License.
