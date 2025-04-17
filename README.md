# MINI-GPT2: A Simplified GPT-2 Implementation

!Work in Progress

## Overview
This project implements a smaller-scale version of the GPT-2 model from scratch, based on the original GPT-2 paper by OpenAI. The goal is to train the model on the Tiny Shakespeare dataset, a simple text corpus, to generate Shakespeare-like text. The model will have fewer layers, a smaller hidden size, and fewer attention heads compared to the original GPT-2, making it suitable for experimentation on limited compute resources.

## Project Structure
- **data/**: Stores the Tiny Shakespeare dataset and preprocessing scripts.
- **models/**: Contains the GPT-2 model implementation.
- **scripts/**: Includes training, evaluation, and inference scripts.
- **config/**: Stores configuration files (e.g., hyperparameters).
- **checkpoints/**: Saves model weights during training.
- **notebooks/**: Jupyter notebooks for data exploration and experimentation.
- **results/**: Stores generated text, metrics, and plots.
- **tests/**: Unit tests for model components.
- **utils/**: Helper functions (e.g., tokenization, logging).

```

MINI-GPT2/
├── .git/             # Git version control files
├── checkpoints/      # Saved model weights during/after training
├── config/           # Configuration files (hyperparameters, model settings)
├── data/             # Datasets (e.g., TinyShakespeare input.txt)
├── models/           # Core GPT-2 model architecture implementation
├── notebooks/        # Jupyter notebooks for experimentation and visualization
├── results/          # Output files (generated text, evaluation metrics, logs)
├── scripts/          # Runnable Python scripts (training, generation, preprocessing)
├── tests/            # Unit tests for code validation
├── utils/            # Utility functions (tokenization, data loading, etc.)
├── venv/             # Python virtual environment files
├── .gitignore        # Files/directories ignored by Git
├── README.md         # This file: Project overview and instructions
└── requirements.txt  # Project dependencies

```

## Setup Instructions
1. **Clone the Repository**  
   ```bash
   git clone https://github.com/rishikesh2715/mini-GPT2.git
   cd mini-GPT2
   ```

2. **Set Up a Virtual Environment**  
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**  
   Install the required packages listed in `requirements.txt`:  
   ```bash
   pip install -r requirements.txt
   ```

## Task Assignment

The MINI-GPT2 project tasks are divided among the team as follows:


- **Afshan**  
  - **Data Preparation and Preprocessing**: Download the Tiny Shakespeare dataset and preprocess it (clean text, handle special characters). 
  Implement a tokenizer (BPE), build a vocabulary, and convert text to token IDs. 
  Save preprocessed data in `data/`. Analyze the dataset in a notebook in `notebooks/` (vocabulary size, sequence length stats).

- **Rishikesh**  
  - **Model Implementation**: Implement the GPT-2 architecture in `models/` (multi-head self-attention, feed-forward layers, positional encoding, layer normalization, dropout).  
  
  - **Model-Related Testing**: Write unit tests for model components in `tests/` (attention, feed-forward layers). Debug the model (verify attention weights, gradient flow).  
  
  - **Training Assistance**: Help training pipeline, ensuring the model integrates well with the training script in `scripts/`.

- **Manashi**  
  - **Training Pipeline**: Write the training script in `scripts/` (`train.py`), 
  including data loading, batching, and padding for variable-length sequences. 
  Implement the training loop with a loss function (cross-entropy), optimization (Adam), 
  and learning rate scheduling. 
  Save checkpoints in `checkpoints/` and log metrics in `utils/`, saving them in `results/`. 
  Create a configuration file in `config/` for hyperparameters.  
  
  - **Evaluation and Inference**: Write evaluation and inference scripts in `scripts/` to 
  compute metrics like perplexity and generate text samples. 
  Save results in `results/`. Visualize metrics in a notebook in `notebooks/`.

- **Nayla**  
  - **Documentation**: Update `README.md` with detailed setup, usage, and results sections. 
  Document code in `models/`, `scripts/`, and `utils/` with comments and docstrings. 
  Track progress (Git issues). Maintain `requirements.txt` and `.gitignore`.

  - **Robust Testing and Evaluation**: Write unit tests in `tests/` for data preprocessing and training utilities. 
  Test the training pipeline on a small data subset. 
  Evaluate model performance under different hyperparameters (learning rate, batch size, number of layers) 
  by running experiments and comparing metrics like perplexity or generated text quality. 
  Save results in `results/` and document findings in a notebook in `notebooks/`.

## Usage
(TBD: Instructions for training, evaluation, and inference will be added once scripts are implemented.)

## Results
(TBD: Generated text samples, training metrics, and plots will be added after training.)


## License
This project is licensed under the MIT License.
