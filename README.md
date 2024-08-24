
# Effective text classification using BERT, MTM LSTM, and DT

This repository contains two Python notebooks that implement BERT-LSTM models for text classification tasks. The IMDB reviews dataset was used for two-class classification and the drug reviews dataset was used for three-class classification. The code also includes integration with [Weights and Biases (WandB)](https://wandb.ai/) for experiment tracking and visualization.

## Article Reference

The code in this repository was developed for the research article titled **"Effective text classification using BERT, MTM LSTM, and DT"**, published in *Computational Science & Technology*. You can read the full article [here](https://www.sciencedirect.com/science/article/abs/pii/S0169023X24000302).

## Notebooks

### 1. **3-Classes BERT-LSTM Classification**
- **Description**: This notebook classifies drug reviews into three categories: negative, neutral, and positive, based on user ratings.
- **Classes**:
  - **Class 1**: Negative (ratings < 4)
  - **Class 2**: Neutral (ratings between 4 and 7)
  - **Class 3**: Positive (ratings > 7)
- **Key Features**:
  - Preprocessing steps such as tokenization and stopword removal.
  - A BERT model followed by an LSTM layer for sequence classification.
  - Integration with WandB for logging metrics such as accuracy, precision, recall, and F1-score.
  - Visualization of results using Plotly.

### 2. **2-Classes BERT-LSTM Classification**
- **Description**: This notebook classifies drug reviews into two categories: negative and positive, based on user ratings.
- **Classes**:
  - **Class 0*: Negative
  - **Class 1**: Positive
- **Key Features**:
  - Similar preprocessing and model structure as the 3-class notebook.
  - Adapted for a binary classification task.
  - WandB integration for tracking and comparing different runs.

## Prerequisites

Before running the notebooks, ensure you have the following libraries installed:

```bash
pip install transformers wandb chart-studio nltk
python -m nltk.downloader stopwords

How to Run

    Clone the repository:

    bash

    git clone https://github.com/saman1374/BERT-and-MTM-LSTM-for-Text-Classification.git
    cd bert-lstm-classification

    Set up Weights and Biases (WandB):
        Create a free account at WandB.
        Replace the login credentials in the notebook with your own.
        Initialize WandB in the notebook by running wandb.login().

    Load your dataset:
        Place your dataset in the appropriate path as specified in the notebook (e.g., dataset = pd.read_csv('your_dataset.csv')).
        Ensure the dataset structure matches the expected format.

    Run the notebooks:
        Open the .ipynb files in Jupyter Notebook or JupyterLab.
        Run the cells sequentially to preprocess data, train the model, and log the results to WandB.

Results and Visualization

After training, the results will be logged to WandB, where you can visualize metrics like accuracy, precision, recall, and F1-score. The notebook also includes a Plotly visualization for comparing model performance across different runs.
Project Structure

plaintext

├── 3_CLASSES_BERT_LSTM.ipynb        # Notebook for 3-class classification
├── 2_CLASSES_BERT_LSTM.ipynb        # Notebook for 2-class classification
└── README.md                        # Project description and instructions

Contributing

Contributions are welcome! Please open an issue or submit a pull request if you have suggestions or improvements.
License

This project is licensed under the MIT License - see the LICENSE file for details.

vbnet


### Additional Notes:
- **Article Reference**: Clearly states that the code is associated with a published article and provides a direct link.
- **Notebooks Section**: Provides concise descriptions of the notebooks and their purposes.
- **How to Run**: Gives instructions on setting up the environment and running the notebooks.
- **License**: Includes a placeholder for the license type. If your project is open-source, you can add an appropriate license.

This README will make it clear to anyone visiting the repository that the code is tied to academic research, and they'll have easy access to the article itself
