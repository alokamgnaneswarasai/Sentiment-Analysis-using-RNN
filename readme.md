# Text Classification with RNN and FastText Embeddings

This project implements a text classification task using a Recurrent Neural Network (RNN) in PyTorch. The task is to classify text data into discrete categories ranging from 1 to 5. FastText pretrained word embeddings are used to represent words, and the classification is performed based on the final hidden state of the RNN.

## Project Structure

- `preprocessing.py`: Handles text preprocessing, including tokenization, converting text to FastText embeddings, and preparing data for training.
- `dataloader.py`: Contains the `TextDataset` class for loading the data into PyTorch DataLoader.
- `model.py`: Defines the RNN-based model architecture for text classification.
- `train.py`: Script to train the RNN model on the provided dataset.
- `eval.py`: Contains functions to evaluate the model's performance, such as accuracy calculation.
- `test.py`: Script to evaluate the trained model on test data.
- `README.md`: Project documentation.

## Dependencies

- Python 3.8+
- PyTorch
- Gensim (for FastText embeddings)
- Pandas
- Numpy

### Installing Dependencies

Install the dependencies using pip:

```bash
pip install torch gensim pandas numpy
```

## Data Format

The dataset should be in CSV format with the following columns:

* **`overall`** : The target label, an integer between 1 and 5, representing the rating or classification.
* **`reviewText`** : The text data to be classified, typically a sentence, review, or paragraph.

## Example Data

overall,reviewText
5,"I bought this memory card for my galaxy note 2 phone. I saved a lot of money by purchasing the 16gb phone and just buying this memory card."
3,"The product is okay but the shipping was very slow."
2,"Not satisfied with the quality of the material."
4,"Works as expected. Decent quality for the price."
1,"Terrible experience, would not recommend."

## Usage

### 1. Preprocessing

The `preprocessing.py` script tokenizes the text, converts it into FastText embeddings, and prepares it for training/testing.

### 2. Training

To train the model, run:

python3 train.py

### 3. Testing

To evaluate the model on test data, run:

python3 test.py

## Files Explanation

### `preprocessing.py`

This file handles the preprocessing of text data. It includes the following functions:

* `load_data(file_path, max_seq_length)`: Loads the data from a CSV file, preprocesses the text, and converts it into a format suitable for the model.
* `text_to_vector(text, max_seq_length)`: Converts the preprocessed text into a sequence of FastText embeddings.
* `preprocess_text(text)`: Tokenizes and cleans the input text.
* `avg_sequence_length(data)`: Computes the average length of the sequences in the dataset.

### `dataloader.py`

This file contains the `TextDataset` class, which prepares the data for the DataLoader. It handles batching, shuffling, and padding sequences to `max_seq_length`.

### `model.py`

This file defines the `RNNClassifier` model, which is composed of:

1. **Embedding Layer** : Loads FastText embeddings for each word in the input text.
2. **RNN Layer** : Processes the sequence of word embeddings.
3. **Fully Connected Layer** : Maps the final hidden state to the output classes.

### `train.py`

This script trains the RNN model. Key functions and components include:

* `train(model, dataloader, criterion, optimizer, device, num_epochs)`: Handles the training loop, including forward passes, loss computation, backpropagation, and model updating.

### `eval.py`

Contains the evaluation functions:

* `calculate_accuracy(model, dataloader, device)`: Calculates and returns the accuracy of the model on a given dataset.

### `test.py`

This script evaluates the trained model on test data. It loads the model from `model.pth` and prints the accuracy on the test set.

## Hyperparameters

Some key hyperparameters used in the project:

* `max_seq_length`: Maximum length of sequences (default: 100)
* `batch_size`: Size of each batch for training/testing (default: 32)
* `hidden_dim`: Number of units in the RNN hidden layer (default: 128)
* `num_epochs`: Number of training epochs (default: 10)

These can be adjusted in the respective scripts.

## Model Architecture

The model is a simple RNN with the following layers:

1. **Embedding Layer** : Loads FastText embeddings for each word in the input text.
2. **RNN Layer** : Processes the sequence of word embeddings.
3. **Fully Connected Layer** : Maps the final hidden state to the output classes.

## Evaluation

The model is evaluated using Cross-Entropy Loss and accuracy metrics. The evaluation results (loss and accuracy) are printed after the training and testing phases.


## Results

Results on SST-dataset for different hyper-parameter and model  configurations

| Emb-dim | lr     | Epochs | batch_size | Dropout | model            | Val-loss | Val-accuracy | F1-score |
| ------- | ------ | ------ | ---------- | ------- | ---------------- | -------- | ------------ | -------- |
| 300     | 0.002  | 20     | 256        | 0.5     | CNNClassifier    | 0.42     | 82.91%       | 0.826    |
| 300     | 0.001  | 20     | 256        | 0.5     | CNNClassifier    | 0.41     | 84.12%       | 0.8358   |
| 300     | 0.0001 | 30     | 64         | 0.5     | CNNClassifier    | 0.39     | 86.3%        |          |
|         | 0.0005 |        | 64         |         | Bi_GRUClassifier |          | 88.1%        |          |
