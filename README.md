# Natural-Language-Processing-with-Disaster-Tweets

This project focuses on classifying whether tweets are about real disasters or not using natural language processing (NLP) techniques. The dataset consists of tweets that are labeled as either referring to a real disaster or not. The aim is to create a deep learning model that is able to classify the label based on the tweet.

## Technologies

* [x] **Python:** Core programming language used for data processing and model training.
* [x] **NLTK:** Used for text preprocessing, tokenization, and lemmatization.
* [x] **Keras:** High-level neural networks API, used to build the deep learning model.
* [x] **TensorFlow:** Backend engine for Keras, used for training the model.
* [x] **NumPy:** Library for numerical computations and data manipulation.
* [x] **Pandas:** Library for numerical computations and data manipulation.

## Dataset

- **Training Set**: 7,613 tweets with their corresponding labels (1 for disaster, 0 for non-disaster).
- **Test Set**: 3,263 tweets without labels.
- **Source**: [Kaggle - NLP with Disaster Tweets](https://www.kaggle.com/competitions/nlp-getting-started).

## Model
The model is a simple yet effective neural network designed for text classification tasks. The architecture includes the following layers:
- **Embedding Layer:** Converts the input words into dense vectors of fixed size, where each word in the input is represented as a word embedding. This helps the model capture semantic relationships between words.
- **GlobalAveragePooling1D Layer:** Reduces the sequence of word embeddings to a single vector by taking the average of the embeddings across all input tokens. This helps summarize the entire sequence into a fixed-length vector.
- **Dense Layer:** A fully connected layer with 64 units and ReLU activation to introduce non-linearity and learn complex patterns in the data.
- **Dropout Layer:** A regularization technique that randomly drops 10% of the neurons during training to prevent overfitting and improve generalization.
- **Output Dense Layer:** A single-unit fully connected layer with a sigmoid activation function to output a probability score between 0 and 1, indicating whether the tweet is disaster-related (1) or not (0).

The model was compiled using the binary cross-entropy loss function suitable for a binary classification task and trained with Adam optimization. This architecture is easy and light, making it efficient in terms of training and inference times, while still achieving good performance on the dataset.

## Results

| Metric                     | Value    |
|----------------------------|----------|
| **Training Accuracy**      | 91.71%   |
| **Training Loss**          | 0.2191   |
| **Validation Accuracy**    | 80.96%   |
| **Validation Loss**        | 0.4662   |
| **Submisson Score**        | 80.26%   |

**Confusion matrix of test data:**

![image](https://github.com/user-attachments/assets/c2e59be9-6c50-4c2d-a217-073d4c780a2f)

