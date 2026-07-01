from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    CharacterTextSplitter,
    TokenTextSplitter,
    MarkdownHeaderTextSplitter,
    Language
)

from langchain_experimental.text_splitter import SemanticChunker
# from  import Embeddings

from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv

load_dotenv()

SAMPLE_TEXT = """
    Different Machine Learning Algorithms
Introduction

Machine learning is a branch of artificial intelligence that enables computers to learn patterns from data rather than relying solely on explicitly programmed instructions. Instead of defining every rule manually, machine learning algorithms analyze historical data, identify relationships, and use these relationships to make predictions or decisions when presented with new information.

Different machine learning algorithms are designed to solve different types of problems. Some algorithms are better suited for predicting numerical values, while others specialize in classification, clustering, anomaly detection, or recommendation systems. The choice of an algorithm depends on several factors, including the nature of the data, the complexity of the problem, computational resources, and the desired level of accuracy and interpretability.

Machine learning algorithms are generally divided into supervised learning, unsupervised learning, semi-supervised learning, and reinforcement learning. Within these categories, numerous algorithms have been developed, each offering unique strengths and limitations.

1. Linear Regression

Linear Regression is one of the simplest and most fundamental machine learning algorithms used for predictive analysis. It models the relationship between one or more independent variables and a continuous dependent variable by fitting a straight line through the observed data.

The algorithm assumes that a linear relationship exists between the input variables and the target variable. During training, it determines the line that minimizes the difference between predicted values and actual observations.

Linear Regression is widely used in finance, economics, healthcare, manufacturing, and scientific research for forecasting and trend analysis.

Characteristics
Supervised learning algorithm
Used for regression problems
Produces continuous numerical outputs
Easy to understand and interpret
Computationally efficient
Advantages
Simple implementation
Fast training
Highly interpretable
Works well for linear relationships
Limitations
Cannot model complex nonlinear relationships
Sensitive to outliers
Performance decreases when assumptions are violated
2. Logistic Regression

Despite its name, Logistic Regression is primarily used for classification rather than regression. It predicts the probability that an observation belongs to a particular category.

Instead of producing a continuous value, Logistic Regression estimates probabilities between 0 and 1, which are then converted into class labels using a decision threshold.

It is commonly applied in medical diagnosis, spam email detection, fraud detection, and customer churn prediction.

Characteristics
Supervised learning
Binary and multiclass classification
Probability-based prediction
Easy to interpret
Advantages
Simple and efficient
Works well on linearly separable data
Provides probability estimates
Less prone to overfitting on small datasets
Limitations
Cannot capture highly nonlinear relationships
Performance depends on feature quality
Sensitive to correlated features
3. Decision Tree

A Decision Tree is a supervised learning algorithm that makes predictions by repeatedly splitting data into smaller subsets based on feature values.

The structure resembles an upside-down tree where each internal node represents a decision, each branch represents an outcome, and each leaf node represents the final prediction.

Decision Trees are intuitive because their decision-making process closely resembles human reasoning.

Characteristics
Supervised learning
Classification and regression
Tree-structured model
Rule-based prediction
Advantages
Easy to visualize
Requires little data preprocessing
Handles numerical and categorical data
Highly interpretable
Limitations
Can easily overfit
Sensitive to small data changes
Lower accuracy than ensemble methods
4. Random Forest

Random Forest is an ensemble learning algorithm that combines multiple Decision Trees to improve prediction accuracy and robustness.

Each tree is trained on a different random subset of the data, and the final prediction is obtained by combining the outputs of all trees.

The algorithm reduces overfitting and improves generalization by averaging the predictions of many trees.

Characteristics
Supervised learning
Ensemble algorithm
Classification and regression
Uses multiple Decision Trees
Advantages
High accuracy
Resistant to overfitting
Handles missing values
Suitable for high-dimensional data
Limitations
Larger memory requirements
Less interpretable
Slower than a single Decision Tree
5. Support Vector Machine (SVM)

Support Vector Machine is a supervised learning algorithm designed to find the optimal boundary that separates different classes with the maximum possible margin.

Rather than simply separating classes, SVM seeks the boundary that provides the greatest confidence in classification.

SVM is effective for text classification, handwriting recognition, image analysis, and bioinformatics.

Characteristics
Supervised learning
Classification and regression
Margin-based learning
Effective in high-dimensional spaces
Advantages
High accuracy
Effective for complex datasets
Works well with limited training data
Robust against overfitting
Limitations
Computationally expensive
Difficult to tune
Less efficient for extremely large datasets
6. K-Nearest Neighbors (KNN)

K-Nearest Neighbors is an instance-based learning algorithm that classifies new data by examining the categories of nearby data points.

Instead of building a mathematical model during training, KNN stores the training data and performs computation only when making predictions.

The algorithm assumes that similar data points exist close to one another in the feature space.

Characteristics
Supervised learning
Classification and regression
Lazy learning algorithm
Distance-based prediction
Advantages
Simple implementation
No training phase
Handles multiclass problems
Flexible
Limitations
Slow prediction
Sensitive to irrelevant features
Requires careful selection of K
7. Naive Bayes

Naive Bayes is a probabilistic classification algorithm based on Bayes' Theorem. It assumes that all input features are statistically independent, even though this assumption rarely holds completely in real-world data.

Despite its simplicity, Naive Bayes performs remarkably well in many text-related applications.

Characteristics
Supervised learning
Probabilistic model
Fast training
Suitable for text data
Advantages
Extremely fast
Requires little training data
Handles high-dimensional datasets
Effective for document classification
Limitations
Assumes feature independence
Performance decreases when assumptions are violated
"""




SAMPLE_CODE = """

### Binary Search
    def recursive_binary_search(arr, left, right, target):
    if left > right:
        return -1

    mid = (left + right) // 2

    if arr[mid] == target:
        return mid

    elif target < arr[mid]:
        return recursive_binary_search(arr, left, mid - 1, target)

    else:
        return recursive_binary_search(arr, mid + 1, right, target)


numbers = [3, 7, 12, 18, 21, 26, 34, 45, 59, 68]

target = 34

index = recursive_binary_search(numbers, 0, len(numbers) - 1, target)

if index != -1:
    print(f"Element found at index {index}")
else:
    print("Element not found")





## Jump Search

import math


def jump_search(arr, target):
    n = len(arr)
    step = int(math.sqrt(n))
    prev = 0

    while prev < n and arr[min(step, n) - 1] < target:
        prev = step
        step += int(math.sqrt(n))

        if prev >= n:
            return -1

    while prev < min(step, n):
        if arr[prev] == target:
            return prev
        prev += 1

    return -1


numbers = [3, 7, 12, 18, 21, 26, 34, 45, 59, 68]

target = 45

index = jump_search(numbers, target)

if index != -1:
    print(f"Element found at index {index}")
else:
    print("Element not found")

"""

class SmartChunker:
    def __init__(
        self,
        embedding_model="BAAI/bge-small-en-v1.5",
        chunk_size=500,
        chunk_overlap=100,
    ):

        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model
        )

        self.semantic_splitter = SemanticChunker(
            embeddings=self.embeddings,
            breakpoint_threshold_type="percentile",
            breakpoint_threshold_amount=90,
        )

        self.recursive_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=[
                "\n\n",
                "\n",
                ". ",
                "? ",
                "! ",
                ";",
                ",",
                " ",
                ""
            ]
        )

        self.max_chunk_size = chunk_size

    def chunk(self, text):

        semantic_docs = self.semantic_splitter.create_documents([text])

        final_chunks = []

        for doc in semantic_docs:

            content = doc.page_content

            # Fallback condition
            if len(content) > self.max_chunk_size:

                recursive_docs = self.recursive_splitter.create_documents(
                    [content]
                )

                final_chunks.extend(recursive_docs)

            else:

                final_chunks.append(doc)

        return final_chunks


def recursive_spliter():

    splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n","\n"," ",""],
        chunk_size=500,
        chunk_overlap=50
    )
    
    chunks = splitter.split_text(SAMPLE_TEXT)
    

    print(f"Original Length : {len(SAMPLE_TEXT)}")
    print(f"Number of Chunks : {len(chunks)}")
    print(f"chunk size : {[len(chunk) for chunk in chunks]}")


def semantic_spliters():
    embeddings = HuggingFaceEmbeddings(
        model_name = "sentence-transformers/all-MiniLM-L6-v2"
    )

    smntic_splitter = SemanticChunker(
        embeddings=embeddings,
        breakpoint_threshold_type="percentile",
        breakpoint_threshold_amount=90 
    )

    documents = [
        Document(page_content=SAMPLE_TEXT),
        # Document(page_content=SAMPLE_CODE)
    ]

    docs = smntic_splitter.split_documents(documents)

    

    print(f"Document Length: {len(documents[0].page_content)}")
    print(f"Number of Chunks : {len(docs)}")
    print(f"Chunk Size: {[len(chunk.page_content) for chunk in docs]}")

    print("Preview Chunks:\n" +
        "\n\n".join(
            chunk.page_content[:100] for chunk in docs
        )
    ) 

def smart_chunker():
    smart_chunker = SmartChunker(
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        chunk_size=500,
        chunk_overlap=100,
    )

    chunks = smart_chunker.chunk(SAMPLE_TEXT)

    print(f"Document Length: {len(SAMPLE_TEXT)}")
    print(f"Number of Chunks : {len(chunks)}")
    print(f"Chunk Size: {[len(chunk.page_content) for chunk in chunks]}")

    print("Preview Chunks:\n" +
        "\n\n".join(
            chunk.page_content[:100] for chunk in chunks
        )
    ) 



if __name__ == "__main__":
    semantic_spliters()
    