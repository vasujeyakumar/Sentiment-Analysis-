# Sentiment Analysis on Women's Clothing E-Commerce Reviews

## Project Overview
This project analyzes customer reviews from a women's clothing e-commerce platform to classify sentiments (positive or negative) and uncover common themes or topics. The model is deployed as a service using **FastAPI**, **Docker**, and **AWS**.

## Data Source
**Dataset**: `Womens Clothing E-Commerce Reviews.csv`  
The dataset contains columns such as `Review Text` and `Sentiment`, providing a rich source of feedback on customer experiences.

## Key Steps in the Project

### Data Preprocessing
- **Text Cleaning**: Removed unwanted characters, stopwords, and applied tokenization to prepare the text data.
- **Vectorization**: Used `Tokenizer` from Keras to convert text into numerical form by converting the review text into sequences and padding them to ensure uniform input size.

### Text Classification Models

#### LSTM Model
- **Architecture**: An LSTM-based deep learning model to capture sequential dependencies in text data.
  - **Input**: Tokenized sequences of reviews.
  - **Model Layers**: Embedding layer, LSTM layer with dropout, fully connected Dense layers.
  - **Output**: Sentiment classification (positive or negative).

#### RoBERTa Pre-trained Model
- **Pre-trained Model**: Hugging Face's `cardiffnlp/twitter-roberta-base-sentiment`.
- **Fine-tuning**: Fine-tuned on the e-commerce review dataset for improved performance in this specific domain.

### Topic Modeling
- **BERTopic**: Used for extracting key topics from customer reviews.
  - **Identified Themes**: Quality, size, shipping, and customer service.
  - **Business Insights**: Helped uncover trends to support better business decisions.

### API Development
- **FastAPI**: Created an API endpoint for real-time predictions.
  - **Endpoint**: Allows users to submit a review and receive a sentiment prediction.

### Model Deployment

#### Docker
- **Docker Image**: Built a Docker image that packages the application, including the trained model, API, and necessary libraries.
- **Best Practices**: Followed best practices, such as minimizing image size with multi-stage builds and specifying dependencies in a `requirements.txt` file.

#### AWS Deployment
- **Deployment**: Deployed the Docker container to AWS (EC2 or ECS) to provide a live service.
  - **Public URL**: The model is accessible via a public URL, enabling predictions from anywhere.

### AWS Deployment Steps
1. Built Docker image locally.
2. Pushed the Docker image to Amazon ECR (Elastic Container Registry).
3. Deployed the image on an EC2 instance or ECS (Elastic Container Service).
4. Configured security groups and load balancing to ensure public access.

## Challenges
- Handling **imbalanced data** in sentiment analysis.
- Optimizing the model to perform well with both **long** and **short customer reviews**.
- Ensuring the deployment is **scalable** and **robust**.

## Future Improvements
- Integrate **real-time feedback loops** to continuously retrain the model on fresh data.
- Improve topic modeling to incorporate more complex aspects of customer sentiment.
