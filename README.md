# image-captioning
# Image Captioning Using CNN and LSTM

## Project Overview
This project aims to develop an Image Captioning system using Convolutional Neural Networks (CNN) for image feature extraction and Long Short-Term Memory (LSTM) networks for caption generation. The system is designed to generate natural language descriptions of images, which can be useful in various applications such as assisting visually impaired individuals, automating social media content, and indexing large image databases.

## Team Members
- **Vaibhav Idupuluri** 
  
## Technologies Used
- **Deep Learning**: TensorFlow, Keras
- **Pretrained Model**: DenseNet201 for image feature extraction
- **RNN**: LSTM for text generation
- **GUI**: Streamlit for interactive web application
- **Other Libraries**: NumPy, Pandas, Matplotlib, Seaborn, tqdm

## Project Description
The goal of this project is to generate a caption for an image using a combination of CNNs (for extracting image features) and LSTMs (for generating captions based on those features). We utilize the **Flickr8k dataset** consisting of 8,000 images with multiple captions per image. The workflow follows these steps:

1. **Image Preprocessing**: Load and preprocess images using the Keras `ImageDataGenerator`.
2. **Text Preprocessing**: Clean and preprocess captions (lowercasing, removing special characters, padding).
3. **Feature Extraction**: Use DenseNet201 (a CNN model) to extract feature vectors from images.
4. **Model Development**: Build a model where the image features are combined with word embeddings and passed through an LSTM network to generate the caption.
5. **GUI**: Implement a simple web interface with Streamlit where users can upload an image and get the generated caption.

## Dataset: Flickr8k
- The **Flickr8k dataset** contains 8,000 images with 5 captions each.
- The dataset is split into a training set (85%) and a validation set (15%).
- The captions are preprocessed by converting them to lowercase, removing special characters, and tokenizing them.

## Model Architecture
- **Image Feature Extractor**: DenseNet201, pretrained on ImageNet, extracts features from input images.
- **Caption Generator**: An LSTM network that uses image features and previously generated words to predict the next word in the caption.
- **Encoder-Decoder Structure**: The model consists of two parts:
  - **Encoder**: DenseNet201 for image feature extraction.
  - **Decoder**: LSTM for generating captions.

## Streamlit Web Application
- Users can upload images, and the system will generate and display captions.
- The application provides an easy interface for generating captions without requiring technical knowledge.

