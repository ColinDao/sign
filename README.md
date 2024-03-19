# What's That Sign Say?

## Project Description

This project aims to build a Convolutional Neural Network (CNN) using TensorFlow to classify images from the German Traffic Sign Recognition Benchmark (GTSRB) dataset. The GTSRB dataset consists of thousands of images of traffic signs, categorized into different classes. The CNN model will be trained to swiftly and accurately classify these images into their respective traffic sign categories, enabling the recognition of traffic signs from real-world images.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Features](#features)
- [Technologies](#technologies)
- [Credit](#credit)
- [License](#license)

## Installation

You'll need to have Python and pip3 installed. You can download them from the [official Python website](https://www.python.org/downloads/).

1. Clone the repository:

```bash
git clone https://github.com/ColinDao/whats-that-sign-say.git
```

2. Navigate to the project directory:

```bash
cd whats-that-sign-say
```

3. Install the required dependencies:

```bash
pip3 install -r requirements.txt
```

## Usage

To play against the AI, run the following command:

```bash
python traffic.py
```

Watch as the model learns from its predictions! It'll learn to astoundingly become more and more accurate in how it distinguishes images.

## Features

**CNN Architecture**: Design and implement a CNN architecture using TensorFlow, comprising convolutional layers, pooling layers, and fully connected layers to extract features from the input images and perform classification. Tuned using specific hyperparameters such as ReLU activation and Categorical Cross-Entropy loss functions. <br />
<br />
**Data Preprocessing**: Preprocess the GTSRB dataset to prepare the images for training, including extracting, resizing, labeling, and data augmentation techniques to enhance model generalization and performance.<br />
<br />
**Model Training**: Train the CNN model using the preprocessed GTSRB dataset, employing techniques such as backpropagation to optimize the model parameters and minimize classification error. <br />
<br />
**Evaluation Metrics**: Evaluate the trained model's performance using evaluation metrics such as accuracy on a separate test dataset to assess its effectiveness in classifying traffic sign images.

## Credit

This project was completed as a part of [CS50's Introduction to Artificial Intelligence with Python](https://cs50.harvard.edu/ai/2024/). Go check them out!

## Technologies
**Language**: Python <br />
**Libraries**: OpenCV, NumPy, OS, Sys, TensorFlow, Scikit-learn

## License

MIT License

Copyright (c) 2024 Colin Dao

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
