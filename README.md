# Image2Code

## Overview

Image2Code aims to provide multiple methods to convert a graphical user interface (GUI) image screenshot into corresponding CSS and HTML code. This project explores and implements various techniques to achieve this conversion efficiently and accurately.
It demonstrates that deep learning methods can be leveraged to train a model end-to-end to automatically generate code from a single input image.

It implements the following approaches for Image to Code Generation

1. **Pix2Code** Train a ml model end-to-end to automatically generate code from a single input image with over **77% accuracy** for three different platforms: **iOS**, **Android**, and **Web**. This transformation process helps developers quickly build customized software, websites, and mobile applications from designer-created GUI screenshots.

## Pix2Code Demo

https://github.com/user-attachments/assets/6db80978-cd63-4d30-8717-3c941cc0cd31


## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Pix2Code](#pix2code)
- [Contributing](#contributing)
- [License](#license)

## Features

- Convert GUI screenshots to HTML and CSS code
- Support for multiple platforms: iOS, Android, and web
- High accuracy in code generation using deep learning models
- Easy-to-use interface for uploading images and obtaining code

## Installation

### Prerequisites

- Python 3.10 or higher
- TensorFlow
- Keras
- OpenCV
- Streamlit (for the web interface)
- Other dependencies listed in `requirements.txt`

### Steps

1. Clone the repository:
    ```bash
    git clone https://github.com/Abhinav1004/Image2Code.git
    cd Image2Code
    ```

2. Create and activate a virtual environment:
    ```bash
    python -m venv venv
    source venv/bin/activate   # On Windows: venv\Scripts\activate
    ```

3. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

1. Start the web interface:
    ```bash
    cd pix2code/src/streamlit_demo
    streamlit run pix2code_demo.py
    ```

2. Open your web browser and go to `http://localhost:5000`.

3. Upload a GUI image screenshot.

4. Click on "Generate Code" to get the corresponding HTML and CSS code.

## Pix2Code

Detailed information about the Pix2Code method, including the model architecture, training process, and evaluation metrics, can be found in the `pix2code` directory within this repository. This section contains:

- Implementation of Pix2Code approach 
- Model architecture and configurations
- Training scripts and instructions
- Evaluation metrics and results

To learn more about the Pix2Code approach, please refer to the contents in the `pix2code` directory.


## Contributing

We welcome contributions to enhance Image2Code! Please follow these steps:

1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Make your changes and commit (`git commit -am 'Add new feature'`).
4. Push to the branch (`git push origin feature-branch`).
5. Create a new Pull Request.

## License

This project is licensed under the MIT License. See the `LICENSE` file for more details.

---

We hope Image2Code helps streamline your development process by converting GUI designs into functional code quickly and accurately. 
