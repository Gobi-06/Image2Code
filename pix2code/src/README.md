# Pix2Code

Transforming a graphical user interface screenshot created by a designer into computer code is a typical task conducted by a developer in order to build customized software, websites, and mobile applications.
In this paper, we show that deep learning methods can be leveraged to train a model end-to-end to automatically generate code from a single input image with over **77% of accuracy** for three different platforms (i.e. **iOS, Android and Web**).


## Table of Contents
- [Pix2Code Approach](#pix2code-approach)
- [Dataset](#dataset)
- [Model Training](#usage)
- [Model Testing](#models-and-training)
- [Contributing](#contributing)
- [License](#license)



## Pix2Code Approach

A novel approach based on **Convolutional and Recurrent Neural Networks** allowing the generation of computer tokens from a single GUI screenshot as input   
It **requires no engineered feature extraction pipeline** nor expert heuristics was designed to process the input data   
The model learns from the pixel values of the input image alone.   
Our experiments demonstrate the effectiveness of our method for generating computer code for various platforms (i.e. **iOS** and **Android** native mobile interfaces, and multi-platform **web**-based HTML/CSS interfaces) without the need for any change or specific tuning to the model.
