# Multi-Modal Emotion Recognition Model 🖼️📝
=====================================

This repository contains a project on multi-modal deep learning models that combine image features 🖼️ with textual descriptions 📝 to predict the dominant emotion conveyed by images. The project is developed and evaluated on two datasets: Emotic (context-based emotion recognition) and Flickr8k (image-text retrieval).

## 📝 Project Attribution
-----------------------

This project was developed as an additional research initiative by the **IDACUS group from Furtwangen University**. This work was conducted as an independent project without any formal requirements from the university side.

## 📦 Data Preparation
-------------------

### Emotic dataset 🎭
1. The emotic dataset can be downloaded from [Emotics Dataset](https://forms.gle/wvhComeDHwQPD6TE6). Request the access to the admin and the admin will send it through the email id submitted in the form.
2. Download the PAMI version from the email. Dataset and annotations both should be downloaded.
3. For converting the dataset into npy files and CSV file use [NPY Generator](mat2py.py) [1]

### Flickr8k dataset 📸
1. Download from [Flickr8k Dataset](https://www.kaggle.com/datasets/adityajn105/flickr8k)
2. For converting the dataset into npy files and CSV file use [NPY Generator_Flickr](prepare_flickr8k.py)

Your data is ready to be processed. The system includes three main components:

## Part 1: Baseline Model 🏗️
------------------------

The baseline model is a multi-modal deep learning model that combines image features with textual descriptions to predict the dominant emotion conveyed by the image. The embedding is separated for the baseline model. <br> **Emotic Implementation**: [Baseline_model_Emotic](Baseline_model.ipynb)
<br> **Flickr8k Implementation**: [Baseline_model_Flickr](Baseline_model_Flickr.ipynb)

## Part 2: Fusion via Joint Embedding 🤝
----------------------------------

The second part of the project involves fusion via joint embedding in a shared latent space. This approach maps features from different modalities (e.g., image, text, audio) into a common embedding space where their relationships and correlations are preserved. 
<br> **Emotic Implementation**: [Joint_model_Emotic](Joint_embedding.ipynb)
<br> **Flickr8k Implementation**: [Joint_model_Flickr](Joint_embedding_Flickr.ipynb)

## Part 3: Security and Robustness Evaluation 🔒
-----------------------------------------

The final part of the project involves evaluating the security and robustness of the multi-modal emotion recognition model based on the above two variants. To study this, we focus on Data Poisoning, specifically the Pixel Attack, which involves modifying a small number of pixels in an image to create an adversarial sample.
<br> **Emotic Implementation**: [Evaluation_Emotic](Robustness_evaluation.ipynb)
<br> **Flickr8k Implementation**: [Evaluation_Flickr](Robustness_evaluation_Flickr.ipynb)

### Datasets 📊
---------

The datasets used in this project are:

| Dataset | Type | Samples | Modalities | Task |
|---------|------|---------|------------|------|
| **Emotic** | Emotion recognition | 23,571 | Image + Text | Emotion classification |
| **Flickr8k** | Image captioning | 8,000 | Image + Text | Cross-modal retrieval |

### Contributing ✨
------------
Contributions are welcome! If you have any questions or issues, please open an issue or pull request.

### Citation 📋
-------------
If you use this repository in your research or projects, please cite it as:

@misc{multimodal_emotion_recognition_2025,
author = {IDACUS Group, Hochschule Furtwangen University},
title = {Multi-Modal Image-Text Matching for Emotion Recognition},
year = {2025},
publisher = {GitHub},
journal = {GitHub repository},
howpublished = {\url{https://github.com/Ruturaj0598/Multi-Modal-Image-Text-Matching-for-Emotion-Recognition}},
note = {Additional research project - no university requirements}
}

### License 📜
-------
This project is licensed under the MIT License.

### References 📚
-------------------
@misc{tandon2020emotic,
  author = {Tandon, Abhishek},
  title = {Emotic: Context Based Emotion Recognition},
  year = {2020},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/Tandon-A/emotic/}},
}

@article{kosti2020context,
  title={Context based emotion recognition using emotic dataset},
  author={Kosti, Ronak and Alvarez, Jose M and Recasens, Adria and Lapedriza, Agata},
  journal={arXiv preprint arXiv:2003.13401},
  year={2020}
}

***Happy coding! 🚀***
<br> Let’s build robust, multi-modal emotion recognition systems together!
