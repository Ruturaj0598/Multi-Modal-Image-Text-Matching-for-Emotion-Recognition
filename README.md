# Multi-Modal Emotion Recognition Model
=====================================

This repository contains a project on multi-modal deep learning model that combines image features with textual descriptions to predict the dominant emotion conveyed by the image. The project is developed in three parts:

## Data Preparation
-------------------
1. The emotic dataset can be downloaded from [Emotics Dataset](https://forms.gle/wvhComeDHwQPD6TE6). Request the access to the admin and the admin will send it through the email id submitted in the form.
2. Download the PAMI version from the email. DAtaset and annotations both should be downloaded.
3. For converting the dataset into npy files and CSV file use [NPY Generator](mat2py.py)

Your data is ready to be processed.

## Part 1: Baseline Model
------------------------

The baseline model is a multi-modal deep learning model that combines image features with textual descriptions to predict the dominant emotion conveyed by the image. The embedding is separated for the baseline model. The code for this part can be found in the [Part 1 Notebook](Baselinemodel.ipynb).

## Part 2: Fusion via Joint Embedding
----------------------------------

The second part of the project involves fusion via joint embedding in a shared latent space. This approach maps features from different modalities (e.g., image, text, audio) into a common embedding space where their relationships and correlations are preserved. The code for this part can be found in the [Part 2 Notebook](YYYY.ipynb).

## Part 3: Security and Robustness Evaluation
-----------------------------------------

The final part of the project involves evaluating the security and robustness of the multi-modal emotion recognition model based on the above two variants. To study this, we focus on Data Poisoning, specifically the Pixel Attack, which involves modifying a small number of pixels in an image to create an adversarial sample. The code for this part can be found in the [Part 3 Notebook](ZZZZ.ipynb).


### Datasets
---------

The datasets used in this project are:

* EMOTIC Dataset: A multi-modal dataset for emotion recognition in images.

### Contributing
------------
Contributions are welcome! If you have any questions or issues, please open an issue or pull request.

### License
-------
This project is licensed under the MIT License.
