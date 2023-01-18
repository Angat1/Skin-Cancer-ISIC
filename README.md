# International Skin Imaging Collaboration (ISIC) Skin Cancer Classification
The ISIC (International Skin Imaging Collaboration) dataset is a publicly available dataset of skin lesion images that can be used for training and evaluating models for the classification of skin cancer. This dataset contains a large number of images of skin lesions.

The classification of skin cancer is a clinically important task, as early detection and diagnosis of melanoma can greatly improve patient outcomes. However, accurately classifying skin lesions can be challenging for both human dermatologists and traditional machine learning methods due to the wide variability in the appearance of skin lesions.

One way to address this challenge is to use deep learning models, such as CNNs, which have been shown to be highly effective for image classification tasks. CNNs are particularly well-suited for this task because they are able to learn hierarchical representations of the image data, capturing both low-level features such as texture and high-level features such as shape.

In this project, a CNN model would be trained on the ISIC dataset to classify skin lesions . The model would be trained using a small number of images of skin lesions, and then tested on a separate set of images to evaluate its performance. The goal of this project would be to develop a model that can accurately classify skin lesions and potentially improve the diagnostic process for dermatologists.


## Table of Contents
* General Info
* Technologies Used
* Conclusions
* Acknowledgements

<!-- You can include any other section that is pertinent to your problem -->

## General Information

A CNN, or Convolutional Neural Network, is a type of deep learning model that is commonly used for image classification tasks. Building a CNN model on the ISIC dataset for skin cancer would involve several steps:

**Data preparation**: The ISIC dataset would need to be loaded and preprocessed, which could include tasks such as resizing images, normalizing pixel values, and splitting the data into training, validation, and test sets.

**Model architecture**: The architecture of the CNN would need to be defined, which could include layers such as convolutional layers, pooling layers, and fully connected layers. The number of layers and the number of filters in each layer would need to be determined based on the complexity of the task and the size of the dataset.

**Model training**: The model would then be trained using the training data, with the goal of minimizing the error between the predicted class labels and the true class labels. This could be done using an optimization algorithm such as stochastic gradient descent (SGD) or Adam.

**Model evaluation**: After training, the model would be evaluated on the validation and test sets to measure its performance. Common metrics used for image classification tasks include accuracy, precision, recall, and F1 score.

**Business Problem**

The business problem that is being addressed by solving the ISIC skin cancer dataset classification using a CNN is the early detection and diagnosis of skin cancer, specifically melanoma. Melanoma is a type of skin cancer that can be highly aggressive and has a high mortality rate if not caught early. Accurately classifying skin lesions is essential for early detection and diagnosis of Skin Cancer.

Currently, the process of identifying skin cancer is often done by human dermatologists, who may have difficulty accurately identifying certain types of skin lesions, particularly in their early stages. This can lead to delayed diagnosis and treatment, which can negatively impact patient outcomes. By developing a model that can accurately classify skin lesions, the diagnostic process for dermatologists could be improved and the rate of early detection and diagnosis  could be increased.

Overall, the goal of this project is to use the power of deep learning to improve the diagnostic process for skin cancer and ultimately save lives by detecting and diagnosing at an early stage.

**Dataset**

The ISIC dataset is a publicly available dataset of skin lesion images that can be used for training and evaluating models for the classification of skin cancer. THis dataset content test and train data sepratly. It has 9 different types of skin cancer.The ISIC dataset is widely used in the research community for developing and evaluating models for the classification of skin cancer. The dataset is particularly useful for training and evaluating deep learning models such as CNNs, as it contains a large number of diverse images of skin lesions.

<!-- You don't have to answer all the questions - just the ones relevant to your project. -->

## Conclusions
- Dataset is imbalanced with respect to the number of images per class. The number of images per class varies greatly, with the class "melanoma" having the most images at 454 and the class "seborrheic keratosis" having the least at 77 images.This class imbalance can cause problems for machine learning models, as they can lead to a bias towards the majority class. This can result in poor performance on the minority class, which can have a significant impact on the overall performance of the model, particularly when trying to detect and diagnose the melanoma, which is the most aggressive form of skin cancer.

- Running a model for only 20 epochs may not be enough to fully train the model and allow it to converge to a good solution. This  lead to underfitting, where the model is not able to accurately capture the underlying patterns in the data. This result in poor performance on the test set and the model  not generalize well to new unseen data.

- The data generator applies a series of random transformations to the images, such as rescaling, rotation, shifting, zooming, flipping and changing brightness. Using data augmentation can help to balance the dataset by creating more examples for the minority classes. By applying these random transformations to the images, the model will be exposed to a wider variety of images during training, which can help to reduce overfitting and improve generalization. 

- Even 30 epochs may not be sufficient to fully train a deep learning model such as a CNN on the ISIC skin cancer dataset classification problem. 

<!-- You don't have to answer all the questions - just the ones relevant to your project. -->


## Technologies Used
- Python - version 3.11. 0
- Panda - version 1.5.1
- Numpy - version 1.24.1
- Matplotlib - version 3.6.3
- Keras - version 2.11.0

<!-- As the libraries versions keep on changing, it is recommended to mention the version of library used in this project -->

## Acknowledgements
- This project was inspired by Upgrad
- https://www.isic-archive.com/#!/topWithHeader/wideContentTop/main


## Contact
Created by [@Angat1] - feel free to contact me!


<!-- Optional -->
<!-- ## License -->
<!-- This project is open source and available under the [... License](). -->

<!-- You don't have to include all sections - just the one's relevant to your project -->
