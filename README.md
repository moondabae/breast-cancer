# Breast Cancer Classification with VGG16

This project involves the use of a deep convolutional neural network (CNN) model based on VGG16 for the classification of breast cancer. The model extracts high-level features from input images to accurately diagnose and classify the stages of breast cancer.

## Table of Contents
- [Description](#description)
- [Key Features](#key-features)
- [Technologies Used](#technologies-used)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Contributors](#contributors)
- [License](#license)

## Description
Breast cancer remains one of the most severe public health problems. Early diagnosis is vital to avoid the development of malignancy. This project proposes a deep convolutional neural network (CNN) model based on VGG16 for breast cancer classification. The model architecture is utilized to extract high-level features from the input images. The experimental results showed that the proposed model outperformed various techniques in terms of accuracy, loss, sensitivity, and specificity.

## Key Features
- Utilizes the VGG16 architecture for feature extraction
- Data augmentation techniques to improve model performance
- Evaluation metrics including accuracy, loss, sensitivity, and specificity

## Technologies Used
- TensorFlow and Keras for model development
- Python for data processing and analysis
- Matplotlib for data visualization

## Installation
1. Clone the repository:
    ```bash
    git clone https://github.com/moondabae/breast-cancer-classification.git
    cd breast-cancer-classification
    ```
2. Create and activate a virtual environment:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```
3. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

## Usage
1. Ensure the dataset is placed in the correct directory:
    ```
    /content/breast_class
    ├── train
    │   ├── Grade 1
    │   ├── Grade 2
    │   └── Grade 3
    └── validation
        ├── Grade 1
        ├── Grade 2
        └── Grade 3
    ```
2. Run the Jupyter notebook to train the model:
    ```bash
    jupyter notebook Finalize_AI_Project.ipynb
    ```
3. Follow the steps in the notebook to preprocess the data, train the model, and evaluate the results.

## Results
The model achieved significant accuracy in classifying the different grades of breast cancer. The use of data augmentation and fine-tuning of the VGG16 model parameters contributed to the improved performance metrics.

## Contributors
- Muna Nazihah Binti Mohamad Asman
- Fahda Nadia Bt Md Shahizan
- Saimon Mah Wei Yung
- Malarkodi A/P Panjawarnam

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
