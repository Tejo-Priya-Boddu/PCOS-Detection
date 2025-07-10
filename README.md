# 🩺 PCOS Detection using Deep Learning

This project focuses on the automated detection of **Polycystic Ovary Syndrome (PCOS)** using deep learning techniques. It leverages medical imaging data and convolutional neural networks (CNNs) to classify images as indicative of PCOS or not, providing an assistive tool for early diagnosis.

## 📌 Problem Statement

PCOS is a common hormonal disorder affecting women of reproductive age. Early and accurate diagnosis is crucial but can be challenging due to overlapping symptoms. This project aims to build a reliable classification model using ultrasound images to assist healthcare professionals in the diagnostic process.

---

## 🧠 Approach

- **Data Collection**: A dataset of labeled ovarian ultrasound images was used. Images are categorized into PCOS and non-PCOS classes.
- **Preprocessing**: 
  - Image resizing and normalization
  - Data augmentation to prevent overfitting
- **Model Architecture**:
  - CNN-based model built using TensorFlow/Keras
  - Experimented with architectures like VGG16, ResNet50 for better accuracy
- **Training**:
  - Train/Validation/Test split
  - Used techniques like dropout, batch normalization
- **Evaluation**:
  - Accuracy, Precision, Recall, F1-score
  - Confusion matrix and ROC curve

---

## 🛠️ Technologies Used

- Python
- TensorFlow / Keras
- NumPy, OpenCV, Matplotlib
- VS Code 
- Scikit-learn
- StreamLit

---

Detected Output :
![detected output](https://github.com/user-attachments/assets/371cc686-e8e2-484f-916f-2140969cc71a)

No PCOS Detected:
![no pcos detected](https://github.com/user-attachments/assets/75b1afa9-3fbb-478f-92aa-65682b2ed3c5)







