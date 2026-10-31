# Brain Tumor Detection from MRI Images using Digital Image Processing and CNN

A Digital Image Processing (DIP) course project that classifies brain MRI images as **Tumor** or **No Tumor** using a Convolutional Neural Network (CNN), with a small Streamlit web app for trying it out.

> **Disclaimer:** This is an educational project. It is **not** a medical device and must not be used for diagnosis or any medical decision.

**Live demo:** https://brain-tumor-detection-6mbapuwampgmf99tnykqht.streamlit.app/

---

## Project Overview

Doctors usually read MRI scans manually, which takes time and can miss small or unclear tumors. This project explores how image processing and deep learning can be combined to build an automatic first-pass classifier.

The project has two parts:

1. **Digital Image Processing pipeline** on a sample MRI scan: grayscale conversion, Gaussian blur (noise removal), histogram equalization (contrast enhancement), thresholding (segmentation) and morphological opening (mask cleanup).
2. **CNN classifier** trained on a labelled MRI dataset to predict Tumor / No Tumor.

All image processing uses built-in OpenCV / NumPy functions and the model uses Keras. No manual pixel loops are written.

## Dataset

- **Name:** Brain MRI Images for Brain Tumor Detection (Kaggle, by Navoneel Chakrabarty)
- **Link:** https://www.kaggle.com/datasets/navoneel/brain-mri-images-for-brain-tumor-detection
- **Size:** 253 images (155 tumor, 98 no tumor)
- **Split used:** 203 training / 50 validation (80/20, seed 42)

The dataset is not included in this repository. Download it from the link above.

## Model

A custom CNN built with TensorFlow / Keras:

- Input: 150 x 150 x 3 images
- Data augmentation: random horizontal flip, rotation, zoom
- Rescaling, then 3 x (Conv2D + MaxPooling2D) blocks with 32, 64, 64 filters
- Flatten, Dense(64, ReLU), Dropout(0.5), Dense(1, sigmoid)
- Loss: binary cross-entropy, optimizer: Adam
- Early stopping on validation loss (best weights restored)

## Results

Evaluated on the 50 held-out validation images:

| Metric | No Tumor | Tumor |
|---|---|---|
| Precision | 0.75 | 0.79 |
| Recall | 0.53 | 0.91 |
| F1-score | 0.62 | 0.85 |
| Support | 17 | 33 |

**Overall accuracy: 78% (39 of 50 images).**

Confusion matrix (rows = actual, columns = predicted):

| | Predicted No Tumor | Predicted Tumor |
|---|---|---|
| **Actual No Tumor** | 9 | 8 |
| **Actual Tumor** | 3 | 30 |

### What I tried

| Experiment | Outcome |
|---|---|
| Basic CNN, no augmentation (15 epochs) | Training accuracy reached about 99% but validation stayed around 78-82%, with validation loss rising: clear overfitting |
| CNN + augmentation + early stopping (final model) | Overfitting reduced, 78% validation accuracy |
| MobileNetV2 transfer learning + class weights | 74% validation accuracy, no improvement over the custom CNN |

### Limitations

- The validation set has only 50 images, so one image changes accuracy by 2 percentage points. These numbers are indicative, not reliable.
- The model leans towards predicting Tumor: it misses only 3 of 33 tumors, but wrongly flags 8 of 17 healthy scans.
- The dataset is small and imbalanced (155 vs 98).
- Only binary classification (tumor type is not predicted).
- Not validated on data from other hospitals or scanners.

## Run the Web App Locally

```bash
git clone https://github.com/tisha-rakholiya/brain-tumor-detection.git
cd brain-tumor-detection
pip install -r requirements.txt
streamlit run app.py
```

Upload any image from the `sample_image/` folder (or your own MRI image) and the app shows the prediction and tumor probability.

## Project Structure

```
brain-tumor-detection/
  app.py                        # Streamlit web app
  brain_tumor_model.keras       # Trained model
  DIP_FINAL_Brain_tumor.ipynb   # Full notebook: DIP pipeline, training, evaluation
  requirements.txt
  sample_image/                 # Sample MRI images for testing the app
  README.md
```

## Tech Stack

Python, OpenCV, NumPy, Matplotlib, TensorFlow / Keras, scikit-learn, Streamlit, Google Colab.

## Future Work

- Train on a larger multi-class dataset (glioma, meningioma, pituitary, no tumor) for real tumor classification
- Use DIP preprocessing (grayscale, blur, histogram equalization) as part of the training pipeline
- Add Grad-CAM heatmaps to show which image regions drive the prediction
- Evaluate with cross-validation and a larger, independent test set

## Author

**Tisha Rakholiya**
B.Tech CSE (AI/DS), 3rd Year
Course project for Digital Image Processing (DIP)
