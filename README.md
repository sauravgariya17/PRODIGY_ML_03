# Prodigy InfoTech Task-03 — Dogs vs Cats Image Classification using SVM

This project implements a **Support Vector Machine (SVM)** to classify images of cats and dogs using the Kaggle Dogs vs Cats competition dataset.

## Task

> Implement a support vector machine (SVM) to classify images of cats and dogs from the Kaggle dataset.

Dataset:
https://www.kaggle.com/competitions/dogs-vs-cats/data

The original competition dataset contains 25,000 labelled training images (12,500 cats and 12,500 dogs) and an unlabeled test set. The training images are the ones used by this project.

## Approach

1. Download the official competition data with the Kaggle API.
2. Extract `train.zip`.
3. Parse labels from filenames such as `cat.123.jpg` and `dog.456.jpg`.
4. Resize images to 64×64.
5. Convert images to grayscale.
6. Extract **HOG (Histogram of Oriented Gradients)** features.
7. Standardize the features.
8. Train an **RBF-kernel SVM**.
9. Evaluate with accuracy, precision, recall, F1-score and confusion matrix.
10. Save the trained model and preprocessing pipeline.
11. Use the saved model to predict a new image.
12. Optionally run a small Flask web app for image upload and prediction.

HOG is used because directly feeding every RGB pixel into an SVM creates a very large feature space. HOG keeps useful shape/edge information while making the SVM practical on a normal laptop.

## Project structure

```text
dogs_vs_cats_svm_task03/
│
├── app.py
├── config.py
├── download_data.py
├── train.py
├── predict.py
├── requirements.txt
├── .gitignore
├── README.md
│
├── templates/
│   └── index.html
│
├── static/
│   └── uploads/
│
├── data/
│   ├── raw/              # Kaggle zip files go here; not included in Git
│   └── processed/
│
├── models/
│   └── svm_pipeline.joblib       # generated after training
│
├── reports/
│   └── training_report.json      # generated after training
│
└── notebooks/
    └── dogs_vs_cats_svm.ipynb
```

## 1. Install Python

Python 3.10+ is recommended.

Create a virtual environment:

### Windows

```powershell
python -m venv venv
venv\Scripts\activate
```

Install dependencies:

```powershell
pip install -r requirements.txt
```

## 2. Configure Kaggle

You need a Kaggle account and access to the competition.

The easiest method is to install the official Kaggle CLI and authenticate it.

```powershell
pip install kaggle
```

Create/download your Kaggle API credentials from your Kaggle account's API settings and configure them according to Kaggle's current instructions.

Then verify:

```powershell
kaggle --help
```

You may also need to open the competition page once and accept its rules before downloading competition files.

## 3. Download the dataset

Run:

```powershell
python download_data.py
```

This downloads the competition files and extracts the labelled `train` images.

You should eventually have:

```text
data/raw/train/
    cat.0.jpg
    cat.1.jpg
    ...
    dog.0.jpg
    dog.1.jpg
    ...
```

**Do not put the dataset inside the GitHub repository.** It is large. The `.gitignore` in this project excludes it.

## 4. Train the SVM

For a normal laptop, start with 2,500 images per class:

```powershell
python train.py --max-per-class 2500
```

For a stronger experiment, use more images:

```powershell
python train.py --max-per-class 5000
```

To use all available labelled images:

```powershell
python train.py --max-per-class 12500
```

The last option can take considerably longer and needs more RAM/CPU time.

The script creates:

```text
models/svm_pipeline.joblib
reports/training_report.json
reports/confusion_matrix.png
reports/sample_predictions.png
```

## 5. Predict a new image

After training:

```powershell
python predict.py --image path\to\your\cat_or_dog.jpg
```

Example:

```powershell
python predict.py --image sample.jpg
```

The output will look like:

```text
Prediction: cat
Confidence score: 0.82
```

The score is based on the SVM decision function and is not a calibrated probability.

## 6. Run the web application

After the model has been trained:

```powershell
python app.py
```

Open:

```text
http://127.0.0.1:5000
```

Upload a cat or dog image and the application will display the prediction.

## 7. Jupyter Notebook

The same workflow is available in:

```text
notebooks/dogs_vs_cats_svm.ipynb
```

Start Jupyter with:

```powershell
jupyter notebook
```

Then open the notebook.

## Important notes

- The dataset itself is intentionally **not included** in this ZIP because the original competition dataset is large.
- The trained model is also generated locally after training rather than bundled into the ZIP.
- The code is configured to use a manageable subset first so you can successfully run the task on a normal computer.
- For your internship submission, include screenshots of:
  - dataset loading
  - sample images
  - training output
  - classification report
  - confusion matrix
  - prediction on a new image
  - Flask application result

## Technologies

- Python
- NumPy
- OpenCV
- scikit-image
- scikit-learn
- Matplotlib
- Seaborn
- Joblib
- Flask
- Jupyter Notebook
- Kaggle API

## Expected result

Because SVM performance depends on the number of images, preprocessing, hardware and train/validation split, the exact accuracy is produced by your run. HOG + RBF SVM should give a meaningful classical computer-vision baseline without using a CNN.

## GitHub

Before pushing:

```powershell
git init
git add .
git commit -m "Prodigy InfoTech Task 03 - Dogs vs Cats SVM"
git branch -M main
git remote add origin YOUR_GITHUB_REPOSITORY_URL
git push -u origin main
```
