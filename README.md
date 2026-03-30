# Australian-Rain-Prediction-App
# 🌧️ Australian Rain Prediction: End-to-End Deep Learning Pipeline

An end-to-end Machine Learning project that predicts whether it will rain tomorrow in various Australian cities. This project features a robust Scikit-Learn preprocessing pipeline, a TensorFlow/Keras neural network, and an interactive Streamlit web dashboard.

## 🚀 Project Overview

The goal of this project is to accurately predict the `RainTomorrow` target variable using historical weather data (`weatherAUS.csv`). 

Moving beyond a simple tutorial, this project focuses on **production-ready data science practices**, specifically addressing common pitfalls like data leakage, categorical encoding errors, and statistical model bias in minority classes.

## ✨ Key Features & Methodology

* **Robust Preprocessing Pipeline:** Built a sequential `ColumnTransformer` and `Pipeline` architecture to handle missing values and encoding.
  * *Numerical Data:* Imputed using the Median (robust to weather outliers) and scaled using `StandardScaler`.
  * *Categorical Data:* Imputed using the Mode and processed via `OneHotEncoder` to prevent nominal data from being misinterpreted mathematically by the neural network.
* **Zero Data Leakage:** Ensured `train_test_split` occurred *before* any imputation or scaling to maintain the integrity of the test set as a true real-world simulation.
* **Deep Learning Model:** Built a Sequential Neural Network using **TensorFlow/Keras** with Dense layers, ReLU activations, and a final Sigmoid layer for binary classification. 
* **Bias Detection & Mitigation:** Conducted Subgroup Analysis to test for geographic bias. Identified the "Accuracy vs. Recall" trap in desert locations (like Uluru) and successfully mitigated it to balance the model's ability to catch minority-class events (storms).
* **Interactive Deployment:** Packaged the model (`.h5`) and the preprocessing pipeline (`.pkl`) into a sleek, user-friendly **Streamlit** web application.

## 🛠️ Tech Stack

* **Data Manipulation:** `pandas`, `numpy`
* **Machine Learning (Preprocessing):** `scikit-learn`
* **Deep Learning:** `tensorflow`, `keras`
* **Web Deployment:** `streamlit`

## 📂 Project Structure

* `experiment.ipynb`: The core Jupyter Notebook containing exploratory data analysis, pipeline construction, model training, and bias testing.
* `app.py`: The Streamlit application script for the interactive front-end.
* `newgen.h5`: The trained TensorFlow/Keras neural network model.
* `preprocessor.pkl`: The saved Scikit-Learn pipeline (contains imputers, scalers, and encoders).
* `test_bias.py`: A dedicated script to evaluate geographic representation bias and recall metrics.

## 💻 How to Run Locally

**1. Clone the repository and install dependencies:**
Ensure you have Python installed, then install the required packages:
```bash
pip install pandas numpy scikit-learn tensorflow streamlit
2. Run the Streamlit App:
Navigate to the project directory in your terminal and run:

Bash
streamlit run app.py
The application will open automatically in your default web browser.

📈 Future Improvements
Implement hyperparameter tuning (e.g., GridSearch or Keras Tuner) to optimize the neural network architecture.

Connect to a live weather API to make real-time predictions for tomorrow rather than relying on manual user input.
