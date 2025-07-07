# ECG Heart Attack Prediction

This project is a web application for predicting heart attacks from ECG images using a Convolutional Neural Network (CNN). Built with TensorFlow, Keras, and Streamlit, it allows users to upload ECG images and receive instant predictions about the likelihood of a heart attack.

## Features
- Upload ECG images for automated heart attack prediction
- Deep learning model (CNN) trained on labeled ECG data
- Data preprocessing, model training, and evaluation scripts included
- Visualizations for results and model performance
- User-friendly web interface via Streamlit

## Project Structure
```
.
├── 1_data_preprocessing.py      # Preprocesses raw ECG images
├── 2_cnn_model.py               # Trains the CNN model
├── 3_model_evaluation.py        # Evaluates the trained model
├── 4_run_model.py               # Script for batch predictions
├── app.py                       # Streamlit web app
├── Code/                        # Preprocessed data (.npy files)
├── Models/                      # Trained model files (.h5, .json)
├── ECG Data/                    # Raw ECG image data (not included in repo)
├── results.png                  # Results visualization
├── confusion matrix.png         # Confusion matrix plot
├── TrainingAndValidation.png    # Training/validation curves
└── README.md                    # Project documentation
```

## Setup Instructions
1. **Clone the repository:**
   ```sh
   git clone https://github.com/KetanChavan24/ECG_Aanalysis.git
   cd ECG_Aanalysis
   ```
2. **Install Python 3.12 and create a virtual environment:**
   ```sh
   python3.12 -m venv venv
   source venv/bin/activate
   ```
3. **Install dependencies:**
   ```sh
   pip install tensorflow==2.16.1 scikit-learn pillow matplotlib seaborn streamlit
   ```
4. **Add your ECG data:**
   - Place your raw ECG image folders inside `ECG Data/` as:
     - Abnormal Heartbeat Patients
     - Myocardial Infarction Patients
     - Normal Person
     - Patient that have History of Myocardial Infraction

5. **Run preprocessing and training:**
   ```sh
   python 1_data_preprocessing.py
   python 2_cnn_model.py
   ```

6. **Launch the web app:**
   ```sh
   streamlit run app.py
   ```

## Usage
- Open the local URL provided by Streamlit in your browser.
- Upload an ECG image to get a prediction.

## Notes
- **Do NOT commit large files or sensitive data:**
  - Exclude `ECG Data/`, `Code/`, `Models/`, and `venv/` from version control (add to `.gitignore`).
- The trained model (`Models/heart_attack_cnn_model_final.h5`) is required to run the app.
- For best performance, install the Watchdog module:
  ```sh
  pip install watchdog
  ```

## Repository
[GitHub Repo](https://github.com/KetanChavan24/ECG_Aanalysis.git)

---

*Developed by Ketan Chavan* 