# Music-Emotion-Recognition-from-EEG-signals---Capstone

Music Emotion Recognition from EEG signals - Capstone project

EEG + Song Model Demo
=====================

This Flask demo serves two pre-trained models and lets you upload an EEG `.npy` and a song `.wav` to get side-by-side predictions.

Setup
-----

1. Create and activate a Python environment (recommended):

   python -m venv venv
   source venv/bin/activate

2. Install dependencies:

   pip install -r requirements.txt

3. Ensure the following files are in the project root:

   - GESSNet_Opensmile_3.pkl
   - GESSNet_Yamnet_4.pkl
   - OpenSMILE_features.csv
   - YAMNet_features.csv

Usage
-----

Run the app:

   python app.py

Open http://localhost:5002 in your browser.

Notes and assumptions
---------------------
- The song filename (without the .wav extension) must match the index/song id used in the respective CSV files. For example, a file named "6Qd... .wav" should have index "6Qd..." in the CSV.
- Models are loaded with pickle or torch.load. If your model requires a custom class, ensure it's importable from the Python path or replace loading logic accordingly.
- This demo normalizes song features using StandardScaler and drops very low-variance columns (variance <= 0.01). This mirrors preprocessing in your snippet.

