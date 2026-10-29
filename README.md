# ISL Alphabet Recognition

A deep learning project that recognizes the 26 letters (A–Z) of the Indian Sign Language alphabet using MobileNetV2 and real-time webcam input.

## Features
- 26-class alphabet classification
- Two-phase transfer learning and fine-tuning
- Real-time hand detection and gesture recognition
- Confidence thresholding and temporal prediction smoothing
- Hold-to-confirm letter input and text composition

## Tech Stack
Python, TensorFlow/Keras, MobileNetV2, OpenCV, MediaPipe, NumPy.

## Project Structure
```text
proj1/
├── models/
│   ├── isl_model.keras
│   └── hand_landmarker.task
├── src/
│   ├── train.py
│   ├── webcam.py
│   ├── models.py
│   └── evaluate.py
├── reports/
├── requirements.txt
└── README.md
```

## Dataset
**RealSign — Indian Sign Language Dataset**

[Dataset source and download](https://github.com/RealSign62/RealSign-Indian-Sign-Language-Dataset/blob/main/Dataset.zip)

The dataset is not included in this repository because of its size. Download and extract it locally, following the directory structure expected by the code.

## Setup
```bash
git clone <YOUR_REPOSITORY_URL>
cd proj1
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

Run commands from the `src/` directory:

**Train the model**
```bash
python train.py
```

**Evaluate the model**
```bash
python evaluate.py
```

**Start webcam recognition**
```bash
python webcam.py
```

## Evaluation
The project includes a confusion matrix and a baseline classification report.

## Limitations
Recognizes individual alphabet gestures, not complete words or sentences. Predictions may vary with lighting, background, and hand positioning.

## Acknowledgements
Credit to the [RealSign dataset creators](https://github.com/RealSign62/RealSign-Indian-Sign-Language-Dataset). Refer to the original repository for licensing and attribution requirements.
