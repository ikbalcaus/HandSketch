# Hand Drawing Canvas

This project implements a character detection system using Python. It utilizes OpenCV, Mediapipe, PyTorch and Tkinter for image processing and model inference.

## Libraries Used

- **OpenCV**: For image processing and handling canvas interactions
- **Mediapipe**: For real-time detection of hand landmarks and gestures
- **PyTorch**: For character recognition and machine learning model inference
- **Tkinter**: For GUI

## Demonstration

![](demonstration/main_screen.gif)
![](demonstration/detect_screen.png)

## Installation

1. Install [Python 3.10.0](https://www.python.org/downloads/release/python-3100/). Ensure that the "**Add python.exe to PATH**" option is selected during installation. 
2. Open terminal in a project directory and run following commands:
```bash
    python -m venv .venv
    .venv\Scripts\activate
    pip install -r requirements.txt
```

## Usage

To start program run a following command inside a project directory:
```bash
    python main.py 
```

## Gestures

Hand Drawing Canvas recognizes the following hand gestures:

- **Writing Gesture**:  
  Extend the point finger and the close thumb

- **Erasing Gesture**:  
  Extend all fingers

- **Approve Gesture**:  
  Perform a like gesture

The gestures are detected in real-time, with visual feedback provided on the screen via circles drawn at the key landmarks, along with labels indicating the detected gesture.  

![](landmarks.png)  

## Manualy Config Data

### Datasets
Download any dataset at: [kaggle.com/datasets](https://www.kaggle.com/datasets)  
Organize your dataset by creating a folder for each character and placing the corresponding images inside. For more details on dataset structure, refer to "**dataset**" > "**readme.md**".

### Training
To train model run:
```bash
    python train.py
```

## Manualy Test Program

### Detection
To test character detection with saved images, place your images inside "**images**" folder and run:
```bash
    python detect.py
```

### Video stream with hand recognition
To try hand recognition through video streaming run:
```bash
    python video_stream.py
```

## Support

Star this repository :star:  
Follow me on [Github](https://github.com/ikbalcaus) and [Linkedin](https://www.linkedin.com/in/ikbalcaus/)  
Share this project with other people
