# Real-Time-Motion-Detector
An AI-based real-time motion detection system using Python and OpenCV
# 🎥 Real-Time Motion Detection with Sound Alert
This is a real-time motion detection system using Python and OpenCV. It detects movement from the webcam and plays a beep sound when motion is detected.

## 🔧 Technologies Used
- Python 3.12
- OpenCV
- imutils
- winsound (for sound alert on Windows)

## 📌 Features
- Real-time webcam feed
- Motion detection using frame differencing and Gaussian blur
- Contour detection to identify moving areas
- Green rectangle drawn around moving objects
- System beep alert (`winsound.Beep`) when motion is detected
- Press `q` to cleanly exit and release the webcam
  
## 📁 Project Structure

FaceMaskDetection/
├── mask_detector.py # Main Python script
├── Real_Time_Motion_Detection_Kirti_PPT.pptx # Presentation file for report
├── README.md # (This file)

## ▶️ How to Run
1. Install required libraries:
 pip install opencv-python imutils

2. Run the script:
   python mask_detector.py
   
3. Usage
- Move in front of the camera → green box will appear

- A beep sound will play when motion is detected

- Press q to exit the app

## 🧠 Project Use
AI/ML internships

Home surveillance prototype

Smart classroom monitoring

Motion-based alert systems


## 👩‍💻 Developed by
**Kirti Kumari**

Computer Science and Engineering Student (Pre-final Year)

Sathyabama Institute of Science & Technology, Chennai
