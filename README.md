# 🚗 Vehicle Counting System (YOLOv8 + Tracking)

This project detects and counts vehicles in a traffic video using YOLOv8 and object tracking. Vehicles are counted **only once** when they cross a predefined line.

## Features
- Vehicle detection (`car`, `truck`, `bus`, `motorcycle`) using YOLOv8
- Unique object tracking with ID assignment to avoid double counting
- Custom counting line support
- Real-time visualization with bounding boxes and counter

## Requirements
- Python
- OpenCV
- cvzone
- ultralytics

## Installation & Run
```bash
pip install ultralytics opencv-python cvzone
python main.py
