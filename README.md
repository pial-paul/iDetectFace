# iDetectFace - Face Recognition Attendance System

A Python-based facial recognition system that automatically takes attendance and stores records in a CSV file. This application uses computer vision to identify registered individuals and mark their attendance with timestamps.

## 📋 Features

- **Automated Attendance**: Recognize faces and mark attendance automatically
- **User-Friendly Interface**: Simple menu-driven console interface
- **Real-Time Face Detection**: Visual feedback with bounding boxes and confidence scores
- **CSV Storage**: Attendance records stored in standard CSV format for easy export
- **One-Time Daily Attendance**: Each person is only marked once per day
- **Visual Feedback**: Shows recognition status and confidence percentages
- **Face Registration**: Simple process to add new faces to the system

## 🖼️ Screenshots

(Will be added)

## 🔧 Prerequisites

- Python 3.6 or higher
- Webcam or built-in camera
- The following Python packages:
  - OpenCV
  - face_recognition
  - NumPy

## ⚙️ Installation

1. Clone this repository:
   ```
   git clone https://github.com/pial-paul/iDetectFace.git
   cd iDetectFace
   ```

2. Install required dependencies:
   ```
   pip install opencv-python face_recognition numpy
   ```

   **Note**: The `face_recognition` library requires `dlib`, which might need additional setup:
   ```
   pip install cmake
   pip install dlib
   ```

   On Windows, you might need to install Visual C++ Build Tools.

## 🚀 Usage

1. **Run the program**:
   ```
   python face_attendance.py
   ```

2. **Main Menu Options**:
   - **Start Face Recognition Attendance**: Begin the recognition process
   - **Add New Face**: Register a new person to the system
   - **View Today's Attendance**: Check who's been marked present today
   - **Exit**: Close the application

### Adding a New Face

1. Select option 2 from the main menu
2. Enter the person's name
3. Position your face in front of the camera
4. Wait for the green box and "Face Detected" message
5. Press SPACE to capture the face
6. Press ESC to cancel

### Taking Attendance

1. Select option 1 from the main menu
2. The system will automatically detect and recognize faces
3. When a registered person is detected, their name will be displayed with a confidence percentage
4. Attendance is automatically marked (indicated by a ✓ symbol)
5. Press 'q' to quit and return to the main menu

## 📁 File Structure

- `face_attendance.py`: Main application file
- `known_faces/`: Directory containing saved face images
- `attendance.csv`: CSV file storing attendance records

## 🔍 How It Works

1. **Face Detection**: The system uses the `face_recognition` library to detect faces in camera frames
2. **Face Encoding**: Each detected face is converted to a mathematical representation (encoding)
3. **Face Comparison**: New face encodings are compared against known faces
4. **Attendance Marking**: When a match is found, attendance is recorded with timestamp
5. **Data Storage**: All records are saved to a CSV file for easy access and analysis

## ⚠️ Limitations

- Requires good lighting conditions for optimal recognition
- Performance depends on your computer's processing power
- May struggle with side profiles or partially occluded faces
- Recognition accuracy can be affected by facial hair, glasses, or other changes in appearance

## 🛠️ Troubleshooting

- **Camera not working**: Ensure no other application is using your camera
- **Slow performance**: Lower the camera resolution in the code or increase the `process_every_n_frames` value
- **Recognition issues**: Try re-registering the face under better lighting conditions
- **Installation problems**: Check if all dependencies are correctly installed

## 🔜 Future Improvements

- GUI interface
- Database integration
- Multiple camera support
- Attendance reports and analytics
- Face liveness detection

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 👥 Contributions

Contributions, issues, and feature requests are welcome! Feel free to check [issues page](https://github.com/yourusername/iDetectFace/issues).

## 🙏 Acknowledgements

- [face_recognition](https://github.com/ageitgey/face_recognition) library
- [OpenCV](https://opencv.org/) library
