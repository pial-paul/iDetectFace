import os
import cv2
import numpy as np
import face_recognition
import csv
from datetime import datetime

class FaceAttendanceSystem:
    def __init__(self):
        # Create directories if they don't exist
        self.known_faces_dir = "known_faces"
        self.attendance_file = "attendance.csv"
        
        if not os.path.exists(self.known_faces_dir):
            os.makedirs(self.known_faces_dir)
            print(f"Created directory: {self.known_faces_dir}")
        
        # Initialize attendance file if it doesn't exist
        if not os.path.exists(self.attendance_file):
            with open(self.attendance_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["Name", "Date", "Time"])
                print(f"Created attendance file: {self.attendance_file}")
        
        # Load known faces
        self.known_faces = []
        self.known_names = []
        self.load_known_faces()
        
        # Camera settings for smoother operation
        self.camera_width = 640
        self.camera_height = 480
        self.process_every_n_frames = 2  # Process every other frame for smoother operation
        self.frame_count = 0
    
    def load_known_faces(self):
        """Load faces from the known_faces directory"""
        print("Loading known faces...")
        self.known_faces = []
        self.known_names = []
        
        # Ensure the directory exists and has files
        if not os.path.exists(self.known_faces_dir) or len(os.listdir(self.known_faces_dir)) == 0:
            print("No faces found. Please add faces first.")
            return
        
        # Process each image in the directory
        for filename in os.listdir(self.known_faces_dir):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                # Get the person's name from the filename
                name = os.path.splitext(filename)[0].replace("_", " ")
                
                # Full path to the image file
                img_path = os.path.join(self.known_faces_dir, filename)
                
                try:
                    # Load the image
                    image = face_recognition.load_image_file(img_path)
                    
                    # Get face encoding - use the first face found
                    encodings = face_recognition.face_encodings(image)
                    
                    if len(encodings) > 0:
                        self.known_faces.append(encodings[0])
                        self.known_names.append(name)
                        print(f"Loaded: {name}")
                    else:
                        print(f"Warning: No face found in {filename}")
                except Exception as e:
                    print(f"Error loading {filename}: {e}")
        
        print(f"Loaded {len(self.known_names)} faces")
    
    def mark_attendance(self, name):
        """Record attendance in the CSV file"""
        # Get the current date and time
        now = datetime.now()
        date_str = now.strftime("%Y-%m-%d")
        time_str = now.strftime("%H:%M:%S")
        
        # Check if person already marked today
        already_marked = False
        if os.path.exists(self.attendance_file):
            with open(self.attendance_file, 'r') as f:
                reader = csv.reader(f)
                next(reader, None)  # Skip header
                for row in reader:
                    if len(row) >= 2 and row[0] == name and row[1] == date_str:
                        already_marked = True
                        break
        
        # If not already marked, add to the attendance file
        if not already_marked:
            with open(self.attendance_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([name, date_str, time_str])
                print(f"Marked attendance for {name} on {date_str} at {time_str}")
            return True
        return False
    
    def setup_camera(self):
        """Initialize the camera with proper settings"""
        cam = cv2.VideoCapture(0)
        if not cam.isOpened():
            print("Error: Could not open camera.")
            return None
            
        # Set camera properties for better performance
        cam.set(cv2.CAP_PROP_FRAME_WIDTH, self.camera_width)
        cam.set(cv2.CAP_PROP_FRAME_HEIGHT, self.camera_height)
        cam.set(cv2.CAP_PROP_FPS, 30)  # Try to get 30fps if possible
        
        # Give camera time to warm up
        for _ in range(10):
            cam.read()
            
        return cam
    
    def add_face(self):
        """Capture and add a new face"""
        name = input("Enter the person's name: ").strip()
        if not name:
            print("Name cannot be empty.")
            return
        
        # Create a filename from the name
        filename = name.lower().replace(" ", "_") + ".jpg"
        output_path = os.path.join(self.known_faces_dir, filename)
        
        print("\nStarting camera to capture face.")
        print("Position face in the frame and press SPACE to capture.")
        print("Press ESC to cancel.")
        
        # Initialize the webcam
        cam = self.setup_camera()
        if cam is None:
            return
            
        face_detected = False
        
        while True:
            ret, frame = cam.read()
            if not ret:
                print("Failed to grab frame.")
                break
                
            # Create a copy for drawing
            display_frame = frame.copy()
            
            # Convert to RGB for face detection
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Detect faces
            face_locations = face_recognition.face_locations(rgb_frame)
            
            # Draw box around face and show status
            if face_locations:
                face_detected = True
                for top, right, bottom, left in face_locations:
                    # Draw rectangle around face
                    cv2.rectangle(display_frame, (left, top), (right, bottom), (0, 255, 0), 2)
                    
                    # Draw face landmarks if possible
                    try:
                        face_landmarks = face_recognition.face_landmarks(rgb_frame, [face_locations[0]])[0]
                        for feature in face_landmarks.values():
                            for point in feature:
                                cv2.circle(display_frame, point, 2, (0, 0, 255), -1)
                    except:
                        pass
                        
                # Show ready status
                cv2.putText(display_frame, "Face Detected - Ready to Capture", 
                           (10, 30), cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 255, 0), 1)
            else:
                face_detected = False
                cv2.putText(display_frame, "No Face Detected", 
                           (10, 30), cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 0, 255), 1)
            
            # Add instructions on screen
            cv2.putText(display_frame, "SPACE: Capture  ESC: Cancel", 
                       (10, display_frame.shape[0] - 20), cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1)
            
            # Show the frame
            cv2.imshow("Add New Face", display_frame)
            
            # Wait for key press
            key = cv2.waitKey(1) & 0xFF
            
            # ESC key - cancel
            if key == 27:
                print("Cancelled face capture.")
                break
            
            # SPACE key - take picture
            elif key == 32:
                if face_detected:
                    # Save the image
                    cv2.imwrite(output_path, frame)
                    print(f"Face captured and saved as {filename}")
                    break
                else:
                    print("No face detected. Please position your face in the frame.")
        
        # Release resources
        cam.release()
        cv2.destroyAllWindows()
        
        # Reload known faces
        self.load_known_faces()
    
    def view_attendance(self):
        """Display attendance records with proper date filtering"""
        print("\nAttendance Records View Options:")
        print("1. View Today's Attendance")
        print("2. View Attendance by Date")
        print("3. View All Attendance Records")
        
        choice = input("\nEnter your choice (1-3): ").strip()
        
        if choice == '1':
            # View today's attendance
            self.view_attendance_by_date(datetime.now().strftime("%Y-%m-%d"))
        elif choice == '2':
            # View attendance by specific date
            date_input = input("Enter date (YYYY-MM-DD): ").strip()
            try:
                # Validate date format
                datetime.strptime(date_input, "%Y-%m-%d")
                self.view_attendance_by_date(date_input)
            except ValueError:
                print("Invalid date format. Please use YYYY-MM-DD.")
        elif choice == '3':
            # View all records
            self.view_all_attendance()
        else:
            print("Invalid choice.")
    
    def view_attendance_by_date(self, date_str):
        """Display attendance for a specific date"""
        attendance_records = []
        
        try:
            with open(self.attendance_file, 'r') as f:
                reader = csv.reader(f)
                headers = next(reader)  # Skip header row
                
                for row in reader:
                    if len(row) >= 2 and row[1] == date_str:
                        attendance_records.append(row)
            
            if attendance_records:
                print(f"\n--- Attendance for {date_str} ---")
                print("Name                  Time")
                print("-" * 30)
                for record in attendance_records:
                    print(f"{record[0]:<20} {record[2]}")
                print(f"Total: {len(attendance_records)} students")
            else:
                print(f"No attendance records for {date_str}")
        except Exception as e:
            print(f"Error reading attendance file: {e}")
    
    def view_all_attendance(self):
        """Display all attendance records grouped by date"""
        records_by_date = {}
        
        try:
            with open(self.attendance_file, 'r') as f:
                reader = csv.reader(f)
                headers = next(reader)  # Skip header row
                
                for row in reader:
                    if len(row) >= 3:
                        date = row[1]
                        if date not in records_by_date:
                            records_by_date[date] = []
                        records_by_date[date].append(row)
            
            if records_by_date:
                print("\n--- All Attendance Records ---")
                # Sort dates in reverse chronological order
                for date in sorted(records_by_date.keys(), reverse=True):
                    print(f"\nDate: {date} (Total: {len(records_by_date[date])})")
                    print("Name                  Time")
                    print("-" * 30)
                    for record in records_by_date[date]:
                        print(f"{record[0]:<20} {record[2]}")
            else:
                print("No attendance records found")
        except Exception as e:
            print(f"Error reading attendance file: {e}")
    
    def run_recognition(self):
        """Run the main face recognition loop"""
        if not self.known_faces:
            print("No known faces loaded. Please add faces first.")
            return
        
        print("\nStarting face recognition...")
        print("Press 'q' to quit")
        
        # Initialize webcam
        cam = self.setup_camera()
        if cam is None:
            return
            
        self.frame_count = 0
        last_names = {}  # To keep track of faces currently being recognized
        
        # Flag to control the loop
        running = True
        
        while running:
            ret, frame = cam.read()
            if not ret:
                print("Failed to grab frame.")
                break
                
            # Create a copy for display
            display_frame = frame.copy()
            
            # Process only every nth frame to improve performance
            process_this_frame = self.frame_count % self.process_every_n_frames == 0
            self.frame_count += 1
            
            if process_this_frame:
                # Resize frame for faster processing
                small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
                
                # Convert to RGB for face_recognition
                rgb_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
                
                # Find faces in frame
                face_locations = face_recognition.face_locations(rgb_frame)
                face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
                
                # Clear previous detections
                last_names = {}
                
                # Process each face found
                for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                    # Scale back up face locations
                    top *= 2
                    right *= 2
                    bottom *= 2
                    left *= 2
                    
                    # See if the face matches any known face
                    matches = face_recognition.compare_faces(self.known_faces, face_encoding, tolerance=0.6)
                    
                    name = "Unknown"
                    face_distances = face_recognition.face_distance(self.known_faces, face_encoding)
                    
                    if len(face_distances) > 0:
                        best_match_index = np.argmin(face_distances)
                        confidence = 1 - min(1, face_distances[best_match_index])
                        
                        if matches[best_match_index] and confidence > 0.5:
                            name = self.known_names[best_match_index]
                            
                            # Mark attendance for recognized person
                            marked = self.mark_attendance(name)
                            last_names[(left, top, right, bottom)] = (name, marked, confidence)
            
            # Display results for all detected faces
            for (left, top, right, bottom), (name, marked, confidence) in last_names.items():
                # Draw rectangle around face
                cv2.rectangle(display_frame, (left, top), (right, bottom), (0, 255, 0), 2)
                
                # Add checkmark if attendance was just marked
                display_name = f"{name} ✓" if marked else name
                
                # Add confidence score
                display_name += f" ({confidence:.0%})"
                
                # Draw label
                cv2.rectangle(display_frame, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
                font = cv2.FONT_HERSHEY_DUPLEX
                cv2.putText(display_frame, display_name, (left + 6, bottom - 6), font, 0.8, (255, 255, 255), 1)
            
            # Add current date and time on the frame
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            cv2.putText(display_frame, current_time, 
                       (10, 30), cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1)
            
            # Add quit instructions
            cv2.putText(display_frame, "Press 'q' to quit", 
                       (10, display_frame.shape[0] - 20), cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1)
            
            # Display the resulting frame
            cv2.imshow('Face Recognition Attendance', display_frame)
            
            # Check for key press - important fix for quitting
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                running = False
        
        # Clean up properly
        print("Closing camera...")
        cam.release()
        cv2.destroyAllWindows()
        # Sometimes OpenCV windows don't close properly, so try again
        for i in range(5):
            cv2.waitKey(1)

def main():
    print("=== Face Recognition Attendance System ===")
    system = FaceAttendanceSystem()
    
    while True:
        print("\nMenu:")
        print("1. Start Face Recognition Attendance")
        print("2. Add New Face")
        print("3. View Attendance Records")
        print("4. Exit")
        
        choice = input("\nEnter your choice (1-4): ")
        
        if choice == '1':
            system.run_recognition()
        elif choice == '2':
            system.add_face()
        elif choice == '3':
            system.view_attendance()
        elif choice == '4':
            print("Exiting program. Goodbye!")
            break
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()