import cv2
import mediapipe as mp
import numpy as np
import json
import os
from datetime import datetime

# File to save/load polygon boundary and settings
boundary_file = "boundary_polygon.json"
video_name = "asd.mp4"

# Create output directory if it doesn't exist
output_dir = "output"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Mediapipe pose setup
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Polygon-related variables
polygon_points = []
drawing = False  # True if user is drawing the polygon
temp_frame = None  # Temporary frame for live drawing

# Video writer setup
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = None  # Will be initialized when we get the first frame


# Function to load polygon and settings from file
def load_polygon():
    try:
        with open(boundary_file, 'r') as f:
            data = json.load(f)
            if isinstance(data, dict):
                return data.get('polygon'), data.get('trigger_direction')
            else:  # Handle legacy format where only polygon was stored
                return data, None
    except FileNotFoundError:
        return None, None


# Function to save polygon and settings to file
def save_polygon(points, trigger_direction=None):
    data = {
        'polygon': points,
        'trigger_direction': trigger_direction
    }
    with open(boundary_file, 'w') as f:
        json.dump(data, f)


# Function to draw the polygon interactively
def draw_polygon(event, x, y, flags, param):
    global drawing, polygon_points, temp_frame

    if event == cv2.EVENT_LBUTTONDOWN:  # Start adding points
        polygon_points.append((x, y))
        drawing = True

    elif event == cv2.EVENT_MOUSEMOVE and drawing:  # Show dynamic lines
        if temp_frame is not None:
            frame_copy = temp_frame.copy()
            if len(polygon_points) > 0:
                cv2.polylines(frame_copy, [np.array(polygon_points)], isClosed=False, color=(0, 255, 0), thickness=2)
                cv2.line(frame_copy, polygon_points[-1], (x, y), (0, 255, 0), 2)
            cv2.imshow("Draw Polygon", frame_copy)

    elif event == cv2.EVENT_LBUTTONUP:  # Finalize point
        polygon_points.append((x, y))
        if temp_frame is not None:
            cv2.imshow("Draw Polygon", temp_frame)

    elif event == cv2.EVENT_RBUTTONDOWN:  # Finish drawing with right click
        print("Polygon drawing completed.")
        cv2.destroyWindow("Draw Polygon")


# Function to allow the user to select a polygon
def select_polygon(frame):
    global temp_frame
    temp_frame = frame.copy()  # Store the original frame for live drawing
    cv2.namedWindow("Draw Polygon")
    cv2.setMouseCallback("Draw Polygon", draw_polygon)

    print("Draw the polygon using left mouse clicks. Right-click to finish.")
    while True:
        cv2.imshow("Draw Polygon", temp_frame)
        key = cv2.waitKey(1)
        if key == ord('c'):  # Confirm the polygon
            break
    cv2.destroyWindow("Draw Polygon")
    return polygon_points


# Function to check if a point is inside the polygon
def is_point_inside_polygon(point, polygon):
    return cv2.pointPolygonTest(np.array(polygon), point, False) >= 0


# Function to determine the direction of movement
def get_keypoint_direction(keypoint_index, landmark, prev_positions, frame_width):
    """
    Determines if the keypoint is moving left or right.
    """
    x = landmark.x * frame_width
    if keypoint_index not in prev_positions:
        prev_positions[keypoint_index] = x
        return None  # No previous data to determine direction

    prev_x = prev_positions[keypoint_index]
    prev_positions[keypoint_index] = x

    if x > prev_x:
        return "right"
    elif x < prev_x:
        return "left"
    else:
        return None


# Function to get person's center position relative to polygon
def get_person_position(landmarks, frame_width, polygon):
    # Use hip points to determine person's center
    left_hip = landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP]
    right_hip = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP]
    
    center_x = (left_hip.x + right_hip.x) / 2 * frame_width
    
    # Get polygon's leftmost and rightmost x coordinates
    polygon_x_coords = [p[0] for p in polygon]
    polygon_center_x = sum(polygon_x_coords) / len(polygon_x_coords)
    
    return "left" if center_x < polygon_center_x else "right"


# Main script
cap = cv2.VideoCapture(video_name)  # Replace with video file path if needed

# Load or define the polygon and trigger direction
polygon, trigger_direction = load_polygon()
if polygon is None:
    ret, frame = cap.read()
    if not ret:
        print("Error reading from the camera.")
        exit(1)
    print("No polygon found. Please draw one.")
    polygon = select_polygon(frame)
    trigger_direction = input("Enter the direction to trigger the alarm (left/right): ").strip().lower()
    save_polygon(polygon, trigger_direction)
elif trigger_direction is None:
    trigger_direction = input("Enter the direction to trigger the alarm (left/right): ").strip().lower()
    save_polygon(polygon, trigger_direction)

polygon = np.array(polygon)  # Convert to numpy array for OpenCV functions
prev_positions = {}  # Store previous positions for each keypoint

# Get first frame to initialize video writer
ret, frame = cap.read()
if ret:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(output_dir, f'output_{timestamp}.mp4')
    out = cv2.VideoWriter(output_path, fourcc, 30.0, (frame.shape[1], frame.shape[0]))
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reset video to start

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Draw the polygon on the frame
    cv2.polylines(frame, [polygon], isClosed=True, color=(0, 255, 0), thickness=2)

    # Convert the frame to RGB for Mediapipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb_frame)

    # Check pose landmarks
    if results.pose_landmarks:
        # Since Mediapipe now detects multiple people, we need to handle each detection
        for person_id, pose_landmarks in enumerate(results.pose_landmarks if isinstance(results.pose_landmarks, list) else [results.pose_landmarks]):
            mp_drawing.draw_landmarks(frame, pose_landmarks, mp_pose.POSE_CONNECTIONS)
            
            # Get person's position relative to polygon
            person_position = get_person_position(pose_landmarks, frame.shape[1], polygon)
            
            # Only check for intersection if person is on the triggering side
            if person_position == trigger_direction:
                # Check if hands intersect with polygon
                alarm_triggered = False
                hand_landmarks = [
                    mp_pose.PoseLandmark.LEFT_WRIST,
                    mp_pose.PoseLandmark.RIGHT_WRIST,
                    mp_pose.PoseLandmark.LEFT_PINKY,
                    mp_pose.PoseLandmark.RIGHT_PINKY,
                    mp_pose.PoseLandmark.LEFT_INDEX,
                    mp_pose.PoseLandmark.RIGHT_INDEX,
                    mp_pose.PoseLandmark.LEFT_THUMB,
                    mp_pose.PoseLandmark.RIGHT_THUMB
                ]
                
                for landmark_id in hand_landmarks:
                    landmark = pose_landmarks.landmark[landmark_id]
                    x = int(landmark.x * frame.shape[1])
                    y = int(landmark.y * frame.shape[0])
                    
                    if is_point_inside_polygon((x, y), polygon):
                        alarm_triggered = True
                        break
                        
                if alarm_triggered:
                    cv2.putText(frame, f"ALARM! Kisi {person_id + 1}", (50, 50 + person_id * 30), 
                              cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                    
            # Draw keypoints
            for landmark in pose_landmarks.landmark:
                x = int(landmark.x * frame.shape[1])
                y = int(landmark.y * frame.shape[0])
                cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

    # Write the frame to output video
    if out is not None:
        out.write(frame)

    # Display the frame
    cv2.imshow("Video Stream", frame)

    # Exit condition
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up
cap.release()
if out is not None:
    out.release()
cv2.destroyAllWindows()
print(f"Output video saved to: {output_path}")
