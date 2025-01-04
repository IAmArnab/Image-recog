import cv2  # Import OpenCV for image and video processing
import numpy as np  # Import NumPy for numerical operations
import os  # Import OS module for directory handling
from datetime import datetime  # Import datetime for time-based operations

# Function to load the YOLO model for object detection
def load_yolo():
    net = cv2.dnn.readNet('yolov3.weights', 'yolov3.cfg')  # Load YOLO model weights and configuration
    with open("coco.names", "r") as f:  # Open class names file
        classes = [line.strip() for line in f.readlines()]  # Read class names into a list
    layer_names = net.getLayerNames()  # Get all layer names from YOLO
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]  # Identify output layers
    return net, output_layers, classes  # Return model, layers, and classes

# Function to filter overlapping bounding boxes
def filter_overlapping_boxes(boxes):
    unique_boxes = []  # List to store non-overlapping bounding boxes
    for x, y, w, h in boxes:  # Iterate over all bounding boxes
        overlaps = False  # Flag to check overlap
        for ux, uy, uw, uh in unique_boxes:  # Compare with unique boxes
            if (x < ux + uw and x + w > ux and y < uy + uh and y + h > uy):  # Check for overlap
                overlaps = True
                break
        if not overlaps:  # Add to unique boxes if no overlap is found
            unique_boxes.append((x, y, w, h))
    return unique_boxes  # Return filtered boxes

# Function to compare images for similarity
def compare_images(reference_image, current_image):
    reference_gray = cv2.cvtColor(reference_image, cv2.COLOR_BGR2GRAY)  # Convert reference image to grayscale
    current_gray = cv2.cvtColor(current_image, cv2.COLOR_BGR2GRAY)  # Convert current image to grayscale
    similarity = cv2.matchTemplate(reference_gray, current_gray, cv2.TM_CCOEFF_NORMED)  # Compute similarity
    return similarity.max() * 100  # Return similarity percentage

# Function to process IP camera stream for object detection and defect analysis
def process_ip_camera(ip_camera_url, reference_folder, defect_folder, similarity_threshold=62):
    if not os.path.exists(defect_folder):  # Ensure defect folder exists
        os.makedirs(defect_folder)

    cap = cv2.VideoCapture(ip_camera_url)  # Open the IP camera stream
    if not cap.isOpened():  # Check if the stream is accessible
        print("Error: Unable to access IP camera stream.")
        return

    net, output_layers, classes = load_yolo()  # Load YOLO model
    while True:
        ret, frame = cap.read()  # Read a frame from the stream
        if not ret:  # Break the loop if no frame is received
            print("Stream ended or interrupted.")
            break

        # Detect objects using YOLO
        blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)  # Prepare image
        net.setInput(blob)  # Set YOLO input
        outs = net.forward(output_layers)  # Perform forward pass

        boxes = []  # List to store bounding boxes
        confidences = []  # List to store confidence scores
        for out in outs:  # Loop over YOLO output layers
            for detection in out:  # Loop over detections
                scores = detection[5:]  # Extract confidence scores for classes
                class_id = np.argmax(scores)  # Find the class with highest confidence
                confidence = scores[class_id]  # Get the confidence value
                if confidence > 0.5:  # Consider detections with confidence > 50%
                    center_x, center_y, w, h = (detection[:4] * [frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]]).astype(int)
                    x = int(center_x - w / 2)  # Calculate top-left corner x-coordinate
                    y = int(center_y - h / 2)  # Calculate top-left corner y-coordinate
                    boxes.append([x, y, w, h])  # Add bounding box to the list
                    confidences.append(float(confidence))  # Add confidence to the list

        indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)  # Apply Non-Maximum Suppression
        #detections = [boxes[i[0]] for i in indices]  # Filter detections
        #detections = [boxes[i[0]] for i in indices] if len(indices) > 0 else []  # chnged
       
        # If indices is not empty, extract corresponding bounding boxes
        if len(indices) > 0:
            detections = [boxes[i] for i in indices.flatten()]  # Use flatten to convert it to 1D array
        else:
            detections = []  # No detections if indices is empty
        
        print(detections)


        # Process detections for defect analysis
        for i, box in enumerate(detections):
            x, y, w, h = box
            cropped = frame[y:y+h, x:x+w]  # Crop the detected object

            is_defective = False  # Initialize defect flag
            for ref_file in os.listdir(reference_folder):  # Loop through reference images
                ref_path = os.path.join(reference_folder, ref_file)  # Get reference file path
                reference_image = cv2.imread(ref_path)  # Read reference image
                if reference_image is None:  # Skip if the image couldn't be loaded
                    continue
                similarity = compare_images(reference_image, cropped)  # Compare images
                if similarity < similarity_threshold:  # Mark defective if similarity is below threshold
                    is_defective = True
                    break

            # Handle defective and non-defective images
            if is_defective:
                defect_path = os.path.join(defect_folder, f"defective_{datetime.now().strftime('%Y%m%d%H%M%S')}_{i}.jpg")
                cv2.imwrite(defect_path, cropped)  # Save defective image
                print(f"Defective image saved: {defect_path}")
            else:
                print("Non-defective image discarded.")  # Log discarded non-defective image

        cv2.imshow("Processing", frame)  # Display the frame in real-time
        if cv2.waitKey(1) & 0xFF == ord('q'):  # Quit if 'q' is pressed
            break

    cap.release()  # Release the video capture object
    cv2.destroyAllWindows()  # Close all OpenCV windows

# Main program
if __name__ == "__main__":
    ip_camera_url = input("Enter the IP camera URL: ")  # Get IP camera URL from user
    reference_folder = "reference_folder"  # Path to reference images folder
    defect_folder = "defected_folder"  # Path to save defective images
    process_ip_camera(ip_camera_url, reference_folder, defect_folder)  # Start processing the IP camera stream
