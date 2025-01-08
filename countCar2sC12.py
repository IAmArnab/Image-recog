import cv2  # Import OpenCV for image and video processing
import pafy
import numpy as np  # Import NumPy for numerical operations
import os  # Import OS module for directory handling
from datetime import datetime  # Import datetime for time-based operations
import logging

def concat_and_show(img_1, img_2, lbl):
    # print(img_1.shape)
    h1, w1 = img_1.shape[:2]
    h2, w2 = img_2.shape[:2]
    img_3 = np.zeros((max(h1, h2), w1 + w2, 3), dtype=np.uint8)
    img_3[:, :, :] = (255, 255, 255)
    img_3[:h1, :w1,:3] = img_1
    img_3[:h2, w1:w1 + w2, :] = img_2
    return img_3


def draw_trapzoid(frame, corners,color):
    cv2.line(frame, (corners[0][0][0],corners[0][0][1]),(corners[1][0][0],corners[1][0][1]),color,1)
    cv2.line(frame, (corners[1][0][0], corners[1][0][1]), (corners[2][0][0], corners[2][0][1]), color, 1)
    cv2.line(frame, (corners[2][0][0], corners[2][0][1]), (corners[3][0][0], corners[3][0][1]), color, 1)
    cv2.line(frame, (corners[3][0][0], corners[3][0][1]), (corners[0][0][0], corners[0][0][1]), color, 1)

def detect_sheets(frame):

    frame_x, frame_y = frame.shape[:2]
    frame_area = frame_x * frame_y
    sheets = []
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.medianBlur(gray, 5)
    sharpen_kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    sharpen = cv2.filter2D(blur, -1, sharpen_kernel)
    # Threshold and morph close
    thresh = cv2.threshold(sharpen, 130, 255, cv2.THRESH_BINARY_INV)[1]

    # cv2.imshow("thresh", thresh)
    # cv2.waitKey()

    contours, hierarchy = cv2.findContours(thresh, 1, 2)

    rect_cnt = 0
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        ratio = float(w) / h
        if (ratio >0.6 and ratio < 1.4 and cv2.contourArea(cnt) > frame_area*0.03 and cv2.contourArea(cnt) < frame_area*0.15) :
            rect_cnt += 1
            print(f"Found rect #{rect_cnt} with area {cv2.contourArea(cnt)}")
            sheets.append([x,y,w,h])
    # cv2.imshow("Shapes", frame)
    return sheets

def filter_sheets(sheets, width):
    filtered_sheets = []
    for sheet in sheets:
        #{CONFIG} Sheets having laft top corner in below window will be considered for processing
        if (sheet[0] < width*.45 and sheet[0] > width*.4):
            filtered_sheets.append(sheet)
    # return sheets
    return filtered_sheets
def pre_process(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.medianBlur(gray, 5)
    sharpen_kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    sharpen = cv2.filter2D(blur, -1, sharpen_kernel)
    return blur

# Function to compare images for similarity
def compare_images(reference_image, current_image):
    similarity = cv2.matchTemplate(reference_image, current_image, cv2.TM_CCOEFF_NORMED)  # Compute similarity
    # img = concat_and_show(reference_image, current_image,"")
    # cv2.putText(img, f"{int(similarity.max() * 100)}", (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
    #                     (0, 0, 0), 1)
    # cv2.imshow("Compare",img)
    return similarity.max() * 100  # Return similarity percentage

# Function to process a frame
def process_frame(frame, frame_i, timestamp_in_mil, reference_folder, similarity_threshold, net, output_layers, classes):
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
                center_x, center_y, w, h = (
                            detection[:4] * [frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]]).astype(
                    int)
                x = int(center_x - w / 2)  # Calculate top-left corner x-coordinate
                y = int(center_y - h / 2)  # Calculate top-left corner y-coordinate
                boxes.append([x, y, w, h])  # Add bounding box to the list
                confidences.append(float(confidence))  # Add confidence to the list

    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)  # Apply Non-Maximum Suppression
    # detections = [boxes[i[0]] for i in indices]  # Filter detections
    # detections = [boxes[i[0]] for i in indices] if len(indices) > 0 else []  # chnged

    # If indices is not empty, extract corresponding bounding boxes
    if len(indices) > 0:
        # timestamps.append(cap.get(cv2.CAP_PROP_POS_MSEC))
        # calc_timestamps.append(calc_timestamps[-1] + 1000/fps)
        detections = [boxes[i] for i in indices.flatten()]  # Use flatten to convert it to 1D array
        # print(detections)
    else:
        detections = []  # No detections if indices is empty

    # Process detections for defect analysis
    for i, box in enumerate(detections):
        x, y, w, h = box
        cropped = frame[y:y + h, x:x + w]  # Crop the detected object
        if cropped is None or h<0 or w<0 or x<0 or y<0:  # Skip if the image couldn't be loaded
            continue
        # print(box)
        # cv2.imshow(f"Error crop", cropped)
        is_defective = False  # Initialize defect flag

        for ref_file in os.listdir(reference_folder):  # Loop through reference images
            ref_path = os.path.join(reference_folder, ref_file)  # Get reference file path
            reference_image = cv2.imread(ref_path)  # Read reference image
            if reference_image is None:  # Skip if the image couldn't be loaded
                continue
            h1, w1 = reference_image.shape[:2]
            if h1<=h or w1<=w : # Skip small reference image
                continue
            try:
                similarity = compare_images(reference_image, cropped)  # Compare images# concatenate image Horizontally
                print(f"Frame {frame_i}, Similarity {similarity}")
                if similarity < similarity_threshold:  # Mark defective if similarity is below threshold
                    is_defective = True
                    break
            except Exception as e:
                logging.error(e, exc_info=True)
                print(f"Error caused while comparing x{x}, y{y}, w{w}, h{h}")
                print(f"Reference Image size height:{h1} width:{w1}")

        # Handle defective and non-defective images
        if is_defective:
            result = concat_and_show(reference_image, cropped, f"Match found {int(similarity)}%")
            defect_path = os.path.join(defect_folder, f"defective_match_{int(similarity)}%_{datetime.now().strftime('%Y%m%d%H%M%S')}_{i}.jpg")
            cv2.imwrite(defect_path, result)  # Save defective image
            print(f"Defective image saved: {defect_path}")
        else:
            print("Non-defective image discarded.")  # Log discarded non-defective image

    cv2.imshow("Processing", frame)  # Display the frame in real-time

def check_for_fault(frame, frame_i, sheets, similarity_threshold, reference_folder):
    # Process detections for defect analysis
    org_frame = frame.copy()
    sheet_i =0
    for sheet in sheets:
        sheet_i+=1
        x, y, w, h = sheet
        cropped = frame[y:y + h, x:x + w]  # Crop the detected object
        is_defective = False  # Initialize defect flag

        for ref_file in os.listdir(reference_folder):  # Loop through reference images
            # print(ref_file)
            ref_path = os.path.join(reference_folder, ref_file)  # Get reference file path
            reference_image = cv2.imread(ref_path)  # Read reference image
            if reference_image is None:  # Skip if the image couldn't be loaded
                continue
            h1, w1 = reference_image.shape[:2]
            if h1 <= h or w1 <= w:  # Skip small reference image
                continue
            try:
                similarity = compare_images(reference_image, cropped)  # Compare images# concatenate image Horizontally
                print(f"Frame {frame_i}, Similarity {similarity}")
                if similarity < similarity_threshold:  # Mark defective if similarity is below threshold
                    is_defective = True
                    break
            except Exception as e:
                logging.error(e, exc_info=True)
                print(f"Error caused while comparing x{x}, y{y}, w{w}, h{h}")
                print(f"Reference Image size height:{h1} width:{w1}")

        # Handle defective and non-defective images
        if is_defective:
            frame_copy = org_frame.copy()
            cv2.putText(frame_copy, f"Bad Sheet {int(similarity)}", (sheet[0]-5, sheet[1]-5), cv2.FONT_HERSHEY_SIMPLEX, 0.3,
                        (0, 0, 255), 1)
            cv2.rectangle(frame_copy, (x, y), (x + w, y + h), (0, 0, 255), 1)

            # result = concat_and_show(reference_image, cropped, f"Match found {int(similarity)}%")
            defect_path = os.path.join(defect_folder,
                                       f"defective_match_{int(similarity)}%_{datetime.now().strftime('%Y%m%d%H%M%S')}_{frame_i}.jpg")
            cv2.imwrite(defect_path, frame_copy)  # Save defective image
            print(f"Defective image saved: {defect_path}")
            cv2.putText(frame, f"Bad Sheet {int(similarity)}", (sheet[0]-5, sheet[1]-5), cv2.FONT_HERSHEY_SIMPLEX, 0.3,
                        (0, 0, 255), 1)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 1)
            # print(f"Found rect #{rect_cnt} with area {cv2.contourArea(cnt)}")
        else:
            print("Non-defective image discarded.")  # Log discarded non-defective image
            cv2.putText(frame, f"Good Sheet {int(similarity)}", (sheet[0]-5, sheet[1]-5), cv2.FONT_HERSHEY_SIMPLEX, 0.3,
                        (0, 255, 0), 1)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 1)

        cv2.imshow("Processing",frame)
    return frame
# Function to process IP camera stream for object detection and defect analysis

def process_ip_camera(ip_camera_url, reference_folder, defect_folder, similarity_threshold=55):
    if not os.path.exists(defect_folder):  # Ensure defect folder exists
        os.makedirs(defect_folder)

    cap = cv2.VideoCapture(ip_camera_url)  # Open the IP camera stream
    if not cap.isOpened():  # Check if the stream is accessible
        print("Error: Unable to access IP camera stream.")
        return

    # We need to set resolutions.
    # so, convert them from float to integer.
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))

    size = (frame_width, frame_height)
    # Below VideoWriter object will create
    # a frame of above defined The output
    # is stored in 'filename.avi' file.
    result = cv2.VideoWriter(f"result_{datetime.now().strftime('%Y%m%d%H%M%S')}.avi",
                         cv2.VideoWriter_fourcc(*'MJPG'),
                         10, size)
    frame_i = 0
    while (cap.isOpened()):
        frame_i +=1
        # if frame_i > 300:
        #     continue
        # if frame_i > 310:
        #     break
        if not cap.isOpened():  # Check if the stream is accessible
            print("Error: Unable to access IP camera stream.")
            return
        ret, frame = cap.read()  # Read a frame from the stream
        if not ret:  # Break the loop if no frame is received
            print("Stream ended or interrupted.")
            break
        sheets = detect_sheets(frame)
        filtered_sheets = filter_sheets(sheets, frame.shape[:2][1])
        result_frame = check_for_fault(frame, frame_i, filtered_sheets, similarity_threshold, reference_folder)
        result.write(result_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):  # Quit if 'q' is pressed
            break

    # When everything done, release
    # the video capture and video
    # write objects
    result.release()
    cap.release()  # Release the video capture object
    cv2.destroyAllWindows()  # Close all OpenCV windows

# Main program
if __name__ == "__main__":
    # ip_camera_url = input("Enter the IP camera URL: ")  # Get IP camera URL from user
    # ip_camera_url = 'http://commondatastorage.googleapis.com/gtv-videos-bucket/sample/VolkswagenGTIReview.mp4'
    # ip_camera_url = 'rtsp://admin:Mztpl123@192.168.1.108:80/cam/realmonitor?channel=1&subtype=0'
    ip_camera_url = 'C:\\Users\\61098198\\Downloads\\videoplayback.mp4'

    reference_folder = "reference_folder"  # Path to reference images folder
    defect_folder = "defected_folder"  # Path to save defective images
    process_ip_camera(ip_camera_url, reference_folder, defect_folder)  # Start processing the IP camera stream
