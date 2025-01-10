import time

import cv2  # Import OpenCV for image and video processing
import numpy as np  # Import NumPy for numerical operations
import os  # Import OS module for directory handling
from datetime import datetime  # Import datetime for time-based operations
import logging
from matplotlib import pyplot as plt
# import tensorflow as tf

# Recreate the exact same model, including its weights and the optimizer
# model = tf.keras.models.load_model('model.h5')

def count_patches(img):

    img = cv2.normalize(src=img, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
    plt.subplot(2, 4, 1)
    plt.imshow(img)
    plt.axis('off')  # Hide the axis labels
    plt.title("sheet")
    test = img.copy()
    # Read gray image
    # Create default parametrization LSD
    lsd = cv2.createLineSegmentDetector(0)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Detect lines in the image
    lines = lsd.detect(gray)[0]  # Position 0 of the returned tuple are the detected lines

    # Draw detected lines in the image
    drawn_img = lsd.drawSegments(img, lines)

    # Show image
    # cv2.imshow("LSD", drawn_img)
    # print(f"cont count {len(lines)}")
    return len(lines)

def concat_and_show(img_1, img_2, lbl):
    # print(img_1.shape)
    h1, w1 = img_1.shape[:2]
    h2, w2 = img_2.shape[:2]
    img_3 = np.zeros((max(h1, h2), w1 + w2, 3), dtype=np.uint8)
    img_3[:, :, :] = (255, 255, 255)
    img_3[:h1, :w1,:3] = img_1
    img_3[:h2, w1:w1 + w2, :] = img_2
    return img_3

def simplify_contour(contour, n_corners=4):
    '''
    Binary searches best `epsilon` value to force contour
        approximation contain exactly `n_corners` points.

    :param contour: OpenCV2 contour.
    :param n_corners: Number of corners (points) the contour must contain.

    :returns: Simplified contour in successful case. Otherwise returns initial contour.
    '''
    n_iter, max_iter = 0, 100
    lb, ub = 0., 1.

    while True:
        n_iter += 1
        if n_iter > max_iter:
            return contour

        k = (lb + ub) / 2.
        eps = k * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, eps, True)

        if len(approx) > n_corners:
            lb = (lb + ub) / 2.
        elif len(approx) < n_corners:
            ub = (lb + ub) / 2.
        else:
            return approx

def draw_trapzoid(frame, corners,color):
    cv2.line(frame, (corners[0][0][0],corners[0][0][1]),(corners[1][0][0],corners[1][0][1]),color,1)
    cv2.line(frame, (corners[1][0][0], corners[1][0][1]), (corners[2][0][0], corners[2][0][1]), color, 1)
    cv2.line(frame, (corners[2][0][0], corners[2][0][1]), (corners[3][0][0], corners[3][0][1]), color, 1)
    cv2.line(frame, (corners[3][0][0], corners[3][0][1]), (corners[0][0][0], corners[0][0][1]), color, 1)

def detect_sheets(frame):
    test = frame.copy()
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
        cnt = simplify_contour(cnt)
        x, y, w, h = cv2.boundingRect(cnt)
        ratio = float(w) / h
        if (len(cnt) == 4 and ratio >0.6 and ratio < 1.4 and cv2.contourArea(cnt) > frame_area*0.03 and cv2.contourArea(cnt) < frame_area*0.15) :
            rect_cnt += 1
            # print(f"Found rect #{rect_cnt} with area {cv2.contourArea(cnt)}")
            sheets.append(cnt)
            # cv2.drawContours(test,[cnt], -1, (0, 255, 0), 1)
            # epsilon = 0.01 * cv2.arcLength(cnt, True)
            # approx = cv2.approxPolyDP(cnt, epsilon, True)
            # cv2.drawContours(test,[cnt], -1, (255, 0, 0), 1)
            # print()
            # cv2.imshow("Shapes", test)
            # cv2.waitKey()
    return sheets

def filter_sheets(sheets, width):
    filtered_sheets = []
    for sheet in sheets:
        #{CONFIG} Sheets having laft top corner in below window will be considered for processing
        # print(sheet[0][0][0])
        # print(width*.60)
        if sheet[0][0][0] < width*.65 and sheet[0][0][0] > width*.60 :
            filtered_sheets.append(sheet)
    return sheets
    # return filtered_sheets
def pre_process(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.medianBlur(gray, 5)
    sharpen_kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    sharpen = cv2.filter2D(blur, -1, sharpen_kernel)
    return blur

# {Not in use}Function to compare images for similarity
def compare_images(reference_image, frame, corners):
    plt.subplot(2, 1, 1)
    plt.imshow(reference_image)
    plt.axis('off')  # Hide the axis labels
    plt.title("Image 1")
    if len(corners) != 4:
        return -1
    img = frame.copy()
    ht, wd = reference_image.shape[:2]
    # reformat input corners to x,y list
    icorners = []
    for corner in corners:
        pt = [corner[0][0], corner[0][1]]
        icorners.append(pt)
    icorners = np.float32(icorners)
    # get corresponding output corners form width and height
    ocorners = [[0, 0], [0, ht], [wd, ht], [wd, 0]]
    ocorners = np.float32(ocorners)

    # get perspective tranformation matrix
    M = cv2.getPerspectiveTransform(icorners, ocorners)

    # do perspective
    warped = cv2.warpPerspective(img, M, (wd, ht))
    plt.subplot(2, 1, 2)
    plt.imshow(warped)
    plt.axis('off')  # Hide the axis labels
    plt.title("Image 2")
    similarity = cv2.matchTemplate(warped, reference_image, cv2.TM_CCOEFF_NORMED)  # Compute similarity
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(similarity)
    top_left = max_loc
    # bottom_right = (top_left[0] + w, top_left[1] + h)

    # print(similarity)
    # img = concat_and_show(reference_image, warped,"")
    # img = np.concatenate((reference_image, warped), axis=1)
    # cv2.putText(img, f"{int(similarity.max() * 100)}", (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
    #                     (0, 0, 0), 1)
    # cv2.imshow("Compare",img)
    # cv2.waitKey()
    # plt.title(f"Similarity {int(similarity.max() * 100)}")
    # plt.show()
    cv2.imwrite(f"samples\\sheet_{datetime.now().strftime('%Y%m%d%H%M%S')}.jpg", warped)
    return similarity.max() * 100  # Return similarity percentage

def is_good_sheet(frame, corners, tolarance_level, skip_this_frame):
    ht, wd = [150, 130]
    # reformat input corners to x,y list
    icorners = []
    for corner in corners:
        pt = [corner[0][0], corner[0][1]]
        icorners.append(pt)
    icorners = np.float32(icorners)
    # get corresponding output corners form width and height
    ocorners = [[0, 0], [0, ht], [wd, ht], [wd, 0]]
    ocorners = np.float32(ocorners)

    # get perspective tranformation matrix
    M = cv2.getPerspectiveTransform(icorners, ocorners)

    # do perspective
    sheet = cv2.warpPerspective(frame, M, (wd, ht))
    count = count_patches(sheet)
    if (not skip_this_frame) and count > tolarance_level:
        cv2.imwrite(f"defected_folder\\sheet_defect_count_{count}_{datetime.now().strftime('%Y%m%d%H%M%S-%f')}.jpg", sheet)
    return count
    # print(sheet.shape)
    # x_val=[]
    # x_val.append(sheet)
    # # resized_arr = cv2.resize(sheet, (150, 130))  # Reshaping images to preferred size
    # x_val = np.array(x_val) / 255
    # x_val.reshape(-1, 150, 130, 1)
    # print(x_val.shape)
    # predictions = model.predict([x_val])
    # return predictions[0]
def check_for_fault(frame, frame_i, sheets, tolerance_level, skip_this_frame):
    # Process detections for defect analysis
    frame_copy = frame.copy()
    sheet_i =0
    for sheet in sheets:
        sheet_i+=1
        is_defective = False  # Initialize defect flag

        # similarities = []
        # for ref_file in os.listdir(reference_folder):  # Loop through reference images
        #     ref_path = os.path.join(reference_folder, ref_file)  # Get reference file path
        #     reference_image = cv2.imread(ref_path)  # Read reference image
        #     if reference_image is None:  # Skip if the image couldn't be loaded
        #         continue
        #     try:
        #         similarities.append(compare_images(reference_image, frame, sheet))  # Compare images# concatenate image Horizontally
        #
        #     except Exception as e:
        #         logging.error(e, exc_info=True)

        result = is_good_sheet(frame, sheet, tolerance_level, skip_this_frame)
        defect_count = result
        if defect_count > tolerance_level:  # Mark defective if similarity is below threshold
            is_defective = True
        # print(f"Frame {frame_i}, Sheet {sheet_i}, Is Bad {is_defective}")

        # Handle defective and non-defective images
        if is_defective:
            cv2.putText(frame_copy, f"Bad Sheet. defect count : {int(defect_count)}", (sheet[0][0][0]-5, sheet[0][0][1]-5), cv2.FONT_HERSHEY_SIMPLEX, 0.3,
                        (0, 0, 255), 1)
            cv2.drawContours(frame_copy,[sheet], -1, (0, 0, 255), 1)

            # result = concat_and_show(reference_image, cropped, f"Match found {int(similarity)}%")
            # defect_path = os.path.join(defect_folder,
            #                            f"defective_match_{int(similarity)}%_{datetime.now().strftime('%Y%m%d%H%M%S')}_{frame_i}.jpg")
            # cv2.imwrite(defect_path, frame_copy)  # Save defective image
            # print(f"Defective image saved: {defect_path}")

        else:
            # print("Non-defective image discarded.")  # Log discarded non-defective image
            cv2.putText(frame_copy, f"Good Sheet. defect count : {int(defect_count)}", (sheet[0][0][0]-5, sheet[0][0][1]-5), cv2.FONT_HERSHEY_SIMPLEX, 0.3,
                        (0, 255, 0), 1)
            cv2.drawContours(frame_copy, [sheet], -1, (0, 255, 0), 1)
        bigger = cv2.resize(frame_copy, (1280, 720))
        cv2.imshow("Processing",bigger)
        # time.sleep(.1)
    return frame_copy
# Function to process IP camera stream for object detection and defect analysis

def process_ip_camera(ip_camera_url, manual_capture_folder, defect_folder, output_video_folder, tolerance_level=10, skip_frames=0):
    if not os.path.exists(defect_folder):  # Ensure defect folder exists
        os.makedirs(defect_folder)
    if not os.path.exists(output_video_folder):  # Ensure defect folder exists
        os.makedirs(output_video_folder)
    if not os.path.exists(manual_capture_folder):  # Ensure defect folder exists
        os.makedirs(manual_capture_folder)

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
    defect_path = os.path.join(output_video_folder,f"result_{datetime.now().strftime('%Y%m%d%H%M%S')}.avi")
    result = cv2.VideoWriter(defect_path,
                         cv2.VideoWriter_fourcc(*'MJPG'),
                         10, size)
    frame_i = 0
    while (cap.isOpened()):
        skip_this_frame = False;
        frame_i +=1
        if not cap.isOpened():  # Check if the stream is accessible
            print("Error: Unable to access IP camera stream.")
            return
        ret, frame = cap.read()  # Read a frame from the stream
        if not ret:  # Break the loop if no frame is received
            print("Stream ended or interrupted.")
            break

        # Skip few frames
        if frame_i % skip_frames != 0:
            skip_this_frame = True
        # if frame_i > 57:
        #     break
        sheets = detect_sheets(frame)
        filtered_sheets = filter_sheets(sheets, frame.shape[:2][1])
        result_frame = check_for_fault(frame, frame_i, filtered_sheets, tolerance_level, skip_this_frame)
        result.write(result_frame)
        key_pressed = cv2.waitKey(1) & 0xFF
        if key_pressed == ord('q'):  # Quit if 'q' is pressed
            break
        elif key_pressed == ord(' '):
            print("space bar pressed")  # Save a snap if space bar is pressed
            manual_capture_path = os.path.join(manual_capture_folder, f"snap_{datetime.now().strftime('%Y%m%d%H%M%S-%f')}.jpg")
            cv2.imwrite(manual_capture_path, frame)
            print(f"Saved manual capture {manual_capture_path}")


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

    # reference_folder = "reference_folder"  # Path to reference images folder
    defect_folder = "defected_folder"  # Path to save defective images
    output_video_folder = "output_video_folder" # Path to save output video recording
    manual_capture_folder = "manual_capture_folder" # Path to save manual capture
    tolerance_level = 10
    skip_frames = 80
    process_ip_camera(ip_camera_url, manual_capture_folder, defect_folder, output_video_folder,tolerance_level, skip_frames)  # Start processing the IP camera stream
