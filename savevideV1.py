
import cv2
import time
import os
import datetime
# frame per second
fps = 30.0
# RTSP stream URL
rtsp_url = "rtsp://admin:Mztpl123@192.168.1.108:80/cam/realmonitor?channel=1&subtype=0"
video_folder = "video/"

# Open the RTSP stream
cap = cv2.VideoCapture(rtsp_url)

# Check if the stream is opened
if not cap.isOpened():
    print("Error: Unable to open RTSP stream")
    exit()

# Set the video codec to H.264
fourcc = cv2.VideoWriter_fourcc(*'X264')

# Set the output video file name and FPS
 #output_file = "output.mp4"

# Video duration in seconds (1 minute)
video_duration = 1 * 10

while True:
    #Save the output image with timestamp
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_file = f"{video_folder}output_{timestamp}.mp4"


    # Get the video frame size
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Create a VideoWriter object
    out = cv2.VideoWriter(output_file, fourcc, fps, (frame_width, frame_height))

    start_time = time.time()

    # Read frames from the RTSP stream and save to the output file
    while True:
        ret, frame = cap.read()
        
        if not ret:
            break

        out.write(frame)

        # Display the frame (optional)
        cv2.imshow('RTSP Stream', frame)

        # Check if the duration has elapsed
        if time.time() - start_time > video_duration:
            break

        # Exit on key press (optional)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            # Release resources
            cap.release()
            out.release()
            cv2.destroyAllWindows()
            exit()
    # Release the VideoWriter for the current segment
    out.release()
