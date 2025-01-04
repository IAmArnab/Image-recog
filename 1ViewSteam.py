import cv2

def main():
    # Replace with your IP camera's stream URL.
    # Format: "http://<username>:<password>@<ip>:<port>/video"
    # Example: "http://admin:12345@192.168.1.100:8080/video"
    #ip_camera_url = "http://admin:Mztpl123@192.168.1.108:80/video"
    #ip_camera_url = "http://admin:Mztpl123@192.168.1.108:80/live"
    ip_camera_url="rtsp://admin:Mztpl123@192.168.1.108:80/cam/realmonitor?channel=1&subtype=0"


    # Create a VideoCapture object
    cap = cv2.VideoCapture(ip_camera_url)

    if not cap.isOpened():
        print("Error: Unable to connect to the IP camera.")
        return

    print("Press 'q' to exit the stream.")

    while True:
        # Read frame from the video stream
        ret, frame = cap.read()

        if not ret:
            print("Error: Unable to retrieve frame from the IP camera.")
            break

        # Display the frame
        cv2.imshow('IP Camera Stream', frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the capture object and close display window
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
