import cv2
import time
import datetime

# Change the Number to your Main Camera, start from 0 until you find your camera>
cap = cv2.VideoCapture(0)

# Assign face and body calssifier to detect the face and body
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
body_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_fullbody.xml")

# Local Variables
recording = True
detection_stopped_time = None
timer_started = False
detection = False

# Delay time before stop recording
SECOND_TO_RECORD_AFTER_DETECTION = 5

# Saving the recorded video
frame_size = (int(cap.get(3)), int(cap.get(4)))
fourcc = cv2.VideoWriter_fourcc(*"mp4v")

while True:
    _, frame = cap.read()  # Read data from the camera

    # Using GrayScale because face & body detection works with grayscale image
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    bodies = face_cascade.detectMultiScale(gray, 1.3, 5)

    # Checking if there is a face or body in the frame
    if len(faces) + len(bodies) > 0:
        if detection:
            timer_started = False
        else:
            # if the face were detected make a new file and start the recording
            detection = True
            current_time = datetime.datetime.now().strftime("%d-%m-%Y-%H-%M-%S")
            out = cv2.VideoWriter(
                f"{current_time}.mp4", fourcc, 20, frame_size)
            print("Started Recording!")

    # if there is no detection for a face or a body
    elif detection:
        if timer_started:  # wait for 5 seconds before stopping recording
            if time.time() - detection_stopped_time >= SECOND_TO_RECORD_AFTER_DETECTION:
                # Stop recording
                detection = False
                timer_started = False
                out.release()
                print("Stopped Recorded!")
        else:
            timer_started = True
            detection_stopped_time = time.time()

    if detection:
        out.write(frame)

    # Draw a rectangle around the face and the body
    # for (x, y, width, height) in faces:
    #     cv2.rectangle(frame, (x, y), (x + width, y + height), (255, 0, 0), 3)

    cv2.imshow("Camera", frame)

    if cv2.waitKey(1) == ord('q'):
        break

out.release()
cap.release()
cv2.destroyAllWindows()
