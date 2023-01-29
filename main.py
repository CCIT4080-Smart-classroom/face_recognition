import face_recognition
import cv2
import numpy as np
import os
import requests
import time
import datetime
        
# Get a reference to webcam #1
video_capture = cv2.VideoCapture(1)

training_face_encodings = []
training_face_ids = []

# Load pictures from the training folder
for file in os.listdir("training"):
    if file.endswith(".jpg"):
        img = face_recognition.load_image_file(f"training/{file}")
        training_face_encodings.append(face_recognition.face_encodings(img)[0])
        training_face_ids.append(os.path.splitext(file)[0])

print(f'Training ids: {training_face_ids}')

# Initialize some variables
face_locations = []
face_encodings = []
face_names = []
frame_count = 0
detected_ids = {}

while True:
    # Grab a single frame of video
    ret, frame = video_capture.read()

    # Only process every 30 frames of video
    if frame_count == 0:
        # Resize frame of video to 1/4 size for faster face recognition processing
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

        # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        rgb_small_frame = small_frame[:, :, ::-1]
        
        # Find all the faces and face encodings in the current frame of video
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_ids = []
        for face_encoding in face_encodings:
            # See if the face is a match for the known face(s)
            matches = face_recognition.compare_faces(training_face_encodings, face_encoding)
            id = "Unknown"

            # # If a match was found in known_face_encodings, just use the first one.
            # if True in matches:
            #     first_match_index = matches.index(True)
            #     name = training_face_ids[first_match_index]

            # Or instead, use the known face with the smallest distance to the new face
            face_distances = face_recognition.face_distance(training_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                id = training_face_ids[best_match_index]

            face_ids.append(id)

    frame_count = (frame_count+1)%30


    # Display the results
    for (top, right, bottom, left), id in zip(face_locations, face_ids):
        # Scale back up face locations since the frame we detected in was scaled to 1/4 size
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Draw a label with a name below the face
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, id, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

        if id!="Unknown" and id not in detected_ids or int(time.time())-detected_ids[id]>60:
            detected_ids.update({id: int(time.time())})
            print(f"{datetime.datetime.now()}: detected", id)
            resp = requests.post("https://api.ccit4080.tylerl.cyou/student/checkin", json={"student_id": int(id)})
            print(resp.text)


    # Display the resulting image
    cv2.imshow('Video', frame)

    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()