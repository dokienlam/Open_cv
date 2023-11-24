import face_recognition
import cv2
import os
import glob
import numpy as np
import tkinter as tk

known_face_encodings = []
known_face_names = []
frame_resizing = 0.25

def load_encoding_images(images_path):
    images_path = glob.glob(
        os.path.join(
            "E:/code/Recognition_face",
            "*.*"
        )
    )
    print("{} encoding images found.".format(len(images_path)))

    for img_path in images_path:
        img = cv2.imread(img_path)
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        basename = os.path.basename(img_path)
        (filename, ext) = os.path.splitext(basename)
        img_encoding = face_recognition.face_encodings(rgb_img)[0]
        known_face_encodings.append(img_encoding)
        known_face_names.append(filename)
        # .known_face_names.append(basename)
    print("Encoding images loaded")

def detect_known_faces(frame, threshold=0.7):
    small_frame = cv2.resize(
        frame, (0, 0), fx=frame_resizing, fy=frame_resizing
    )
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(
        rgb_small_frame, face_locations
    )
    # print(len(face_encodings))

    face_names = []
    confidence_scores = []  

    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(
            known_face_encodings, face_encoding, tolerance=threshold
        )
        name = "Unknown"
        confidence = None  

        if True in matches:
            match_indices = [i for i, match in enumerate(matches) if match]
            distances = face_recognition.face_distance(
                known_face_encodings, face_encoding
            )
            best_match_index = np.argmin(distances)

            if distances[best_match_index] <= threshold:
                name = known_face_names[best_match_index]
                confidence = 1 - distances[best_match_index]

        face_names.append(f"{name}: {confidence:.2f}")
        confidence_scores.append(confidence)

    face_locations = np.array(face_locations)
    face_locations = face_locations / frame_resizing
    return face_locations.astype(int), face_names, confidence_scores

def Input_data():
    
    a = name_entry.get()  
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    cap = cv2.VideoCapture(0)
    save_folder = "E:/code/Recognition_face"
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    while True:
        ret, frame = cap.read()

        faces = face_cascade.detectMultiScale(frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        if not ret:
            break

        for (x, y, w, h) in faces:
    
            # face_roi = frame[y:y+h, x:x+w]

            image_name = f"{a}.jpg"
            save_path = os.path.join(save_folder, image_name)
            cv2.imwrite(save_path, frame)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
        cv2.imshow('Face_recognition', frame)

        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def face_recog():

    load_encoding_images("E:/code/Recognition_face")

    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()

        face_locations, face_names, confidence_scores = detect_known_faces(frame)
        for face_loc, name in zip(face_locations, face_names):
            y1, x2, y2, x1 = face_loc[0], face_loc[1], face_loc[2], face_loc[3]

            cv2.putText(
                frame, name, (x1, y1 - 10), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 200), 2
            )
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 200), 4)

        cv2.imshow("Frame", frame)
        if cv2.waitKey(1) == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

root = tk.Tk()

label = tk.Label(root, text="Name:")
label.pack()


name_entry = tk.Entry(root)
name_entry.pack()

select_button = tk.Button(root, text="Input data", command=Input_data)
select_button.pack(pady=20)

detect_button = tk.Button(root, text="Detect Faces", command=face_recog)
detect_button.pack(pady=10)

root.mainloop()