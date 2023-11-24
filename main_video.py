import cv2
from simple_facerec import SimpleFacerec
import tkinter as tk
sfr = SimpleFacerec()
# sfr.main()
root = tk.Tk()
root.title("Ứng dụng chạy model")
name_label = tk.Label(root, text="Nhập tên: ")
name_label.pack()
name_entry = tk.Entry(root)
name_entry.pack()
def run_model():
    sfr.main()
# Tạo nút để chạy model
run_button = tk.Button(root, text="Chạy model", command=run_model)
run_button.pack()
# Bắt đầu vòng lặp giao diện
root.mainloop()


sfr.load_encoding_images("E:/code/Recognition_face")

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    face_locations, face_names, confidence_scores = sfr.detect_known_faces(frame)
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

