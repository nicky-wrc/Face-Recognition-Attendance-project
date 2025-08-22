import dlib
import cv2

def detect_faces(image_path):
    detector = dlib.get_frontal_face_detector()
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    print(f"เจอใบหน้า {len(faces)} คน")
    for i, face in enumerate(faces):
        x, y, w, h = (face.left(), face.top(), face.width(), face.height())
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        print(f"Face {i+1}: x={x}, y={y}, w={w}, h={h}")

    cv2.imshow("Faces", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
