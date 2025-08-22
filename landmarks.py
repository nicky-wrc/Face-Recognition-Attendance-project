import dlib
import cv2

def detect_landmarks(image_path):
    img = cv2.imread(image_path)
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    faces = detector(img)

    for face in faces:
        landmarks = predictor(img, face)
        for n in range(68):
            x, y = landmarks.part(n).x, landmarks.part(n).y
            cv2.circle(img, (x, y), 1, (0, 0, 255), -1)
    cv2.imshow("Landmarks", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
