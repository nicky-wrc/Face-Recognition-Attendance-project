import dlib
import cv2

sp = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
facerec = dlib.face_recognition_model_v1("dlib_face_recognition_resnet_model_v1.dat")

def recognize_faces(image_path):
    img = cv2.imread(image_path)
    detector = dlib.get_frontal_face_detector()
    faces = detector(img, 1)

    for face in faces:
        shape = sp(img, face)
        face_descriptor = facerec.compute_face_descriptor(img, shape)
        print("Face Descriptor:", list(face_descriptor)[:5], "...") 

    if not faces:
        print("ไม่พบใบหน้าในภาพ")
