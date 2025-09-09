import argparse
from pathlib import Path
import cv2
import dlib

PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat"

detector = dlib.get_frontal_face_detector()

def detect_landmarks(
    image_path: str,
    predictor_path: str = PREDICTOR_PATH,
    upsample: int = 1,
    show: bool = True,
    save: bool = True,
):
    img_path = Path(image_path)
    if not img_path.exists():
        raise FileNotFoundError(f"ไม่พบไฟล์ภาพ: {img_path.resolve()}")

    if not Path(predictor_path).exists():
        raise FileNotFoundError(f"ไม่พบไฟล์โมเดล: {Path(predictor_path).resolve()}")

    img = cv2.imread(str(img_path))
    if img is None:
        raise ValueError(f"อ่านรูปไม่สำเร็จ: {img_path}")

    out = img.copy()

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = detector(gray, upsample)
    print(f"เจอใบหน้า {len(faces)} คน")

    predictor = dlib.shape_predictor(predictor_path)

    for i, face in enumerate(faces, start=1):
        x1, y1, x2, y2 = face.left(), face.top(), face.right(), face.bottom()
        x1, y1 = max(0, x1), max(0, y1)
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)

        shape = predictor(img, face)
        count = 0
        for n in range(68):
            px, py = shape.part(n).x, shape.part(n).y
            cv2.circle(out, (px, py), 1, (0, 0, 255), -1)
            count += 1
        print(f"Face {i}: วาด landmark {count} จุด")

    if show:
        cv2.imshow("Landmarks (กดปุ่มใดก็ได้เพื่อปิด)", out)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    if save:
        out_path = img_path.with_name(img_path.stem + "_landmarks" + img_path.suffix)
        cv2.imwrite(str(out_path), out)
        print(f"บันทึกผลลัพธ์ที่: {out_path.resolve()}")

    return len(faces)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ตรวจจับจุดใบหน้า (68 จุด) ด้วย dlib + OpenCV")
    parser.add_argument("image", nargs="?", default="test1.jpg", help="พาธไฟล์ภาพ (ค่าเริ่มต้น: test1.jpg)")
    parser.add_argument("--predictor", default=PREDICTOR_PATH, help="พาธไปยัง shape_predictor_68_face_landmarks.dat")
    parser.add_argument("--upsample", type=int, default=1, help="ระดับ upsample สำหรับตัวตรวจจับ (0/1/2)")
    parser.add_argument("--no-show", action="store_true", help="ไม่เปิดหน้าต่างแสดงผล")
    parser.add_argument("--no-save", action="store_true", help="ไม่บันทึกรูปผลลัพธ์")

    args = parser.parse_args()
    detect_landmarks(
        image_path=args.image,
        predictor_path=args.predictor,
        upsample=args.upsample,
        show=not args.no_show,
        save=not args.no_save,
    )
