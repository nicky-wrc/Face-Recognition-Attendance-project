import sys
import argparse
from pathlib import Path
import cv2
import dlib

detector = dlib.get_frontal_face_detector()

def detect_faces(image_path: str, upsample: int = 1, show: bool = True, save: bool = True):
    img_path = Path(image_path)
    if not img_path.exists():
        raise FileNotFoundError(f"ไม่พบไฟล์ภาพ: {img_path.resolve()}")

    img = cv2.imread(str(img_path))
    if img is None:
        raise ValueError(f"อ่านรูปไม่สำเร็จ: {img_path}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = detector(gray, upsample)   

    print(f"เจอใบหน้า {len(faces)} คน")
    for i, face in enumerate(faces, start=1):
        x, y, w, h = face.left(), face.top(), face.width(), face.height()
        x, y = max(0, x), max(0, y)
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        print(f"Face {i}: x={x}, y={y}, w={w}, h={h}")

    if show:
        cv2.imshow("Faces - press any key to close", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    if save:
        out_path = img_path.with_name(img_path.stem + "_out" + img_path.suffix)
        cv2.imwrite(str(out_path), img)
        print(f"บันทึกรูปผลลัพธ์ที่: {out_path.resolve()}")

    return len(faces)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ตรวจจับใบหน้าด้วย dlib + OpenCV")
    parser.add_argument("image", nargs="?", default="test1.jpg", help="พาธไฟล์ภาพ (ค่าเริ่มต้น: test1.jpg)")
    parser.add_argument("--upsample", type=int, default=1, help="ระดับ upsample สำหรับตัวตรวจจับ (0/1/2)")
    parser.add_argument("--no-show", action="store_true", help="ไม่เปิดหน้าต่างแสดงผล")
    parser.add_argument("--no-save", action="store_true", help="ไม่บันทึกรูปผลลัพธ์")
    args = parser.parse_args()

    detect_faces(
        image_path=args.image,
        upsample=args.upsample,
        show=not args.no_show,
        save=not args.no_save
    )
