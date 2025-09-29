# capture_dataset.py
import cv2, os, time, csv
from datetime import datetime

OUT_DIR = "dataset"
CLASSES = {"1":"occupied", "0":"empty"}
os.makedirs(OUT_DIR, exist_ok=True)
for v in CLASSES.values():
    os.makedirs(os.path.join(OUT_DIR, v), exist_ok=True)

cap = cv2.VideoCapture(0)  # 웹캠 장치 번호 필요시 변경
if not cap.isOpened():
    raise RuntimeError("웹캠을 열 수 없습니다. 번호/권한 확인하세요.")

csvfile = open(os.path.join(OUT_DIR, "labels.csv"), "a", newline="")
writer = csv.writer(csvfile)
writer.writerow(["filename","label","timestamp"])

count = {k: len(os.listdir(os.path.join(OUT_DIR,v))) for k,v in CLASSES.items()}

print("키: 1=occupied, 0=empty, q=quit")
while True:
    ret, frame = cap.read()
    if not ret:
        break
    # 필요한 경우 ROI(책상 영역) 미리 자르기: frame = frame[y1:y2, x1:x2]
    cv2.imshow("capture", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    if chr(key) in CLASSES:
        label = CLASSES[chr(key)]
        ts = datetime.utcnow().isoformat()
        fname = f"{label}_{int(time.time()*1000)}.jpg"
        fpath = os.path.join(OUT_DIR, label, fname)
        cv2.imwrite(fpath, frame)
        writer.writerow([fpath, label, ts])
        csvfile.flush()
        print("Saved:", fpath)
cap.release()
cv2.destroyAllWindows()
csvfile.close()
