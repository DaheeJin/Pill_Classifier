
import os
import cv2

def crop_yolo_bboxes(image_dir, label_dir, output_dir):
    """
    YOLO format 라벨(txt)을 기반으로 bounding box 영역을 크롭하여
    클래스별 디렉토리에 저장합니다.

    Args:
        image_dir (str): 원본 이미지 경로
        label_dir (str): YOLO 형식 라벨(txt) 경로
        output_dir (str): 크롭된 이미지 저장 경로 (클래스별로 정리됨)
    """
    os.makedirs(output_dir, exist_ok=True)
    image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    for img_file in image_files:
        stem = os.path.splitext(img_file)[0]
        label_file = stem + ".txt"
        label_path = os.path.join(label_dir, label_file)
        img_path = os.path.join(image_dir, img_file)

        if not os.path.exists(label_path):
            print(f"⚠️ 라벨 파일 없음: {label_path}")
            continue

        image = cv2.imread(img_path)
        if image is None:
            print(f"❌ 이미지 로딩 실패: {img_path}")
            continue

        h, w, _ = image.shape
        with open(label_path, 'r') as f:
            lines = f.readlines()

        for idx, line in enumerate(lines):
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            class_id, x_center, y_center, box_w, box_h = map(float, parts)

            x_center *= w
            y_center *= h
            box_w *= w
            box_h *= h

            x1 = int(x_center - box_w / 2)
            y1 = int(y_center - box_h / 2)
            x2 = int(x_center + box_w / 2)
            y2 = int(y_center + box_h / 2)

            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            cropped = image[y1:y2, x1:x2]

            class_dir = os.path.join(output_dir, f"{int(class_id)}")
            os.makedirs(class_dir, exist_ok=True)
            save_path = os.path.join(class_dir, f"{stem}_crop_{idx}.png")
            cv2.imwrite(save_path, cropped)

        print(f"✅ 크롭 완료: {img_file}")

# ✅ Colab 또는 다른 스크립트에서 아래처럼 호출하세요
# crop_yolo_bboxes(
#     image_dir="/content/yolo_dataset/images/train",
#     label_dir="/content/yolo_dataset/labels/train",
#     output_dir="/content/cropped_pills"
# )

