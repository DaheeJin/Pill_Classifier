import os
import cv2
import argparse
from ultralytics import YOLO

def load_class_to_category_map(mapping_path):
    class_to_category = {}
    with open(mapping_path, 'r', encoding='utf-8') as f:
        for class_id, line in enumerate(f):
            line = line.strip()
            if line:
                try:
                    category_id = int(line)
                    class_to_category[class_id] = category_id
                except ValueError:
                    print(f"⚠️ 잘못된 category_id 무시됨: '{line}'")
    return class_to_category

def crop_from_yolo_labels(image_dir, label_dir, output_dir, mapping_path, conf_thresh=0.95):
    os.makedirs(output_dir, exist_ok=True)
    class_to_category = load_class_to_category_map(mapping_path)
    image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    for img_file in image_files:
        stem = os.path.splitext(img_file)[0]
        label_path = os.path.join(label_dir, f"{stem}.txt")
        img_path = os.path.join(image_dir, img_file)

        if not os.path.exists(label_path):
            print(f"⚠️ 라벨 없음: {label_path}")
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
            class_id = int(class_id)
            category_id = class_to_category.get(class_id)
            if category_id is None:
                print(f"⚠️ category_id 없음: class_id {class_id}")
                continue

            x_center *= w
            y_center *= h
            box_w *= w
            box_h *= h
            x1 = int(max(0, x_center - box_w / 2))
            y1 = int(max(0, y_center - box_h / 2))
            x2 = int(min(w, x_center + box_w / 2))
            y2 = int(min(h, y_center + box_h / 2))

            crop = image[y1:y2, x1:x2]
            category_dir = os.path.join(output_dir, f"{category_id}")
            os.makedirs(category_dir, exist_ok=True)
            save_path = os.path.join(category_dir, f"{stem}_crop_{idx}.png")
            cv2.imwrite(save_path, crop)

        print(f"✅ YOLO 라벨 기반 크롭 완료: {img_file}")

def crop_from_model_inference(image_dir, yolo_weights, output_dir, conf_thresh=0.95):
    os.makedirs(output_dir, exist_ok=True)
    model = YOLO(yolo_weights)
    image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    for img_file in image_files:
        img_path = os.path.join(image_dir, img_file)
        image = cv2.imread(img_path)
        if image is None:
            print(f"❌ 이미지 로딩 실패: {img_path}")
            continue

        results = model.predict(source=img_path, conf=0.3, verbose=False)[0]
        for i, box in enumerate(results.boxes):
            cls_id = int(box.cls.item())
            conf = float(box.conf.item())
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            if conf < conf_thresh:
                crop = image[y1:y2, x1:x2]
                save_path = os.path.join(output_dir, f"{os.path.splitext(img_file)[0]}_crop_{i}_cls{cls_id}_conf{conf:.2f}.png")
                cv2.imwrite(save_path, crop)

        print(f"✅ 모델 기반 크롭 완료: {img_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLO 라벨 또는 모델 기반 crop")
    parser.add_argument("--image_dir", required=True)
    parser.add_argument("--label_dir", help="YOLO txt 라벨 경로 (라벨 기반 사용 시 필수)")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--mapping_path", help="class_id → category_id 매핑 파일 (라벨 기반 시 필수)")
    parser.add_argument("--yolo_weights", help="YOLO 모델 가중치 경로 (모델 추론 시 필수)")
    parser.add_argument("--use_model_inference", action="store_true", help="YOLO 모델 직접 추론으로 crop할지 여부")
    parser.add_argument("--conf_thresh", type=float, default=0.95, help="confidence threshold")

    args = parser.parse_args()

    if args.use_model_inference:
        if not args.yolo_weights:
            raise ValueError("YOLO 모델 기반 추론에는 --yolo_weights 인자가 필요합니다.")
        crop_from_model_inference(args.image_dir, args.yolo_weights, args.output_dir, args.conf_thresh)
    else:
        if not (args.label_dir and args.mapping_path):
            raise ValueError("YOLO 라벨 기반 크롭에는 --label_dir 과 --mapping_path 인자가 필요합니다.")
        crop_from_yolo_labels(args.image_dir, args.label_dir, args.output_dir, args.mapping_path, args.conf_thresh)
