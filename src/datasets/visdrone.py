from src.utils.constants import BASE, CLASSES


def preprocess_annotation(annotation_path):
    answer = []

    with open(annotation_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()

            # 빈 줄 건너뛰기
            if not line:
                continue

            # 필드 개수 검사
            raw_values = line.split(",")

            if len(raw_values) != 8:
                raise ValueError(
                    f"Line {line_num}: Expected 8 fields, got {len(raw_values)}"
                )

            # 정수 변환
            try:
                x, y, w, h, score, category, truncation, occlusion = map(
                    int, raw_values
                )
            except ValueError as e:
                raise ValueError(
                    f"Line {line_num}: Invalid integer value"
                ) from e

            # Bounding Box 검사
            if w <= 0 or h <= 0:
                raise ValueError(
                    f"Line {line_num}: Invalid width or height"
                )

            # 클래스 ID 검사 (CLASSES가 dict 또는 list인 경우)
            if isinstance(CLASSES, dict):
                valid_category = category in CLASSES
            else:
                valid_category = 0 <= category < len(CLASSES)

            if not valid_category:
                raise ValueError(
                    f"Line {line_num}: Invalid category {category}"
                )

            # Bounding Box 좌표 변환
            bbox_xyxy = (x, y, x + w, y + h)
            area = w * h

            answer.append({
                "category": category,
                "bbox": bbox_xyxy,
                "area": area,
                "score": score,
                "truncation": truncation,
                "occlusion": occlusion
            })

    return answer


def main():
    path = BASE/"data"/"raw"/"VisDrone2019-DET-train"/"annotations"
    data = preprocess_annotation(path/"0000002_00005_d_0000014.txt")
    for obj in data[:5]:
        print(obj[0], obj[1], obj[2])

if __name__ == "__main__":
    main()