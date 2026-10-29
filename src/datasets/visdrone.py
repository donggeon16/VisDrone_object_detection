
from src.utils.constants import BASE, CLASSES


def preprocess_annotation(annotation_path, errors=None):
    answer = []

    with open(annotation_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                # 필드 개수 검사
                values = line.split(",")

                if len(values) != 8:
                    raise ValueError(
                        f"Expected 8 fields, got {len(values)}"
                    )

                # 정수 변환
                x, y, w, h, score, category, truncation, occlusion = map(int, values)

                # Bounding Box 검사
                if w <= 0 or h <= 0:
                    raise ValueError(
                        f"Invalid w={w} or h={h}"
                    )

                # 클래스 ID 검사
                valid_category = category in CLASSES

                if not valid_category:
                    raise ValueError(
                        f"Invalid category {category}"
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

            except ValueError as e:
                message = f"{annotation_path}:{line_num} - {e}"

                if errors is None:
                    raise ValueError(message) from e

                errors.append(message)
                continue

    return answer


def main():
    path = BASE / "data" / "raw" / "VisDrone2019-DET-train" / "annotations"

    errors = []

    data = preprocess_annotation(
        path / "0000002_00005_d_0000014.txt",
        errors=errors
    )

    for obj in data[:5]:
        print(obj["category"], obj["bbox"], obj["area"])

    print(f"Invalid annotations: {len(errors)}")

    for error in errors:
        print(error)


if __name__ == "__main__":
    main()
