from src.utils.constants import BASE, CLASSES
from src.datasets.visdrone import preprocess_annotation
from collections import Counter

"""
visdrone annotation analyze & check

"""

def check_size(area):
    if area < 1024:
        return "small"
    elif area < 9216:
        return "medium"
    else:
        return "large"

def main():
    annotation_dir = BASE / "data" / "raw" / "VisDrone2019-DET-train" / "annotations"
    files = sorted(annotation_dir.glob("*.txt"))
    if not files:
        raise FileNotFoundError(
            f"No annotation files found: {annotation_dir}"
        )
    
    total_annotation_files = len(files)
    total_objects = 0
    total_valid_objects = 0

    class_counts = Counter()
    size_counts = Counter()
    objects_per_image = Counter()
    errors = []

    for file in files:
        objects = preprocess_annotation(file, errors=errors)
        total_objects += len(objects)
        objects_per_image[file.stem] = 0

        for obj in objects:
            category = obj["category"]
            score = obj["score"]
            area = obj["area"]
            if 0 < category < 11 and score == 1:
                total_valid_objects += 1
                class_counts[category] += 1
                size_counts[(category, check_size(area))] += 1
                objects_per_image[file.stem] += 1

    # 1. 전체 통계
    print("\n===== Dataset Summary =====")
    print(f"Total annotation files: {total_annotation_files}")
    print(f"Total annotation rows: {total_objects}")
    print(f"Total valid objects: {total_valid_objects}")
    print(f"Excluded rows: {total_objects - total_valid_objects}")
    print(f"Total invalid annotations: {len(errors)}")

    for error in errors[:20]:
        print(error)

    # 2. 클래스별 객체 개수
    print("\n===== Class Distribution =====")

    for category in range(1, 11):
        class_name = CLASSES[category]
        count = class_counts[category]

        print(f"{class_name:<20}: {count}")

    # 3. 클래스별 객체 크기 분포
    print("\n===== Size Distribution =====")

    print(
        f"{'Class':<20}"
        f"{'Small':>10}"
        f"{'Medium':>10}"
        f"{'Large':>10}"
        f"{'Total':>10}"
    )

    for category in range(1, 11):
        class_name = CLASSES[category]

        small = size_counts[(category, "small")]
        medium = size_counts[(category, "medium")]
        large = size_counts[(category, "large")]

        total = small + medium + large

        print(
            f"{class_name:<20}"
            f"{small:>10}"
            f"{medium:>10}"
            f"{large:>10}"
            f"{total:>10}"
        )

    # 4. 이미지당 객체 수
    counts = list(objects_per_image.values())

    mean_count = sum(counts) / len(counts)
    min_count = min(counts)
    max_count = max(counts)

    print("\n===== Objects per Image =====")
    print(f"Mean: {mean_count:.2f}")
    print(f"Min : {min_count}")
    print(f"Max : {max_count}")

    # 5. 전체 객체 크기 분포
    print("\n===== Overall Size Distribution =====")

    for size in ("small", "medium", "large"):
        count = sum(
            size_counts[(category, size)]
            for category in range(1, 11)
        )

        ratio = (
            count / total_valid_objects * 100
            if total_valid_objects > 0 else 0
        )

        print(f"{size:<10}: {count:>8} ({ratio:.2f}%)")



if __name__ == "__main__":
    main()