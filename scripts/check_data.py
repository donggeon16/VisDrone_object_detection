from src.utils.constants import BASE, CLASSES
from src.datasets.visdrone import preprocess_annotation
from collections import Counter
from PIL import Image, ImageDraw
import hashlib
import csv

"""
visdrone annotation analyze & image check

"""

def check_size(area):
    if area < 1024:
        return "small"
    elif area < 9216:
        return "medium"
    else:
        return "large"


def analyze_annotation(mode):
    annotation_dir = BASE / "data" / "raw" / f"VisDrone2019-DET-{mode}" / "annotations"
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
    print(f"\n===== Dataset Summary ({mode}) =====")
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


def check_images(mode, save = False):

    image_dir = BASE / "data" / "raw" / f"VisDrone2019-DET-{mode}" / "images"
    images = sorted(image_dir.glob("*.jpg"))
    if not images:
        raise FileNotFoundError(
            f"No annotation files found: {image_dir}"
        )
    
    annotation_dir = BASE / "data" / "raw" / f"VisDrone2019-DET-{mode}" / "annotations"
    files = sorted(annotation_dir.glob("*.txt"))
    if not files:
        raise FileNotFoundError(
            f"No annotation files found: {annotation_dir}"
        )

    errors = []

    if save:
        output_dir = BASE / "reports" / f"bbox_samples_{mode}"
        output_dir.mkdir(parents=True, exist_ok=True)
        for image_path in images[:20]:

            file = annotation_dir / f"{image_path.stem}.txt"
            if not file.is_file():
                raise FileNotFoundError(
                    f"Missing annotation: {file.name}"
                )
            objects = preprocess_annotation(file, errors=errors)
            image = Image.open(image_path).convert("RGB")
            draw = ImageDraw.Draw(image)

            for obj in objects:
                if obj["score"] == 1:
                    draw.rectangle(obj["bbox"], outline="red", width=2)
                    draw.text((
                        obj["bbox"][0], 
                        obj["bbox"][1]),
                        CLASSES[obj["category"]],
                        fill="yellow"
                    )
            image.save(output_dir / image_path.name)
        errors = []

    bbox_errors = []

    for image_path in images:

        file = annotation_dir / f"{image_path.stem}.txt"
        if not file.is_file():
            raise FileNotFoundError(
                f"Missing annotation: {file.name}"
            )
        objects = preprocess_annotation(file, errors=errors)
        with Image.open(image_path) as image:
            img_w, img_h = image.size

        for obj in objects:
            x1, y1, x2, y2 = obj["bbox"]
            if not (0 <= x1 < x2 <= img_w and 0 <= y1 < y2 <= img_h):
                bbox_errors.append({
                    "file": image_path.name,
                    "bbox": obj["bbox"],
                    "image_size": (img_w, img_h),
                    "category": obj["category"],
                    "score": obj["score"],
                    "truncation": obj["truncation"]
                })

    print(f"\n===== Image Integrity ({mode}) =====")
    print(f"Total images: {len(images)}")
    print(f"Invalid annotation rows: {len(errors)}")
    print(f"Out-of-bounds boxes: {len(bbox_errors)}")

    for error in bbox_errors[:20]:
        print(error)


def file_hash(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()
    

def check_data_leakage():
    train_dir = BASE / "data" / "raw" / "VisDrone2019-DET-train" / "images"
    val_dir = BASE / "data" / "raw" / "VisDrone2019-DET-val" / "images"

    train_ids = {p.stem for p in train_dir.glob("*.jpg")}
    val_ids = {p.stem for p in val_dir.glob("*.jpg")}
    overlap = train_ids & val_ids

    train_hashes = {file_hash(p) for p in train_dir.glob("*.jpg")}
    val_hashes = {file_hash(p) for p in val_dir.glob("*.jpg")}
    overlap_hash = train_hashes & val_hashes

    print(f"\n===== Data Leakage =====")
    print(f"Duplicate image IDs: {len(overlap)}, image hashes: {len(overlap_hash)}")


def create_split_manifest():
    output_path = BASE / "data" / "splits" / "visdrone_det_split_manifest.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    split_ids = {}

    expected_counts = {
        "train": 6471,
        "val": 548
    }

    for mode in ("train", "val"):
        data_dir = BASE / "data" / "raw" / f"VisDrone2019-DET-{mode}"
        image_dir = data_dir / "images"
        annotation_dir = data_dir / "annotations"

        images = sorted(image_dir.glob("*.jpg"))
        annotations = sorted(annotation_dir.glob("*.txt"))

        image_ids = {p.stem for p in images}
        annotation_ids = {p.stem for p in annotations}

        # 이미지와 어노테이션 대응 검사
        if image_ids != annotation_ids:
            missing_annotations = image_ids - annotation_ids
            missing_images = annotation_ids - image_ids

            raise ValueError(
                f"{mode} split mismatch\n"
                f"Missing annotations: {sorted(missing_annotations)[:10]}\n"
                f"Missing images: {sorted(missing_images)[:10]}"
            )

        # 공식 Split 개수 검사
        if len(images) != expected_counts[mode]:
            raise ValueError(
                f"{mode}: Expected {expected_counts[mode]} images, "
                f"got {len(images)}"
            )

        split_ids[mode] = image_ids

        for image_path in images:
            annotation_path = annotation_dir / f"{image_path.stem}.txt"

            rows.append({
                "split": mode,
                "image_id": image_path.stem,
                "image_path": image_path.relative_to(BASE).as_posix(),
                "annotation_path": annotation_path.relative_to(BASE).as_posix()
            })

    # Train/Val 이미지 ID 중복 검사
    overlap = split_ids["train"] & split_ids["val"]

    if overlap:
        raise ValueError(
            f"Train/Val duplicate IDs: {sorted(overlap)[:10]}"
        )

    # CSV 저장
    fieldnames = [
        "split",
        "image_id",
        "image_path",
        "annotation_path"
    ]

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    # 저장된 CSV 다시 읽어서 검증
    with open(output_path, "r", newline="", encoding="utf-8") as f:
        saved_rows = list(csv.DictReader(f))

    counts = Counter(row["split"] for row in saved_rows)

    if len(saved_rows) != 7019:
        raise ValueError(
            f"Expected 7019 rows, got {len(saved_rows)}"
        )

    for mode, expected in expected_counts.items():
        if counts[mode] != expected:
            raise ValueError(
                f"{mode}: Expected {expected}, got {counts[mode]}"
            )

    saved_ids = [row["image_id"] for row in saved_rows]

    if len(saved_ids) != len(set(saved_ids)):
        raise ValueError("Duplicate image IDs in manifest")

    print("\n===== Split Manifest =====")
    print(f"Train: {counts['train']}")
    print(f"Val: {counts['val']}")
    print(f"Total: {len(saved_rows)}")
    print(f"Duplicate IDs: {len(overlap)}")
    print(f"Saved: {output_path}")

def main():
    modes = ["train", "val"]
    for mode in modes:
        analyze_annotation(mode)
        check_images(mode, save = False)
    check_data_leakage()
    create_split_manifest()

if __name__ == "__main__":
    main()