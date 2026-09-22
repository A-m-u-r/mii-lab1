"""Лабораторная работа №1: детекция объектов Faster R-CNN (не YOLO)."""

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class Detection:
    """Одна детекция в координатах исходного изображения (x1, y1, x2, y2)."""

    box: tuple[int, int, int, int]
    label: str
    score: float


RUSSIAN_LABELS = {
    "person": "человек",
    "bicycle": "велосипед",
    "car": "автомобиль",
    "motorcycle": "мотоцикл",
    "bus": "автобус",
    "truck": "грузовик",
    "bird": "птица",
    "cat": "кошка",
    "dog": "собака",
    "backpack": "рюкзак",
    "umbrella": "зонт",
    "handbag": "сумка",
    "tie": "галстук",
    "suitcase": "чемодан",
    "bottle": "бутылка",
    "cup": "чашка",
    "bowl": "миска",
    "chair": "стул",
    "couch": "диван",
    "dining table": "обеденный стол",
    "potted plant": "растение",
    "tv": "телевизор",
    "laptop": "ноутбук",
    "mouse": "мышь",
    "remote": "пульт",
    "keyboard": "клавиатура",
    "cell phone": "телефон",
    "book": "книга",
    "clock": "часы",
    "vase": "ваза",
}


def translate_label(label: str) -> str:
    """Возвращает русское название распространённого COCO-класса."""
    return RUSSIAN_LABELS.get(label, label)


def label_font_size(image_size: tuple[int, int]) -> int:
    """Подбирает читаемый размер подписи относительно меньшей стороны изображения."""
    return max(16, min(28, round(min(image_size) * 0.022)))


def load_label_font(font_size: int, image_font: Any = None) -> Any:
    """Загружает шрифт с кириллицей в Colab, Windows или локальном окружении."""
    if image_font is None:
        from PIL import ImageFont

        image_font = ImageFont

    font_paths = (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "C:/Windows/Fonts/arialbd.ttf",
        "DejaVuSans-Bold.ttf",
        "arialbd.ttf",
        "arial.ttf",
    )
    for font_path in font_paths:
        try:
            return image_font.truetype(font_path, font_size)
        except OSError:
            continue
    return image_font.load_default()


def _as_list(value: Any) -> list[Any]:
    """Преобразует список, NumPy-массив или Tensor в обычный Python-список."""
    return value.tolist() if hasattr(value, "tolist") else list(value)


def filter_predictions(
    prediction: Mapping[str, Any],
    label_names: Sequence[str],
    score_threshold: float,
) -> list[Detection]:
    """Отбрасывает слабые предсказания Faster R-CNN и добавляет названия классов."""
    boxes = _as_list(prediction["boxes"])
    labels = _as_list(prediction["labels"])
    scores = _as_list(prediction["scores"])

    detections = []
    for box, label_index, score in zip(boxes, labels, scores):
        score = float(score)
        if score < score_threshold:
            continue
        x1, y1, x2, y2 = (round(float(coordinate)) for coordinate in box)
        label_index = int(label_index)
        label = label_names[label_index] if label_index < len(label_names) else f"class_{label_index}"
        detections.append(Detection((x1, y1, x2, y2), label, score))
    return detections


def keep_labels(detections: Sequence[Detection], allowed_labels: set[str]) -> list[Detection]:
    """Оставляет детекции только заданных классов."""
    return [detection for detection in detections if detection.label in allowed_labels]


def draw_detections(image: Any, detections: Sequence[Detection]) -> Any:
    """Рисует красные рамки и читаемые подписи на копии изображения."""
    from PIL import ImageDraw

    annotated = image.copy()
    painter = ImageDraw.Draw(annotated)
    font_size = label_font_size(annotated.size)
    font = load_label_font(font_size)

    color = (220, 38, 38)
    line_width = max(3, font_size // 12)
    for detection in detections:
        x1, y1, x2, y2 = detection.box
        painter.rectangle((x1, y1, x2, y2), outline=color, width=line_width)
        label_text = f"{translate_label(detection.label)} {detection.score:.0%}"
        text_box = painter.textbbox((0, 0), label_text, font=font)
        text_height = text_box[3] - text_box[1]
        painter.text(
            (x1 + line_width, max(0, y1 - text_height - line_width)),
            label_text,
            fill=color,
            font=font,
        )
    return annotated


def preprocess_image(image: Any, tensorizer: Any = None) -> Any:
    """Приводит изображение к RGB и к float-тензору ``[C, H, W]`` в диапазоне [0, 1]."""
    if tensorizer is None:
        from torchvision.transforms.functional import to_tensor

        tensorizer = to_tensor
    return tensorizer(image.convert("RGB"))


def run_inference(
    model: Any,
    image: Any,
    label_names: Sequence[str],
    score_threshold: float,
    device: str,
    tensorizer: Any = None,
    no_grad: Any = None,
) -> list[Detection]:
    """Запускает Faster R-CNN для одного изображения и фильтрует его выход."""
    if no_grad is None:
        import torch

        no_grad = torch.no_grad

    tensor = preprocess_image(image, tensorizer=tensorizer)
    if hasattr(tensor, "to"):
        tensor = tensor.to(device)
    with no_grad():
        prediction = model([tensor])[0]
    return filter_predictions(prediction, label_names, score_threshold)


def load_detector(
    device: Any,
    model_builder: Any = None,
    default_weights: Any = None,
) -> tuple[Any, Sequence[str]]:
    """Загружает COCO-веса Faster R-CNN и переводит модель в ``eval``-режим."""
    if model_builder is None or default_weights is None:
        from torchvision.models.detection import (
            FasterRCNN_ResNet50_FPN_Weights,
            fasterrcnn_resnet50_fpn,
        )

        model_builder = fasterrcnn_resnet50_fpn
        default_weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT

    model = model_builder(weights=default_weights).to(device).eval()
    return model, default_weights.meta["categories"]


def create_argument_parser() -> argparse.ArgumentParser:
    """Создаёт интерфейс командной строки для запуска лабораторной работы."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path, help="путь к исходному изображению")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/lab1_detections.jpg"),
        help="куда сохранить изображение с детекциями",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.6,
        help="минимальная уверенность детекции от 0 до 1",
    )
    parser.add_argument("--no-show", action="store_true", help="не открывать окно с результатом")
    parser.add_argument(
        "--only-person",
        action="store_true",
        help="оставить на изображении только детекции людей",
    )
    return parser


if __name__ == "__main__":
    arguments = create_argument_parser().parse_args()
    if not 0 <= arguments.threshold <= 1:
        raise SystemExit("--threshold должен быть в диапазоне от 0 до 1")

    import torch
    from PIL import Image

    selected_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Устройство: {selected_device}")
    print("Загрузка Faster R-CNN ResNet-50 FPN с весами COCO...")
    detector, coco_labels = load_detector(selected_device)

    with Image.open(arguments.image) as opened_image:
        source_image = opened_image.convert("RGB")

    detected_objects = run_inference(
        model=detector,
        image=source_image,
        label_names=coco_labels,
        score_threshold=arguments.threshold,
        device=selected_device,
    )
    if arguments.only_person:
        detected_objects = keep_labels(detected_objects, {"person"})
    annotated_image = draw_detections(source_image, detected_objects)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    annotated_image.save(arguments.output)

    print(f"Найдено объектов: {len(detected_objects)}")
    for number, detection in enumerate(detected_objects, start=1):
        print(f"{number}. {detection.label}: {detection.score:.2f}, box={detection.box}")
    print(f"Изображение с детекциями сохранено: {arguments.output}")

    if not arguments.no_show:
        import matplotlib.pyplot as plt

        plt.figure(figsize=(12, 8))
        plt.imshow(annotated_image)
        plt.axis("off")
        plt.title("Faster R-CNN: детекции")
        plt.show()
