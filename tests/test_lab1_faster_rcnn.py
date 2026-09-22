import unittest
from contextlib import nullcontext

from PIL import Image

from lab1_faster_rcnn import (
    Detection,
    create_argument_parser,
    draw_detections,
    keep_labels,
    filter_predictions,
    load_detector,
    label_font_size,
    load_label_font,
    preprocess_image,
    run_inference,
    translate_label,
)


class FilterPredictionsTests(unittest.TestCase):
    def test_keeps_only_detections_at_or_above_threshold(self):
        prediction = {
            "boxes": [[10.2, 20.7, 110.9, 220.1], [1, 2, 3, 4]],
            "labels": [1, 3],
            "scores": [0.91, 0.49],
        }

        detections = filter_predictions(
            prediction=prediction,
            label_names=["background", "person", "bicycle", "car"],
            score_threshold=0.5,
        )

        self.assertEqual(
            detections,
            [Detection(box=(10, 21, 111, 220), label="person", score=0.91)],
        )

    def test_keeps_requested_labels_only(self):
        detections = [
            Detection((0, 0, 10, 10), "person", 0.9),
            Detection((10, 10, 20, 20), "cup", 0.8),
            Detection((20, 20, 30, 30), "person", 0.7),
        ]

        people = keep_labels(detections, {"person"})

        self.assertEqual(
            people,
            [
                Detection((0, 0, 10, 10), "person", 0.9),
                Detection((20, 20, 30, 30), "person", 0.7),
            ],
        )


class DrawDetectionsTests(unittest.TestCase):
    def test_draws_on_a_copy_of_the_source_image(self):
        source = Image.new("RGB", (500, 500), "white")

        annotated = draw_detections(
            source,
            [Detection(box=(50, 100, 400, 400), label="person", score=0.9)],
        )

        self.assertIsNot(annotated, source)
        self.assertEqual(source.getpixel((50, 100)), (255, 255, 255))
        self.assertEqual(annotated.getpixel((50, 100)), (220, 38, 38))


class LabelTranslationTests(unittest.TestCase):
    def test_translates_common_coco_labels_and_keeps_unknown_labels(self):
        self.assertEqual(translate_label("person"), "человек")
        self.assertEqual(translate_label("cell phone"), "телефон")
        self.assertEqual(translate_label("unknown object"), "unknown object")


class LabelFontSizeTests(unittest.TestCase):
    def test_scales_label_font_for_large_photographs_with_safe_bounds(self):
        self.assertEqual(label_font_size((4000, 3000)), 28)
        self.assertEqual(label_font_size((640, 480)), 16)


class LabelFontTests(unittest.TestCase):
    def test_prefers_the_colab_dejavu_font_with_cyrillic_support(self):
        class FakeImageFont:
            attempts = []

            @classmethod
            def truetype(cls, path, size):
                cls.attempts.append((path, size))
                if path == "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf":
                    return "colab font"
                raise OSError

            @staticmethod
            def load_default():
                return "fallback font"

        font = load_label_font(21, image_font=FakeImageFont)

        self.assertEqual(font, "colab font")
        self.assertEqual(
            FakeImageFont.attempts[0],
            ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 21),
        )


class PreprocessImageTests(unittest.TestCase):
    def test_converts_an_input_image_to_rgb_before_tensorization(self):
        captured = {}

        def tensorizer(image):
            captured["mode"] = image.mode
            captured["size"] = image.size
            return "CHW float tensor"

        tensor = preprocess_image(Image.new("RGBA", (20, 10)), tensorizer=tensorizer)

        self.assertEqual(tensor, "CHW float tensor")
        self.assertEqual(captured, {"mode": "RGB", "size": (20, 10)})


class RunInferenceTests(unittest.TestCase):
    def test_passes_one_preprocessed_image_to_model_and_returns_detections(self):
        class FakeTensor:
            def to(self, device):
                self.device = device
                return self

        class FakeModel:
            def __call__(self, images):
                self.images = images
                return [{"boxes": [[1, 2, 30, 40]], "labels": [1], "scores": [0.8]}]

        tensor = FakeTensor()
        model = FakeModel()
        detections = run_inference(
            model=model,
            image=Image.new("RGB", (50, 50)),
            label_names=["background", "person"],
            score_threshold=0.5,
            device="cpu",
            tensorizer=lambda image: tensor,
            no_grad=lambda: nullcontext(),
        )

        self.assertEqual(model.images, [tensor])
        self.assertEqual(tensor.device, "cpu")
        self.assertEqual(detections, [Detection((1, 2, 30, 40), "person", 0.8)])


class LoadDetectorTests(unittest.TestCase):
    def test_loads_default_weights_and_sets_model_to_evaluation_mode(self):
        class FakeWeights:
            meta = {"categories": ["background", "person"]}

        class FakeModel:
            def eval(self):
                self.in_evaluation_mode = True
                return self

            def to(self, device):
                self.device = device
                return self

        model = FakeModel()
        received = {}

        def model_builder(*, weights):
            received["weights"] = weights
            return model

        loaded_model, labels = load_detector(
            device="cpu", model_builder=model_builder, default_weights=FakeWeights
        )

        self.assertIs(loaded_model, model)
        self.assertIs(received["weights"], FakeWeights)
        self.assertTrue(model.in_evaluation_mode)
        self.assertEqual(model.device, "cpu")
        self.assertEqual(labels, ["background", "person"])


class CommandLineTests(unittest.TestCase):
    def test_parser_accepts_image_path_and_uses_lab_defaults(self):
        arguments = create_argument_parser().parse_args(["photo.jpg"])

        self.assertEqual(str(arguments.image), "photo.jpg")
        self.assertEqual(arguments.output.as_posix(), "results/lab1_detections.jpg")
        self.assertEqual(arguments.threshold, 0.6)
        self.assertFalse(arguments.no_show)
        self.assertFalse(arguments.only_person)

    def test_parser_enables_person_only_mode(self):
        arguments = create_argument_parser().parse_args(["photo.jpg", "--only-person"])

        self.assertTrue(arguments.only_person)


if __name__ == "__main__":
    unittest.main()
