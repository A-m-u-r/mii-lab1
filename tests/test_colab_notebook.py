import json
import unittest
from pathlib import Path


NOTEBOOK_PATH = Path(__file__).parents[1] / "lab1_faster_rcnn_colab.ipynb"


class ColabNotebookTests(unittest.TestCase):
    def test_notebook_contains_setup_source_upload_and_inline_result_cells(self):
        self.assertTrue(NOTEBOOK_PATH.exists())

        notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
        cell_sources = "\n".join("".join(cell["source"]) for cell in notebook["cells"])

        self.assertEqual(notebook["nbformat"], 4)
        self.assertIn("Faster R-CNN", cell_sources)
        self.assertIn("files.upload()", cell_sources)
        self.assertIn("run_inference(", cell_sources)
        self.assertIn("plt.show()", cell_sources)
        self.assertIn("git pull --ff-only", cell_sources)
        self.assertIn("fonts-dejavu-core", cell_sources)
        self.assertIn("importlib.reload", cell_sources)


if __name__ == "__main__":
    unittest.main()
