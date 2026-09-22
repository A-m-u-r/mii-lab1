# Лабораторная работа №1 — детекция объектов без YOLO

В файле `lab1_faster_rcnn.py` выполнен полный путь инференса модели
**Faster R-CNN ResNet-50 FPN** из TorchVision с предобученными весами COCO.
Это двухэтапный детектор, а не YOLO.

## Что делает программа

1. Считывает указанное изображение через Pillow.
2. Конвертирует его в `RGB` и `float`-тензор размера `[C, H, W]` со значениями
   `[0, 1]`. Этот тензор передаётся в нейросеть.
3. Загружает Faster R-CNN в режиме `eval()` и запускает инференс без градиентов.
4. Отбрасывает предсказания с уверенностью ниже порога.
5. Рисует на исходном изображении прямоугольники, классы и уверенности,
   сохраняет картинку и показывает её через Matplotlib.

В терминале также выводятся количество объектов, класс, confidence и координаты
каждой рамки. При запуске в Google Colab `plt.show()` выводит изображение прямо
под ячейкой.

## Локальный запуск

Нужен Python 3.10+. В Google Colab PyTorch и TorchVision обычно уже установлены.

```powershell
py -3.14 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements-lab1.txt
python lab1_faster_rcnn.py путь\к\изображению.jpg
```

Результат появится в `results/lab1_detections.jpg` и в окне Matplotlib.
Для запуска без окна используйте `--no-show`; порог можно изменить, например,
`--threshold 0.7`.

```powershell
python lab1_faster_rcnn.py image.jpg --output results\my_result.jpg --threshold 0.7
```

На групповой фотографии рамок может быть слишком много. Для задачи обнаружения
людей используйте только класс `person` и порог 0.8:

```powershell
python lab1_faster_rcnn.py image.jpg --only-person --threshold 0.8
```

Подписи рисуются увеличенным красным шрифтом и переводятся на русский для
распространённых классов. Модель определяет класс объекта (например,
«человек»), но не распознаёт личность конкретного человека.

При первом запуске TorchVision автоматически скачает веса Faster R-CNN.

## Google Colab

В первой ячейке Colab выполните:

```python
!git clone https://github.com/RadaevKirill/danger-zone-detection.git
%cd danger-zone-detection
!pip install -q -r requirements-lab1.txt

from google.colab import files
uploaded = files.upload()  # выберите .jpg или .png
image_name = next(iter(uploaded))
```

Во второй ячейке запустите инференс и выведите итоговое изображение:

```python
!python lab1_faster_rcnn.py "$image_name" --no-show

from IPython.display import Image, display
display(Image(filename="results/lab1_detections.jpg"))
```

## Проверки

```powershell
python -m unittest tests/test_lab1_faster_rcnn.py -v
```

Тесты проверяют фильтрацию выходов модели, препроцессинг, передачу тензора в
модель, режим инференса, отрисовку и параметры CLI.
