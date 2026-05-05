---
jupyter:
  accelerator: GPU
  colab:
    collapsed_sections:
    - \_wDYXdHKevjQ
    - PW9WHjcoevjX
    gpuType: T4
  kernelspec:
    display_name: Python 3
    name: python3
  language_info:
    name: python
  nbformat: 4
  nbformat_minor: 0
---

::: {.cell .markdown id="_wDYXdHKevjQ"}

------------------------------------------------------------------------

## Stage 0 --- Install Dependencies & Mount Drive {#stage-0--install-dependencies--mount-drive}
:::

::: {.cell .code colab="{\"base_uri\":\"https://localhost:8080/\"}" id="O0cEsIENevjR" outputId="dd77b3de-79a4-4c08-8a2c-72d9b68ce829"}
``` python
!pip install -q scikit-image opencv-python-headless matplotlib seaborn scikit-learn

from google.colab import drive
drive.mount('/content/drive')

print('Drive mounted and dependencies ready.')
```

::: {.output .stream .stdout}
    Drive already mounted at /content/drive; to attempt to forcibly remount, call drive.mount("/content/drive", force_remount=True).
    Drive mounted and dependencies ready.
:::
:::

::: {.cell .code colab="{\"base_uri\":\"https://localhost:8080/\"}" collapsed="true" id="c7766c30" outputId="e22a6262-6640-4e12-d4f2-d3ec7fd0ae93"}
``` python
MY_DRIVE_FOLDER = '/content/drive/MyDrive/capstone_dataset'
import os
if os.path.exists(MY_DRIVE_FOLDER):
    print(f'Folder found at: {MY_DRIVE_FOLDER}')

else:
    print(f'Folder not found at: {MY_DRIVE_FOLDER}. Please check the path.')
```

::: {.output .stream .stdout}
    Folder found at: /content/drive/MyDrive/capstone_dataset
:::
:::

::: {.cell .code colab="{\"base_uri\":\"https://localhost:8080/\"}" id="TmOAhYpKevjV" outputId="4c6b6b58-070f-4c2d-eb75-7495041d5059"}
``` python
import shutil
import random
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import MobileNetV2
from sklearn.metrics import classification_report, confusion_matrix
from skimage.morphology import skeletonize
from skimage.measure import label, regionprops
import warnings
warnings.filterwarnings('ignore')

# ── CONFIG ──────────────────────────────────────────────────────────────────
SOURCE_DIR   = '/content/drive/MyDrive/crack_dataset'        #original dataset
SPLIT_DIR    = '/content/drive/MyDrive/crack_dataset_split'  #split folder
MODEL_PATH   = '/content/drive/MyDrive/crack_classifier.h5'
CSV_PATH     = '/content/drive/MyDrive/crack_analysis_results.csv'

IMG_SIZE     = (224, 224)
BATCH_SIZE   = 32
EPOCHS_HEAD  = 20    # train top layers only
EPOCHS_FINE  = 10    # fine-tune unfrozen base layers
SEED         = 42
PIXELS_PER_CM = None

CLASS_NAMES = [
    'deck_cracked', 'deck_uncracked', 'pavement_cracked', 'pavement_uncracked',
    'wall_cracked', 'wall_uncracked']

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

print('Config loaded.')
print(f'TensorFlow version: {tf.__version__}')
```

::: {.output .stream .stdout}
    Config loaded.
    TensorFlow version: 2.20.0
:::
:::

::: {.cell .markdown id="PW9WHjcoevjX"}

------------------------------------------------------------------------

## Stage 1 --- Dataset Split (70 / 15 / 15) {#stage-1--dataset-split-70--15--15}
:::

::: {.cell .code colab="{\"base_uri\":\"https://localhost:8080/\"}" id="ndKh0TOhevjZ" outputId="123440b3-6335-4e59-88da-daf144bfbaa3"}
``` python
def split_dataset(source_dir, output_dir, ratios=(0.70, 0.15, 0.15)):

    assert abs(sum(ratios) - 1.0) < 1e-6, 'Ratios must sum to 1.0'
    assert os.path.exists(source_dir), 'Source directory does not exist'
    os.makedirs(output_dir, exist_ok=True)

    classes = [d for d in sorted(os.listdir(source_dir))
        if os.path.isdir(os.path.join(source_dir, d))]
    print(f'Found classes: {classes}\n')

    split_names = ['train', 'val', 'test']
    summary = {}

    for cls in classes:
        cls_path = os.path.join(source_dir, cls)
        images   = [f for f in os.listdir(cls_path)
                    if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        random.shuffle(images)

        n       = len(images)
        n_train = int(n * ratios[0])
        n_val   = int(n * ratios[1])

        splits = {'train': images[:n_train],
            'val'  : images[n_train : n_train + n_val],
            'test' : images[n_train + n_val:]}

        for split, files in splits.items():
            dest = os.path.join(output_dir, split, cls)
            os.makedirs(dest, exist_ok=True)
            for f in files:
                shutil.copy(os.path.join(cls_path, f), os.path.join(dest, f))

        summary[cls] = {s: len(splits[s]) for s in split_names}
        print(f'  {cls:<25} train={splits["train"].__len__():>4}  '
              f'val={splits["val"].__len__():>4}  test={splits["test"].__len__():>4}')

    print('\nDataset split complete.')
    return summary


if not os.path.exists(SPLIT_DIR):
    split_summary = split_dataset(SOURCE_DIR, SPLIT_DIR)
else:
    print(' Split folder already exists — skipping split. Delete it to re-split.')
```

::: {.output .stream .stdout}
     Split folder already exists — skipping split. Delete it to re-split.
:::
:::

::: {.cell .markdown id="q9zTpEtmevja"}

------------------------------------------------------------------------

## Stage 2 --- Data Loading & Augmentation {#stage-2--data-loading--augmentation}
:::

::: {.cell .code colab="{\"base_uri\":\"https://localhost:8080/\"}" collapsed="true" id="VXDH-L-3evjc" outputId="033eb7a8-5648-4eff-f7a2-3603f94bf080"}
``` python
AUTOTUNE = tf.data.AUTOTUNE

train_ds = tf.keras.utils.image_dataset_from_directory(
    os.path.join(SPLIT_DIR, 'train'),
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode='categorical',
    seed=SEED)

val_ds = tf.keras.utils.image_dataset_from_directory(
    os.path.join(SPLIT_DIR, 'val'),
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode='categorical',
    seed=SEED)

test_ds = tf.keras.utils.image_dataset_from_directory(
    os.path.join(SPLIT_DIR, 'test'),
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode='categorical',
    seed=SEED,
    shuffle=False)

CLASS_NAMES = train_ds.class_names
NUM_CLASSES = len(CLASS_NAMES)
print(f'Classes ({NUM_CLASSES}): {CLASS_NAMES}')
```

::: {.output .stream .stdout}
    Found 39261 files belonging to 6 classes.
    Found 8411 files belonging to 6 classes.
    Found 8420 files belonging to 6 classes.
    Classes (6): ['deck_cracked', 'deck_uncracked', 'pavement_cracked', 'pavement_uncracked', 'wall_cracked', 'wall_uncracked']
:::
:::

::: {.cell .code colab="{\"base_uri\":\"https://localhost:8080/\"}" collapsed="true" id="tyMz57Ejevje" outputId="f42dd17e-acab-44e8-fb7a-83696aa2dff4"}
``` python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# ── Data Augmentation (applied only to training set) ────────────────────────
augmentation = tf.keras.Sequential([
    layers.RandomFlip('horizontal_and_vertical'),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
    layers.RandomBrightness(0.1),], name='augmentation')

normalization = layers.Rescaling(1./255)

def prepare_train(ds):
    return (ds
            .map(lambda x, y: (augmentation(x, training=True), y), num_parallel_calls=AUTOTUNE)
            .map(lambda x, y: (normalization(x), y), num_parallel_calls=AUTOTUNE)
            .cache().shuffle(1000).prefetch(AUTOTUNE))

def prepare_eval(ds):
    return (ds
            .map(lambda x, y: (normalization(x), y), num_parallel_calls=AUTOTUNE)
            .cache().prefetch(AUTOTUNE))

train_ds = prepare_train(train_ds)
val_ds   = prepare_eval(val_ds)
test_ds  = prepare_eval(test_ds)

print('Data pipelines ready.')
```

::: {.output .stream .stdout}
    Data pipelines ready.
:::
:::

::: {.cell .markdown id="G59C87QHevjg"}

------------------------------------------------------------------------

## Stage 3 --- Model: Transfer Learning with MobileNetV2 {#stage-3--model-transfer-learning-with-mobilenetv2}
:::

::: {.cell .code colab="{\"base_uri\":\"https://localhost:8080/\",\"height\":380}" collapsed="true" id="ICqhAlJCevjg" outputId="ada39b61-4353-420f-cb9c-fc53b1f9f109"}
``` python
def build_model(num_classes):
    base_model = MobileNetV2(
        input_shape=(*IMG_SIZE, 3),
        include_top=False,
        weights='imagenet'
    )
    base_model.trainable = False

    inputs  = keras.Input(shape=(*IMG_SIZE, 3))
    x       = base_model(inputs, training=False)
    x       = layers.GlobalAveragePooling2D()(x)
    x       = layers.Dense(128, activation='relu')(x)
    x       = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = keras.Model(inputs, outputs)
    return model, base_model

model, base_model = build_model(NUM_CLASSES)
model.summary()
```

::: {.output .display_data}
```{=html}
<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold">Model: "functional_1"</span>
</pre>
```
:::

::: {.output .display_data}
```{=html}
<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┓
┃<span style="font-weight: bold"> Layer (type)                    </span>┃<span style="font-weight: bold"> Output Shape           </span>┃<span style="font-weight: bold">       Param # </span>┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━┩
│ input_layer_2 (<span style="color: #0087ff; text-decoration-color: #0087ff">InputLayer</span>)      │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">224</span>, <span style="color: #00af00; text-decoration-color: #00af00">224</span>, <span style="color: #00af00; text-decoration-color: #00af00">3</span>)    │             <span style="color: #00af00; text-decoration-color: #00af00">0</span> │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ mobilenetv2_1.00_224            │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">7</span>, <span style="color: #00af00; text-decoration-color: #00af00">7</span>, <span style="color: #00af00; text-decoration-color: #00af00">1280</span>)     │     <span style="color: #00af00; text-decoration-color: #00af00">2,257,984</span> │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">Functional</span>)                    │                        │               │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ global_average_pooling2d        │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">1280</span>)           │             <span style="color: #00af00; text-decoration-color: #00af00">0</span> │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">GlobalAveragePooling2D</span>)        │                        │               │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense (<span style="color: #0087ff; text-decoration-color: #0087ff">Dense</span>)                   │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">128</span>)            │       <span style="color: #00af00; text-decoration-color: #00af00">163,968</span> │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dropout (<span style="color: #0087ff; text-decoration-color: #0087ff">Dropout</span>)               │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">128</span>)            │             <span style="color: #00af00; text-decoration-color: #00af00">0</span> │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense_1 (<span style="color: #0087ff; text-decoration-color: #0087ff">Dense</span>)                 │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">6</span>)              │           <span style="color: #00af00; text-decoration-color: #00af00">774</span> │
└─────────────────────────────────┴────────────────────────┴───────────────┘
</pre>
```
:::

::: {.output .display_data}
```{=html}
<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold"> Total params: </span><span style="color: #00af00; text-decoration-color: #00af00">2,422,726</span> (9.24 MB)
</pre>
```
:::

::: {.output .display_data}
```{=html}
<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold"> Trainable params: </span><span style="color: #00af00; text-decoration-color: #00af00">164,742</span> (643.52 KB)
</pre>
```
:::

::: {.output .display_data}
```{=html}
<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold"> Non-trainable params: </span><span style="color: #00af00; text-decoration-color: #00af00">2,257,984</span> (8.61 MB)
</pre>
```
:::
:::

::: {.cell .markdown id="cwtWjHKqevjg"}
### 3a --- Phase 1: Train top layers (base frozen) {#3a--phase-1-train-top-layers-base-frozen}
:::

::: {.cell .code colab="{\"base_uri\":\"https://localhost:8080/\"}" id="JDGD-Oh6evjh" outputId="1847d7e8-cefa-40bc-8cb0-e94f961eb370"}
``` python
model.compile(
    optimizer=keras.optimizers.Adam(1e-3),
    loss='categorical_crossentropy',
    metrics=['accuracy'])

callbacks_head = [
    keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
    keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=3, verbose=1)]

history_head = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS_HEAD,
    callbacks=callbacks_head)

print('Phase 1 training complete.')
```

::: {.output .stream .stdout}
    Epoch 1/20
:::
:::

::: {.cell .markdown id="nRdIiPrgevjh"}
### 3b --- Phase 2: Fine-tune (unfreeze last 30 layers) {#3b--phase-2-fine-tune-unfreeze-last-30-layers}
:::

::: {.cell .code id="JVmvmIqtevjh"}
``` python
base_model.trainable = True

# Freeze all layers except the last 30
for layer in base_model.layers[:-30]:
    layer.trainable = False

model.compile(
    optimizer=keras.optimizers.Adam(1e-5),   # lower LR for fine-tuning
    loss='categorical_crossentropy',
    metrics=['accuracy'])

callbacks_fine = [
    keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
    keras.callbacks.ModelCheckpoint(MODEL_PATH, save_best_only=True, verbose=1)]

history_fine = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS_FINE,
    callbacks=callbacks_fine)

print('Fine-tuning complete. Best model saved.')
```
:::

::: {.cell .code id="h1voFN36evji"}
``` python
# ── Training curves ──────────────────────────────────────────────────────────
def plot_history(h1, h2):
    acc  = h1.history['accuracy']      + h2.history['accuracy']
    val  = h1.history['val_accuracy']  + h2.history['val_accuracy']
    loss = h1.history['loss']          + h2.history['loss']
    vloss= h1.history['val_loss']      + h2.history['val_loss']
    ep   = range(1, len(acc) + 1)
    fine_start = len(h1.history['accuracy']) + 1

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, train_vals, val_vals, title in zip(
        axes,
        [acc, loss],
        [val, vloss],
        ['Accuracy', 'Loss']):
        ax.plot(ep, train_vals, label='Train')
        ax.plot(ep, val_vals,   label='Val')
        ax.axvline(fine_start, color='gray', linestyle='--', label='Fine-tune start')
        ax.set_title(title, fontweight='bold')
        ax.set_xlabel('Epoch')
        ax.legend()
        ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()

plot_history(history_head, history_fine)
```
:::

::: {.cell .markdown id="XiZqCbnDevjk"}
### 3c --- Evaluation on Test Set {#3c--evaluation-on-test-set}
:::

::: {.cell .code id="XOT8gTnuevjk"}
``` python
# ── Predictions ──────────────────────────────────────────────────────────────
y_pred_probs = model.predict(test_ds)
y_pred       = np.argmax(y_pred_probs, axis=1)
y_true       = np.concatenate([np.argmax(y, axis=1) for _, y in test_ds])

print('Classification Report:')
print(classification_report(y_true, y_pred, target_names=CLASS_NAMES))

# Confusion matrix
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(9, 7))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=CLASS_NAMES,
            yticklabels=CLASS_NAMES, cmap='Blues')
plt.title('Confusion Matrix — Test Set', fontweight='bold')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.xticks(rotation=30, ha='right')
plt.tight_layout()
plt.show()
```
:::

::: {.cell .markdown id="3D2vvAaFevjl"}

------------------------------------------------------------------------

## Stage 4 --- Crack Detection & Measurement {#stage-4--crack-detection--measurement}
:::

::: {.cell .code id="Ct8U9e-Bevjl"}
``` python
def detect_crack(image_path, pixels_per_cm=None, debug=False):

    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        return None

    gray    = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    thresh = cv2.adaptiveThreshold(
        blurred, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 11, 2)

    edges  = cv2.Canny(blurred, 50, 150)
    combined = cv2.bitwise_or(thresh, edges)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    closed = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return {'length_px': 0, 'width_mean_px': 0, 'width_max_px': 0, 'area_px2': 0}

    main_contour = max(contours, key=cv2.contourArea)

    mask = np.zeros_like(gray)
    cv2.drawContours(mask, [main_contour], -1, 255, thickness=cv2.FILLED)

    area_px2 = float(np.sum(mask > 0))

    binary = (mask > 0).astype(np.uint8)
    skel   = skeletonize(binary)
    length_px = float(np.sum(skel))

    width_mean_px = float(area_px2 / length_px) if length_px > 0 else 0.0

    if len(main_contour) >= 5:
        _, (ma, MA), _ = cv2.fitEllipse(main_contour)
        width_max_px = float(min(ma, MA))
    else:
        _, _, w, h = cv2.boundingRect(main_contour)
        width_max_px = float(min(w, h))

    result = {
        'length_px'     : length_px,
        'width_mean_px' : width_mean_px,
        'width_max_px'  : width_max_px,
        'area_px2'      : area_px2,
    }

    if pixels_per_cm and pixels_per_cm > 0:
        ppc = pixels_per_cm
        result['length_cm']    = length_px    / ppc
        result['width_mean_cm']= width_mean_px/ ppc
        result['area_cm2']     = area_px2     / (ppc ** 2)
    else:
        result['length_cm']    = float('nan')
        result['width_mean_cm']= float('nan')
        result['area_cm2']     = float('nan')

    if debug:
        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
        titles = ['Original', 'Edge/Thresh', 'Crack Mask', 'Skeleton']
        imgs   = [
            cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB),
            closed, mask,
            (skel * 255).astype(np.uint8)
        ]
        cmaps  = [None, 'gray', 'gray', 'hot']
        for ax, im, title, cmap in zip(axes, imgs, titles, cmaps):
            ax.imshow(im, cmap=cmap)
            ax.set_title(title)
            ax.axis('off')
        plt.suptitle(os.path.basename(image_path), fontsize=10)
        plt.tight_layout()
        plt.show()

    return result


print('Crack detection function defined.')
```
:::

::: {.cell .markdown id="CCNG6y_sevjm"}

------------------------------------------------------------------------

## Stage 5 --- Depth Estimation & Severity Classification {#stage-5--depth-estimation--severity-classification}
:::

::: {.cell .code id="6or3j6fCevjm"}
``` python
def estimate_depth(image_path, crack_mask_area, crack_length):
    """
    depth estimation based on:
      - Contrast ratio (darkness of crack vs. surroundings)
      - Width-to-length ratio
    Returns: 'Hairline' | 'Moderate' | 'Deep'
    """
    img   = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return 'Unknown'

    mean_full  = float(np.mean(img))

    threshold  = np.percentile(img, 5)
    crack_region = img[img <= threshold]

    mean_crack = float(np.mean(crack_region)) if len(crack_region) > 0 else mean_full

    contrast_ratio = (mean_full - mean_crack) / (mean_full + 1e-6)
    wl_ratio       = (crack_mask_area / crack_length) if crack_length > 0 else 0

    if contrast_ratio > 0.4 or wl_ratio > 5:
        return 'Deep'
    elif contrast_ratio > 0.2 or wl_ratio > 2:
        return 'Moderate'
    else:
        return 'Hairline'


def classify_severity(width_cm, length_cm, depth_label,
                       width_px=None, length_px=None):
    """
    Returns 'LOW' | 'MEDIUM' | 'HIGH'.
    Falls back to pixel thresholds if cm values are NaN.
    """
    import math
    use_cm = not (math.isnan(width_cm) or math.isnan(length_cm))

    if use_cm:
        if depth_label == 'Deep' or width_cm > 0.5 or length_cm > 20:
            return 'HIGH'
        elif depth_label == 'Moderate' or 0.2 <= width_cm <= 0.5 or 5 <= length_cm <= 20:
            return 'MEDIUM'
        else:
            return 'LOW'
    else:
        # Pixel-based fallback (assumes ~50px per cm reference)
        w, l = width_px or 0, length_px or 0
        if depth_label == 'Deep' or w > 25 or l > 1000:
            return 'HIGH'
        elif depth_label == 'Moderate' or w > 10 or l > 250:
            return 'MEDIUM'
        else:
            return 'LOW'


print('Severity functions defined.')
```
:::

::: {.cell .markdown id="_zz1ZWgQevjn"}

------------------------------------------------------------------------

## Stage 6 --- Material Estimation {#stage-6--material-estimation}
:::

::: {.cell .code id="HbP_MvShevjn"}
``` python
DEPTH_ASSUMPTION_CM = {
    'Hairline': 0.1,
    'Moderate': 0.3,
    'Deep'    : 0.6,
    'Unknown' : 0.2}

FILLER_RECOMMENDATION = {
    ('deck',      'LOW')   : 'Polyurethane sealant',
    ('deck',      'MEDIUM'): 'Epoxy injection',
    ('deck',      'HIGH')  : 'Structural epoxy + professional inspection advised',
    ('pavement',  'LOW')   : 'Asphalt crack filler',
    ('pavement',  'MEDIUM'): 'Cold pour crack filler',
    ('pavement',  'HIGH')  : 'Hot rubberized asphalt + professional inspection advised',
    ('wall',      'LOW')   : 'Hairline crack filler / interior caulk',
    ('wall',      'MEDIUM'): 'Cement-based patching compound',
    ('wall',      'HIGH')  : 'Structural repair mortar + professional inspection advised',}


def estimate_material(length_cm, width_cm, depth_label, structure_type, severity,
                       length_px=None, width_px=None, pixels_per_cm=None):
    """
    Returns (volume_cm3, material_cm3, filler_type).
    Falls back to pixel-based estimate with assumed scale if cm values are NaN.
    """
    import math
    assumed_depth = DEPTH_ASSUMPTION_CM.get(depth_label, 0.2)
    use_cm = not (math.isnan(length_cm) or math.isnan(width_cm))

    if use_cm:
        l, w = length_cm, width_cm
    elif pixels_per_cm and pixels_per_cm > 0:
        l = (length_px or 0) / pixels_per_cm
        w = (width_px  or 0) / pixels_per_cm
    else:
        # Rough proxy: assume 50px ≈ 1cm
        l = (length_px or 0) / 50
        w = (width_px  or 0) / 50

    volume_cm3   = l * w * assumed_depth
    material_cm3 = volume_cm3 * 1.2   # 20% overfill

    struct = structure_type.lower().replace('_cracked', '').replace('_uncracked', '')
    filler = FILLER_RECOMMENDATION.get((struct, severity), 'General crack filler')

    return round(volume_cm3, 4), round(material_cm3, 4), filler


print('Material estimation function defined.')
```
:::

::: {.cell .markdown id="g2sAOJe6evjo"}

------------------------------------------------------------------------

## Stage 7 --- Full Pipeline: Process All Images → CSV {#stage-7--full-pipeline-process-all-images--csv}
:::

::: {.cell .code id="pipwrgEhevjo"}
``` python
def preprocess_for_model(image_path):
    """Load, resize, and normalise a single image for inference."""
    img = tf.keras.utils.load_img(image_path, target_size=IMG_SIZE)
    arr = tf.keras.utils.img_to_array(img) / 255.0
    return np.expand_dims(arr, axis=0)


def run_pipeline(image_dir, model, class_names,
                 pixels_per_cm=None, debug_first_n=3):
    """
    Walk image_dir (all subdirs), classify each image, measure cracks,
    estimate materials, and return a DataFrame.
    """
    valid_ext = ('.jpg', '.jpeg', '.png', '.bmp')
    rows      = []
    debug_count = 0

    # Collect all image paths
    all_paths = []
    for root, _, files in os.walk(image_dir):
        for f in files:
            if f.lower().endswith(valid_ext):
                all_paths.append(os.path.join(root, f))

    print(f'Processing {len(all_paths)} images...')

    for idx, img_path in enumerate(all_paths):
        if idx % 100 == 0:
            print(f'  [{idx}/{len(all_paths)}]')

        x       = preprocess_for_model(img_path)
        probs   = model.predict(x, verbose=0)[0]
        pred_idx= int(np.argmax(probs))
        pred_cls= class_names[pred_idx]
        conf    = float(probs[pred_idx])

        is_cracked     = 'cracked' in pred_cls and 'uncracked' not in pred_cls
        structure_type = pred_cls.split('_')[0]   # deck / pavement / wall

        row = {
            'image_filename'   : os.path.basename(img_path),
            'structure_type'   : structure_type,
            'cracked'          : is_cracked,
            'confidence_score' : round(conf, 4),
        }

        if is_cracked:
            show_debug = debug_count < debug_first_n

            m = detect_crack(img_path, pixels_per_cm=pixels_per_cm, debug=show_debug)
            if m is None:
                m = {k: 0 for k in [
                    'length_px','width_mean_px','width_max_px','area_px2',
                    'length_cm','width_mean_cm','area_cm2' ]}

            depth    = estimate_depth(img_path, m['area_px2'], m['length_px'])
            severity = classify_severity(
                m['width_mean_cm'], m['length_cm'], depth,
                m['width_mean_px'], m['length_px'])

            # ── Material estimation ───────────────────────────────────────────
            vol, mat, filler = estimate_material(
                m['length_cm'], m['width_mean_cm'], depth,
                structure_type, severity,
                m['length_px'], m['width_mean_px'], pixels_per_cm)

            row.update({
                'crack_length_px'    : round(m['length_px'],     2),
                'crack_width_mean_px': round(m['width_mean_px'], 2),
                'crack_area_px2'     : round(m['area_px2'],      2),
                'crack_length_cm'    : round(m['length_cm'],     4) if not pd.isna(m['length_cm'])     else float('nan'),
                'crack_width_cm'     : round(m['width_mean_cm'], 4) if not pd.isna(m['width_mean_cm']) else float('nan'),
                'crack_area_cm2'     : round(m['area_cm2'],      4) if not pd.isna(m['area_cm2'])      else float('nan'),
                'depth_estimate'     : depth,
                'severity'           : severity,
                'estimated_volume_cm3': vol,
                'material_needed_cm3' : mat,
                'recommended_filler' : filler,
                'flagged_for_review' : severity == 'HIGH',})

            if show_debug:
                debug_count += 1
        else:
            row.update({
                'crack_length_px': 0, 'crack_width_mean_px': 0,
                'crack_area_px2': 0,  'crack_length_cm': float('nan'),
                'crack_width_cm': float('nan'), 'crack_area_cm2': float('nan'),
                'depth_estimate': 'N/A', 'severity': 'N/A',
                'estimated_volume_cm3': 0, 'material_needed_cm3': 0,
                'recommended_filler': 'None', 'flagged_for_review': False,})

        rows.append(row)

    df = pd.DataFrame(rows)
    print(f'\nPipeline complete. {len(df)} images processed.')
    return df

TEST_IMG_DIR = os.path.join(SPLIT_DIR, 'test')

results_df = run_pipeline(
    image_dir    = TEST_IMG_DIR,
    model        = model,
    class_names  = CLASS_NAMES,
    pixels_per_cm= PIXELS_PER_CM,
    debug_first_n= 3
)

results_df.head(10)
```
:::

::: {.cell .markdown id="g8iInWoGevjp"}

------------------------------------------------------------------------

## Stage 8 --- Save CSV {#stage-8--save-csv}
:::

::: {.cell .code id="xUm07Z-9evjp"}
``` python
results_df.to_csv(CSV_PATH, index=False)
print(f'Results saved to: {CSV_PATH}')
print(f'   Rows: {len(results_df)}  |  Columns: {len(results_df.columns)}')
print(f'\nColumn list:\n{list(results_df.columns)}')
```
:::

::: {.cell .markdown id="sf4tzi_Levjq"}

------------------------------------------------------------------------

## Stage 9 --- Summary Analytics & Visualisations {#stage-9--summary-analytics--visualisations}
:::

::: {.cell .code id="xwo4aJTQevjq"}
``` python
cracked_df = results_df[results_df['cracked'] == True].copy()

print('=== Dataset Summary ===')
print(f'Total images processed : {len(results_df)}')
print(f'Cracked                : {len(cracked_df)}')
print(f'Uncracked              : {len(results_df) - len(cracked_df)}')
print(f'Flagged for review     : {results_df["flagged_for_review"].sum()}')
print()
print('Severity distribution:')
print(cracked_df['severity'].value_counts())
print()
print('Structure type breakdown:')
print(results_df['structure_type'].value_counts())
```
:::

::: {.cell .code id="hiW962c5evjq"}
``` python
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle('Crack Analysis — Summary Dashboard', fontsize=15, fontweight='bold')

# 1. Cracked vs Uncracked
counts = results_df['cracked'].value_counts()
axes[0,0].pie(counts, labels=['Uncracked','Cracked'], autopct='%1.1f%%',
              colors=['#4CAF50','#F44336'], startangle=90)
axes[0,0].set_title('Cracked vs Uncracked')

# 2. Structure type distribution
results_df['structure_type'].value_counts().plot(
    kind='bar', ax=axes[0,1], color=['#2196F3','#FF9800','#9C27B0'], edgecolor='black')
axes[0,1].set_title('Images by Structure Type')
axes[0,1].set_xlabel('')
axes[0,1].tick_params(axis='x', rotation=0)

# 3. Severity distribution
if len(cracked_df) > 0:
    cracked_df['severity'].value_counts().reindex(['LOW','MEDIUM','HIGH'], fill_value=0).plot(
        kind='bar', ax=axes[0,2], color=['#8BC34A','#FFC107','#F44336'], edgecolor='black')
axes[0,2].set_title('Severity Distribution (Cracked)')
axes[0,2].tick_params(axis='x', rotation=0)

# 4. Depth estimate distribution
if len(cracked_df) > 0:
    cracked_df['depth_estimate'].value_counts().plot(
        kind='bar', ax=axes[1,0], color='#03A9F4', edgecolor='black')
axes[1,0].set_title('Depth Estimate Distribution')
axes[1,0].tick_params(axis='x', rotation=0)

# 5. Material needed distribution
if len(cracked_df) > 0:
    axes[1,1].hist(cracked_df['material_needed_cm3'].dropna(), bins=20,
                   color='#FF5722', edgecolor='black', alpha=0.8)
axes[1,1].set_title('Material Needed (cm³) Distribution')
axes[1,1].set_xlabel('cm³')

# 6. Severity by structure type
if len(cracked_df) > 0:
    pivot = cracked_df.groupby(['structure_type','severity']).size().unstack(fill_value=0)
    pivot.reindex(columns=['LOW','MEDIUM','HIGH'], fill_value=0).plot(
        kind='bar', ax=axes[1,2],
        color=['#8BC34A','#FFC107','#F44336'], edgecolor='black')
axes[1,2].set_title('Severity by Structure Type')
axes[1,2].tick_params(axis='x', rotation=0)
axes[1,2].legend(title='Severity')

plt.tight_layout()
plt.show()
```
:::

::: {.cell .markdown id="HD70zQw2evjq"}

------------------------------------------------------------------------

## Stage 10 --- Single Image Inference (Demo) {#stage-10--single-image-inference-demo}
:::

::: {.cell .code id="-DrnBYO7evjr"}
``` python
def analyse_single_image(image_path, model, class_names, pixels_per_cm=None):
    """End-to-end analysis for a single image with visual output."""
    x      = preprocess_for_model(image_path)
    probs  = model.predict(x, verbose=0)[0]
    pred   = class_names[np.argmax(probs)]
    conf   = float(np.max(probs))

    is_cracked     = 'cracked' in pred and 'uncracked' not in pred
    structure_type = pred.split('_')[0]

    print(f'Image        : {os.path.basename(image_path)}')
    print(f'Prediction   : {pred}  (confidence: {conf:.2%})')
    print(f'Structure    : {structure_type}')
    print(f'Cracked      : {is_cracked}')

    if is_cracked:
        m        = detect_crack(image_path, pixels_per_cm=pixels_per_cm, debug=True)
        depth    = estimate_depth(image_path, m['area_px2'], m['length_px'])
        severity = classify_severity(m['width_mean_cm'], m['length_cm'], depth,
                                     m['width_mean_px'], m['length_px'])
        vol, mat, filler = estimate_material(
            m['length_cm'], m['width_mean_cm'], depth,
            structure_type, severity,
            m['length_px'], m['width_mean_px'], pixels_per_cm
        )

        print(f'\n── Crack Measurements ──────────────────')
        print(f'Length           : {m["length_px"]:.1f} px  /  {m["length_cm"]:.2f} cm')
        print(f'Mean Width       : {m["width_mean_px"]:.1f} px  /  {m["width_mean_cm"]:.2f} cm')
        print(f'Area             : {m["area_px2"]:.1f} px²')
        print(f'Depth Estimate   : {depth}')
        print(f'Severity         : {severity}')
        print(f'\n── Material Estimate ───────────────────')
        print(f'Fill Volume      : {vol:.4f} cm³')
        print(f'Material Needed  : {mat:.4f} cm³  (with 20% overfill)')
        print(f'Recommended Fill : {filler}')
        if severity == 'HIGH':
            print('⚠️  FLAGGED FOR PROFESSIONAL REVIEW')


# ── Usage: Replace with any image path ───────────────────────────────────────
# SAMPLE_IMAGE = '/content/drive/MyDrive/crack_dataset/wall_cracked/some_image.jpg'
# analyse_single_image(SAMPLE_IMAGE, model, CLASS_NAMES, pixels_per_cm=PIXELS_PER_CM)

print(' Single image inference function ready.')
print('   Uncomment the last two lines above and set SAMPLE_IMAGE to test it.')
```
:::

::: {.cell .markdown id="_qVxXVOoevjr"}

------------------------------------------------------------------------

## Assumptions & Limitations {#assumptions--limitations}

  -----------------------------------------------------------------------
  Item                                Detail
  ----------------------------------- -----------------------------------
  **Depth**                           Estimated via contrast &
                                      width/length ratio --- not true
                                      physical depth

  **Real-world scale**                Set `PIXELS_PER_CM` if a known
                                      reference is in your images;
                                      otherwise measurements are in
                                      pixels

  **Material estimates**              Rectangular cross-section assumed;
                                      20% overfill applied

  **Filler recommendations**          Simplified guidelines only --- not
                                      engineering specifications

  **HIGH severity cracks**            Always involve a professional
                                      structural engineer

  **Dataset balance**                 If classes are imbalanced, consider
                                      adding `class_weight` to
                                      `model.fit()`
  -----------------------------------------------------------------------
:::
