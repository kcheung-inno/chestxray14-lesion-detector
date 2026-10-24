---
license: apache-2.0
library_name: rfdetr
tags:
  - object-detection
  - medical-imaging
  - chest-x-ray
  - radiology
  - onnx
  - rf-detr
datasets:
  - NIH-ChestX-ray14
metrics:
  - mAP
---

# RF-DETR-S — 8-class lesion detector for chest radiographs

Fine-tuned **RF-DETR-S** for 8-class lesion localisation on chest radiographs, trained on the
box-annotated subset of **NIH ChestX-ray14**. Exported to ONNX at 576×576; runs in the browser
via onnxruntime-web with no server-side inference.

**Research use only — not a medical device.** No regulatory clearance, no clinical validation,
no prospective testing. Do not use for diagnosis, screening or triage.

## Results

Evaluated on **348 held-out images**, split **by patient** so no patient appears in both halves:

| Metric | Value |
|---|---|
| mAP @50:95 | **0.078** |
| mAP @50 | 0.144 |
| mAP @75 | 0.075 |
| mAR @100 | 0.263 |
| Inference | 33 ms/image |

Per-class AP @50:95:

| Finding | AP@50:95 | boxes (val) |
|---|---|---|
| Cardiomegaly | 0.395 | 28 |
| Pneumothorax | 0.067 | 17 |
| Nodule | 0.062 | 17 |
| Mass | 0.051 | 22 |
| Infiltrate | 0.026 | 32 |
| Effusion | 0.012 | 23 |
| Atelectasis | 0.008 | 34 |
| Pneumonia | 0.006 | 27 |

These numbers are low, and the reason is the data. The box-annotated subset is only 880 images
and 984 boxes; nodule and mass have just 17 and 22 validation boxes between them, so those
per-class figures are the least reliable here.

The spread across classes tracks how many boxes each class has. Cardiomegaly — large,
consistently shaped, 28 validation boxes — reaches 0.395, while findings with fewer and more
variable annotations sit an order of magnitude lower. Confidence and accuracy also come apart:
mass reaches 0.504 median confidence at 0.051 AP, so a high score does not imply a correct box.

## Using it

```python
import numpy as np, onnxruntime as ort
from PIL import Image

RES, MEAN, STD = 576, np.array([0.485,0.456,0.406],np.float32), np.array([0.229,0.224,0.225],np.float32)
sess = ort.InferenceSession("rfdetr_nih.onnx", providers=["CPUExecutionProvider"])
img  = Image.open("chest.png").convert("RGB").resize((RES, RES), Image.BILINEAR)
x = ((np.asarray(img, np.float32)/255 - MEAN)/STD).transpose(2,0,1)[None]
dets, labels = sess.run(None, {"input": x.astype(np.float32)})
# dets: (1,300,4) normalised cxcywh;  labels: (1,300,9)
probs = 1/(1+np.exp(-labels[0]))          # sigmoid, NOT softmax
```

Two conventions worth knowing, both established by comparing against PyTorch:

1. **Scores are `sigmoid(logit)`**, not softmax.
2. **The graph emits 9 class logits for 8 classes.** ONNX argmax index `i` maps to COCO category
   `i+1` (index 1 is cardiomegaly, category 2). Only indices 0–7 are meaningful.

Class order (COCO ids 1–8): `Atelectasis, Cardiomegaly, Effusion, Infiltrate, Mass, Nodule,
Pneumonia, Pneumothorax`.

## Training

| | |
|---|---|
| Base | RF-DETR-S, COCO-pretrained |
| Epochs | 30 |
| Batch / grad-accum | 2 / 8 |
| Resolution | 576 |
| LR | 1e-4 |
| Wall clock | 74.7 min (149.4 s/epoch) |
| Peak VRAM | 3.1 GB |
| Split | 1,432 train / 348 val, by patient |

30 epochs is short for DETR-family models, so this is partially converged. Single seed, no
confidence intervals — differences under roughly 0.02 mAP are noise.

## Limitations

- 984 boxes total, averaging ~120 per class. Nothing here is well supervised.
- Nodule AP rests on 17 validation boxes; a different split seed would move it substantially.
- Partially converged, single seed, no repeats.
- The two models were scored with different evaluators (pycocotools for RF-DETR, D-FINE's own
  `CocoEvaluator`), so the gap between them is indicative only.
- Trained on 8 findings; it is not a nodule-specific detector.
- Not a medical device.

## Licence and attribution

Apache-2.0. The training data carries its own terms: NIH ChestX-ray14 asks that redistribution
carry a link to the [download site](https://nihcc.app.box.com/v/ChestXray-NIHCC), a citation of
the CVPR 2017 paper, and acknowledgement of the NIH Clinical Center as data provider.

> Wang X, Peng Y, Lu L, Lu Z, Bagheri M, Summers RM. *ChestX-ray8: Hospital-scale Chest X-ray
> Database and Benchmarks on Weakly-Supervised Classification and Localization of Common Thorax
> Diseases.* IEEE CVPR, pp. 3462–3471, 2017. DOI: [10.1109/CVPR.2017.369](https://doi.org/10.1109/CVPR.2017.369)

Provided "AS IS", without warranties or conditions of any kind (Apache-2.0 §7–§8).

## Links

- Code, documentation and the browser demo: <https://github.com/kcheung-inno/chestxray14-lesion-detector>
- Full model card: `docs/MODEL_CARD.md` in that repository
