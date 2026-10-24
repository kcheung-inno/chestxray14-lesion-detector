# Model card — NIH ChestX-ray14 lesion detector (RF-DETR-S / D-FINE-S)

## Summary

Two object detectors fine-tuned for **8-class lesion localisation** on chest radiographs,
trained on the box-annotated subset of NIH ChestX-ray14.

This is a proof of concept, not a clinical or benchmark-grade detector — read
[Limitations](#limitations) before quoting any figure.

## Model details

| | RF-DETR-S | D-FINE-S |
|---|---|---|
| Architecture | DINOv2 ViT + DETR (NMS-free) | HGNetv2-B0 + DETR (FDR / GO-LSD) |
| Parameters | ~32 M (published) | 10.18 M (measured) |
| Input resolution | 576 × 576 | 640 × 640 |
| Pretrained init | COCO (RF-DETR) | COCO (`dfine_s_coco.pth`) |
| Licence (code) | Apache-2.0 | Apache-2.0 |
| Output | `dets` (1,300,4 normalized cxcywh) + `labels` (1,300,9) | COCO detections |

### Class index convention (verified, not assumed)

The exported ONNX emits **9** class logits for 8 classes. Checked against PyTorch on a
confident detection: **ONNX argmax index `i` ↔ COCO category `i+1`** (index 1 → Cardiomegaly,
which is category 2). The 9th logit is unused; consumers must restrict to indices 0–7.
Score is **`sigmoid(logit)`**, not softmax.

## Training data

NIH ChestX-ray14 box-annotated subset (`BBox_List_2017.csv`): **880 images, 984
radiologist-drawn boxes, 8 findings**, plus 900 "No Finding" negatives.

Split **by patient** (no leakage): train 1,432 images (701 with boxes), val 348 (179 with boxes).

Per-finding GT boxes in the **held-out** split:

| Finding | GT boxes (val) |
|---|---|
| Atelectasis | 34 |
| Infiltrate | 32 |
| Cardiomegaly | 28 |
| Pneumonia | 27 |
| Effusion | 23 |
| **Mass** | **22** |
| **Nodule** | **17** |
| Pneumothorax | 17 |

## Results — measured, not quoted

Evaluated on the 348 held-out images. RF-DETR scored with an independent pycocotools harness;
D-FINE with its own `CocoEvaluator`. Both compute COCO mAP over the same images, but they are
**different harnesses** — treat the gap as indicative.

### Overall

| Metric | RF-DETR-S | D-FINE-S |
|---|---|---|
| mAP @50:95 | **0.0783** | **0.088** |
| mAP @50 | 0.1439 | 0.163 |
| mAP @75 | 0.0751 | 0.073 |
| mAR @100 | 0.2631 | — |
| Inference latency | **33.3 ms/img** | — |
| Training wall-clock (30 ep) | 74.7 min | **~52 min** |
| Seconds / epoch | 149.4 | ~104 |
| Peak VRAM | 3.1 GB | ~3.0 GB |

### Per-class (RF-DETR-S, AP @50:95)

| Finding | GT boxes | AP@50:95 | AP@50 |
|---|---|---|---|
| Cardiomegaly | 28 | **0.3952** | 0.6445 |
| Pneumothorax | 17 | 0.0671 | 0.1253 |
| **Nodule** | **17** | **0.0615** | 0.1110 |
| **Mass** | **22** | **0.0508** | 0.1226 |
| Infiltrate | 32 | 0.0264 | 0.0621 |
| Effusion | 23 | 0.0117 | 0.0476 |
| Atelectasis | 34 | 0.0078 | 0.0265 |
| Pneumonia | 27 | 0.0056 | 0.0120 |

D-FINE's per-class breakdown was not computed; its overall figure only is reported.

### The model is uneven, and that is the finding

Confidence on held-out positives (max sigmoid across the 300 queries, median over 6 images):

| Finding | median max-sigmoid | AP@50:95 |
|---|---|---|
| Cardiomegaly | 0.746 | 0.395 |
| Mass | 0.504 | 0.051 |
| Atelectasis | 0.281 | 0.008 |
| Pneumothorax | 0.218 | 0.067 |
| Effusion | 0.165 | 0.012 |
| **Nodule** | **0.106** | **0.062** |

Cardiomegaly — a large, consistent, high-prevalence silhouette — localises well. **Nodule and
Mass, the findings most relevant to lung cancer screening, are the weak points**, and high
confidence in those classes does not imply correct localisation (Mass: 0.504 confidence at
0.051 AP).

## Box supervision is the binding constraint

The per-class spread above is a property of the *data*, not the architectures. Cardiomegaly —
28 validation boxes, a large and consistently-shaped structure — reaches 0.3952, while classes
with fewer and more variable annotations sit up to two orders of magnitude lower. Detector
quality here tracks how many boxes a class has.

This is the central finding, and it is a statement about the dataset rather than about training:
**a larger annotated corpus would improve these numbers far more than a better architecture
would.** Downstream users should expect the per-class ranking above to persist.

Note that the largest public chest radiograph corpora are released for research use only, so the
extra supervision that would close this gap is not readily available.

## Limitations

1. **Tiny supervision.** 984 boxes total; 8 classes averaging ~120 boxes each. Nodule/Mass:
   164 boxes combined, 125 train / 39 val.
2. **High variance on the target finding.** Nodule AP rests on **17 validation boxes**. Do not
   read precision into 4 decimal places — the point estimate could move substantially with a
   different split seed.
3. **Partial convergence.** 30 epochs is short for DETR-family models (D-FINE's own custom
   config specifies 220). Both were still improving.
4. **Single run, single seed.** No confidence intervals, no repeats. Differences under
   ~0.02 mAP should be treated as noise — which means the RF-DETR/D-FINE gap is **not**
   decisive.
5. **Different evaluation harnesses** for the two models (see Results).
6. **Image-level vs box-level.** The other ~111,000 ChestX-ray14 images have image-level labels
   only and were not used; the negatives here are a 900-image sample of "No Finding".
7. **Not a medical device.** No regulatory clearance, no clinical validation, no prospective
   testing. Must not be used for diagnosis or screening.

## Licence and attribution

Released under the Apache License 2.0. Two sets of bundled material carry their own terms:
the demo radiographs from NIH ChestX-ray14, and the onnxruntime-web binaries in `demo/ort/`
(MIT, © Microsoft Corporation). See [LICENSING.md](../LICENSING.md) for the breakdown and
[LICENSE](../LICENSE) for the full text.

NIH ChestX-ray14 asks that redistribution carry three things:

1. Link: <https://nihcc.app.box.com/v/ChestXray-NIHCC>
2. Citation: Wang X, Peng Y, Lu L, Lu Z, Bagheri M, Summers RM. *ChestX-ray8: Hospital-scale
   Chest X-ray Database and Benchmarks on Weakly-Supervised Classification and Localization of
   Common Thorax Diseases.* IEEE CVPR, pp. 3462–3471, 2017.
   DOI: [10.1109/CVPR.2017.369](https://doi.org/10.1109/CVPR.2017.369)
3. Acknowledgement: the **NIH Clinical Center** is the data provider.

## Intended use

- Research on detection architectures for chest radiographs.
- A worked example of training a detector on a small, openly available annotated set.

**Out of scope:** any clinical decision, screening programme, triage, or diagnostic use.

## Reproduce

```bash
.venv/Scripts/python.exe scripts/prep_nih.py            # select images, split by patient
.venv/Scripts/python.exe scripts/nih_download.py        # fetch images (bring your own HF access)
.venv/Scripts/python.exe scripts/build_nih_coco.py      # COCO + RF-DETR + D-FINE layouts
.venv/Scripts/python.exe scripts/train_rfdetr.py --variant small --epochs 30 \
    --batch 2 --grad-accum 8 --resolution 576 \
    --dataset-dir data/rfdetr_nih --output outputs/rfdetr_nih
.venv/Scripts/python.exe scripts/eval_rfdetr.py --weights outputs/rfdetr_nih/checkpoint_best_ema.pth \
    --dataset-dir data/rfdetr_nih --split test --variant small --tag rfdetr_nih
.venv/Scripts/python.exe scripts/eval_per_class.py --gt data/rfdetr_nih/test/_annotations.coco.json \
    --det outputs/eval/detections_rfdetr_nih_test.json --tag rfdetr_nih
```

## Citation

If you use this model, cite the dataset:

```bibtex
@inproceedings{Wang_2017,
  doi = {10.1109/cvpr.2017.369},
  year = {2017}, month = {jul}, publisher = {{IEEE}},
  author = {Xiaosong Wang and Yifan Peng and Le Lu and Zhiyong Lu and Mohammadhadi Bagheri and Ronald M. Summers},
  title = {{ChestX}-Ray8: Hospital-Scale Chest X-Ray Database and Benchmarks on Weakly-Supervised Classification and Localization of Common Thorax Diseases},
  booktitle = {2017 {IEEE} Conference on Computer Vision and Pattern Recognition ({CVPR})}
}
```
