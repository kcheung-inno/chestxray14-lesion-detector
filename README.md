# Chest X-ray lesion detection

A small object-detection model for chest radiographs. It localises eight findings —
atelectasis, cardiomegaly, effusion, infiltrate, mass, nodule, pneumonia, pneumothorax — and
ships with a browser demo that runs the model locally, with no server-side inference.

Trained on [NIH ChestX-ray14](https://nihcc.app.box.com/v/ChestXray-NIHCC), using the 880
radiographs in that dataset that carry radiologist-drawn bounding boxes.

## Results

Evaluated on 348 held-out radiographs, split by patient so no patient appears in both halves:

| | RF-DETR-S | D-FINE-S |
|---|---|---|
| mAP @50:95 | 0.078 | **0.088** |
| mAP @50 | 0.144 | 0.163 |
| Inference | **33 ms/img** | — |
| Training (30 epochs) | 75 min | **52 min** |
| Peak VRAM | 3.1 GB | 3.0 GB |

Per-class AP @50:95 for RF-DETR-S:

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

These numbers are low, and the reason is the data rather than the models. The box-annotated
subset of ChestX-ray14 is only 880 images and 984 boxes; nodule and mass — the findings that
matter most for lung cancer — have just 17 and 22 validation boxes between them.

The spread across classes tracks how many boxes each class has. Cardiomegaly, a large and
consistently-shaped structure with 28 validation boxes, reaches 0.395. Findings with fewer and
more variable annotations sit an order of magnitude lower. More annotated data would improve
this far more than a different architecture would.

Confidence and accuracy also come apart: mass reaches 0.504 median confidence at 0.051 AP, so a
high score does not imply a correct box.

## Demo

Runs the exported ONNX in the browser via onnxruntime-web — WebGPU where available, WASM
otherwise. About 0.75 s per image locally.

```bash
python scripts/serve_demo.py     # http://127.0.0.1:8777/
```

Pick from 11 held-out images, toggle the gold-standard overlay, and drag the score threshold.
The set is a spread rather than a showcase: three cardiomegaly cases where the model works,
two mass and three nodule cases where it does not, and two negatives.

A server is needed rather than opening the file directly: browsers block fetching a 122 MB
model from a `file://` origin, and WASM threading needs COOP/COEP headers.

### Conventions

Two things about the exported graph that are easy to get wrong, both established by comparing
against PyTorch:

- **Scores are `sigmoid(logit)`**, not softmax.
- **The graph emits 9 class logits for 8 classes.** ONNX argmax index `i` maps to COCO category
  `i+1` (index 1 is cardiomegaly, category 2). Only indices 0–7 are meaningful.

Class order (COCO ids 1–8): `Atelectasis, Cardiomegaly, Effusion, Infiltrate, Mass, Nodule,
Pneumonia, Pneumothorax`.

## Getting the data

Images are not included — they are ~1.3 GB and freely available from the provider:

```bash
python scripts/nih_list_files.py    # index the mirror (~1 min)
python scripts/prep_nih.py          # pick boxed + negative images, split by patient
python scripts/nih_download.py      # fetch the 1,780 selected images
python scripts/build_nih_coco.py    # build COCO / RF-DETR / D-FINE layouts
```

The scripts fetch from a HuggingFace mirror (`iraqigold/nih-chest-xray-14-flat`) because it
allows per-file access, so you download 1,780 images instead of the full 42 GB. The
[official NIH download](https://nihcc.app.box.com/v/ChestXray-NIHCC) works just as well — put the
images in `data/nih/images/` and everything downstream is unchanged.

## Training

```bash
python scripts/train_rfdetr.py --variant small --epochs 30 --batch 2 --grad-accum 8 \
    --resolution 576 --dataset-dir data/rfdetr_nih --output outputs/rfdetr_nih
python scripts/eval_rfdetr.py --weights outputs/rfdetr_nih/checkpoint_best_ema.pth \
    --dataset-dir data/rfdetr_nih --split test --variant small --tag rfdetr_nih
python scripts/eval_per_class.py --gt data/rfdetr_nih/test/_annotations.coco.json \
    --det outputs/eval/detections_rfdetr_nih_test.json --tag rfdetr_nih
python scripts/export_onnx.py --weights outputs/rfdetr_nih/checkpoint_best_ema.pth \
    --outdir demo/model_nih --resolution 576
```

`scripts/chain_nih3.sh` runs the whole train-and-evaluate sequence, one GPU job at a time.

Requirements: Python 3.11, an NVIDIA GPU (developed on an RTX 5070 Laptop 8 GB, sm_120 → torch
cu128), `rfdetr[train,loggers]`, `pycocotools`, `onnxruntime`, `onnx`, Pillow.

If you have Ollama installed, stop it before training — its desktop app respawns `llama-server.exe`
and takes roughly 4 GB of VRAM, which roughly doubles the per-epoch time.

## Scripts

```
prep_nih.py            select box-annotated + negative images, split by patient
nih_list_files.py      index the mirror's 100k files -> filename to path map
nih_download.py        fetch just the selected images
build_nih_coco.py      COCO + RF-DETR (Roboflow) + D-FINE layouts
prep_dfine_nih.py      write the D-FINE config + dataset descriptor
fetch_rfdetr_weights.py  fetch pretrained COCO weights
fetch_weights_chunked.py fetch large weights in ranged chunks (resumable)
train_rfdetr.py        train RF-DETR-S (class count read from the dataset)
eval_rfdetr.py         independent pycocotools evaluation
eval_per_class.py      per-class AP breakdown
export_onnx.py         export to ONNX at the trained resolution
verify_onnx_nih.py     check ONNX reproduces PyTorch before shipping it
prep_demo_images_nih.py  stage held-out images + gold-standard boxes for the demo
serve_demo.py          local HTTP server for the browser demo
verify_deploy.js       drive real Chrome against a deployed URL; reports console errors
upload_to_hf.py        publish the ONNX + model card to the HuggingFace model repo
republish_hf.py        republish that repo as a single clean commit (destructive)
chain_nih3.sh          end-to-end train + evaluate, one GPU job at a time
check_demo_html.js     syntax + element-id check for the demo page
check_publishable.sh   pre-commit checks: no weights, caches or obvious problems
```

## Documentation

| File | Contents |
|---|---|
| `docs/MODEL_CARD.md` | architectures, per-class results, conventions, limitations |
| `docs/DATA_CARD.md` | dataset description and licence |
| `docs/HF_MODEL_CARD.md` | the model card used on the HuggingFace repo |

## Licence and attribution

The code, weights and documentation are under the [Apache License 2.0](LICENSE). Two sets of
material carry their own terms:

- **`demo/nih_images/`** — radiographs from NIH ChestX-ray14, used under that dataset's terms
  (see [DATA_LICENSE.md](DATA_LICENSE.md)).
- **`demo/ort/`** — onnxruntime-web, © Microsoft Corporation, MIT licensed
  (see [demo/ort/LICENSE](demo/ort/LICENSE)).

If you redistribute this work, carry the NIH attribution listed in
[DATA_LICENSE.md](DATA_LICENSE.md). See [NOTICE](NOTICE) and
[THIRD-PARTY-NOTICES.md](THIRD-PARTY-NOTICES.md) for the full breakdown.

Provided "as is", without warranty of any kind — see
[LICENSING.md](LICENSING.md) and Apache-2.0 §7–§8.

## Not a medical device

Research and educational use only. No regulatory clearance, no clinical validation, no
prospective testing. Do not use for diagnosis, screening or triage.

## Citation

```bibtex
@inproceedings{Wang_2017,
  doi = {10.1109/cvpr.2017.369},
  year = {2017}, month = {jul}, publisher = {{IEEE}},
  author = {Xiaosong Wang and Yifan Peng and Le Lu and Zhiyong Lu and Mohammadhadi Bagheri and Ronald M. Summers},
  title = {{ChestX}-Ray8: Hospital-Scale Chest X-Ray Database and Benchmarks on Weakly-Supervised Classification and Localization of Common Thorax Diseases},
  booktitle = {2017 {IEEE} Conference on Computer Vision and Pattern Recognition ({CVPR})}
}
```
