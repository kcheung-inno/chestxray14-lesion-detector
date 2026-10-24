# Third-party notices

## Redistributed in this repository (notice required)

### onnxruntime-web v1.30.0
- **Licence:** MIT
- **Copyright:** Copyright (c) Microsoft Corporation
- **Where:** `demo/ort/` (`ort.min.js`, `ort-wasm-simd-threaded.mjs`,
  `ort-wasm-simd-threaded.wasm`, `ort-wasm-simd-threaded.jsep.mjs`,
  `ort-wasm-simd-threaded.jsep.wasm`)
- **Source:** <https://github.com/microsoft/onnxruntime>
- **Full text:** [`demo/ort/LICENSE`](demo/ort/LICENSE)

Redistributed unmodified so that `demo/index.html` runs offline. The MIT licence requires the
copyright and permission notice to accompany all copies or substantial portions of the
software; `demo/ort/LICENSE` satisfies that.

## Required at build/run time (not redistributed here)

### RF-DETR
- **Licence:** Apache License 2.0
- **Source:** <https://github.com/roboflow/rf-detr>
- **Use:** the detection architecture this model was fine-tuned from.
- **Note:** only the N/S/M/L variants are Apache-2.0. RF-DETR-XL, RF-DETR-2XL and
  `rfdetr_plus` are under PML-1.0 (source-available, restricted). This project uses
  **small** only.

### D-FINE
- **Licence:** Apache License 2.0
- **Source:** <https://github.com/Peterande/D-FINE>
- **Use:** second detector in the head-to-head comparison. Cloned to `third_party/` (gitignored).

### PyTorch
- **Licence:** BSD-3-Clause — <https://github.com/pytorch/pytorch>

### pycocotools
- **Licence:** FreeBSD — <https://github.com/cocodataset/cocoapi>

### Pillow, NumPy
- **Licences:** MIT-CMU (Pillow), BSD-3-Clause (NumPy)

None of the above is modified or redistributed by this repository; they are ordinary
dependencies. They are listed for transparency, and because the Apache License 2.0 permits
(but does not require) attribution of such components.
