# Licence and attribution

Three sets of material with three different owners, so three different sets of terms apply.

| Material | Owner | Terms |
|---|---|---|
| Code, weights, documentation, annotation files | this project | Apache-2.0 ([`LICENSE`](LICENSE)) |
| `demo/nih_images/` — 11 radiographs | NIH Clinical Center | Dataset terms ([`DATA_LICENSE.md`](DATA_LICENSE.md)) |
| `demo/ort/` — onnxruntime-web | Microsoft | MIT ([`demo/ort/LICENSE`](demo/ort/LICENSE)) |

[`NOTICE`](NOTICE) and [`THIRD-PARTY-NOTICES.md`](THIRD-PARTY-NOTICES.md) spell out which terms
reach which files.

## Warranty and liability

The work is provided **"AS IS", WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND**, express or
implied. Apache-2.0 §7 places the risk of using or redistributing it on the user, and §8 limits
contributor liability for any damages arising from its use, under any legal theory — including
tort and negligence — except where the law requires otherwise (for example deliberate or grossly
negligent acts). See [`LICENSE`](LICENSE) for the full text.

In short: you are responsible for deciding whether this is fit for your purpose.

Those disclaimers do not, and cannot, cover everything. They do not bind anyone who has not
accepted the licence, they do not override statutory rights, and they say nothing about
regulatory obligations. **Clinical and regulatory liability is not a copyright question** — if
this were deployed in a diagnostic pathway, medical device regulation and professional
responsibility would apply regardless of what this file says.

## Not a medical device

Research and educational use only. No regulatory clearance, no clinical validation, no
prospective testing. Accuracy is low (see the [model card](docs/MODEL_CARD.md)). Do not use for
diagnosis, screening or triage.

## Dataset attribution

NIH ChestX-ray14 asks that redistribution carry three things: a link to the
[NIH download site](https://nihcc.app.box.com/v/ChestXray-NIHCC), a citation of the CVPR 2017
paper, and acknowledgement of the NIH Clinical Center as the data provider. All three are in
[`DATA_LICENSE.md`](DATA_LICENSE.md).
