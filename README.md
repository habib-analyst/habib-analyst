# Habib Ur Rehman

ML/AI engineer working on healthcare AI. I build models, then build the tools that check them — and I release both with the evidence attached: tests, evaluation scripts, and the failure cases included.

## What I do

- ML/AI Engineer at SANWA SYSTEM SERVICE CO. LTD (Jul 2025 – present): training, evaluating, and deploying ML models on custom datasets, including ML/AI work on ERP systems.
- Before that, Undergraduate Research Assistant at the Center of Data Science, GCUF (Dec 2024 – Jun 2025), working on transformer-based medical imaging.
- BS in Data Analytics, Government College University Faisalabad (2021 – 2025).
- Core stack: Python, PyTorch, TensorFlow, Hugging Face, LangChain / LangGraph, FastAPI, Docker, SQL.

Most of my recent open-source work is about one question: how do you know whether a classifier deserves trust? The libraries below each take one part of that question. The models are small, trained from scratch, and published with scripts that re-derive their headline metrics from the released weights.

## Open-source libraries & models

### Libraries

Eight small, single-purpose Python packages. Each ships with tests and a worked example, and each README states plainly what the tool does not do.

| Library | What it does |
|---|---|
| [ViewConsensus](https://github.com/habib-analyst/viewconsensus) | Runs a trained image classifier over a panel of transformed views of the same input, and returns the consensus probability, the view-spread as uncertainty, the per-view votes, and a verdict: confident, or unsure — defer to a human. |
| [DatasetReceipt](https://github.com/habib-analyst/datasetreceipt) | Scans an image dataset before training for cross-split leakage, metadata shortcuts, and background-only label signals, and returns one verdict: GO, CAUTION, or NO-GO. |
| [ViewTrainer](https://github.com/habib-analyst/viewtrainer) | The train-time sibling of ViewConsensus. Trains a PyTorch classifier so its augmented views agree, then predicts with the same panel, uncertainty, votes, and defer verdict, behind one `fit` / `evaluate` / `predict` Trainer. |
| [SliceSniff](https://github.com/habib-analyst/slicesniff) | Mines a classifier's predictions for the slices where it is confidently wrong, and returns ranked, reproducible failure cards: the slice definition, accuracy against confidence, and the exact example indices. |
| [ProbeShift](https://github.com/habib-analyst/probeshift) | Measures how little an image must change before a prediction flips — the smallest effective brightness, contrast, blur, noise, shift, or rotation — for any predict function, and summarises the result as a fragility score. |
| [DriftReceipt](https://github.com/habib-analyst/driftreceipt) | Compares two windows of logged predictions from a deployed classifier and returns a GO / CAUTION / NO-GO drift receipt: what shifted, in which metadata slice, and the exact rows driving the change. |
| [ChoiceShift](https://github.com/habib-analyst/choiceshift) | Permutes the options of multiple-choice benchmark questions and measures how much an LLM's answers move: stability score, flip rate, and the accuracy range a single benchmark number hides. |
| [ShotShift](https://github.com/habib-analyst/shotshift) | Permutes the demonstrations in a few-shot prompt and measures how much a classifier's predictions move, with per-query flip cards and leave-one-out example influence. |

### Models & live demos

Seven models, each released on GitHub and Hugging Face with a live demo. Every headline number below is the number on the model card, re-derived from the released weights by the verification script in its repository.

| Model | Task | Headline result | Links |
|---|---|---|---|
| SonoCNN | BreastMNIST — benign vs. malignant classification of 28×28 breast-ultrasound images | Test accuracy 0.8718 · ROC AUC 0.8709 | [GitHub](https://github.com/habib-analyst/sonocnn) · [Hugging Face](https://huggingface.co/habib-analyst/sonocnn) · [Live demo](https://huggingface.co/spaces/habib-analyst/sonocnn-demo) |
| PathoCNN | PathMNIST — 9-class classification of colorectal histology patches | Test accuracy 0.8712 · macro AUC 0.9855, on a test split collected at a different clinical center | [GitHub](https://github.com/habib-analyst/pathocnn) · [Hugging Face](https://huggingface.co/habib-analyst/pathocnn) · [Live demo](https://huggingface.co/spaces/habib-analyst/pathocnn-demo) |
| Sukhan-Nano | Urdu language model (5.57M parameters), trained from scratch on a documented, openly licensed corpus | Held-out perplexity 48.48 | [GitHub](https://github.com/habib-analyst/sukhan-nano) · [Hugging Face](https://huggingface.co/habib-analyst/sukhan-nano) · [Live demo](https://huggingface.co/spaces/habib-analyst/sukhan-nano-demo) |
| Nuqta-Nano | Roman Urdu → Urdu script transliteration, one word at a time | Held-out corpus CER 0.2020 · exact-match 0.3606 on unseen words | [GitHub](https://github.com/habib-analyst/nuqta-nano) · [Hugging Face](https://huggingface.co/habib-analyst/nuqta-nano) · [Live demo](https://huggingface.co/spaces/habib-analyst/nuqta-nano-demo) |
| Awaaz-Nano | Urdu speech recognition — character-level CTC trained from scratch on FLEURS ur_pk, no pretrained weights | Held-out test CER 0.5131 · WER 0.9852 — honestly weak; an educational end-to-end ASR run, not usable transcription | [GitHub](https://github.com/habib-analyst/awaaz-nano) · [Hugging Face](https://huggingface.co/habib-analyst/awaaz-nano) · [Live demo](https://huggingface.co/spaces/habib-analyst/awaaz-nano-demo) |
| VesselCNN | VesselMNIST3D — vessel vs. aneurysm classification of 28×28×28 brain-vessel volumes with a tiny 3D CNN | Test accuracy 0.7539 — below the 0.8874 majority baseline; aneurysm recall 0.4884. A measured failure with an interpretable cause, published as-is | [GitHub](https://github.com/habib-analyst/vesselcnn) · [Hugging Face](https://huggingface.co/habib-analyst/vesselcnn) · [Live demo](https://huggingface.co/spaces/habib-analyst/vesselcnn-demo) |
| OrganCNN | OrganMNIST3D — 11-class classification of 28×28×28 organ volumes with a tiny 3D CNN | Test accuracy 0.8574 · macro AUC 0.9860; 61% of its errors are left/right confusions (femur-left recall 0.5692) | [GitHub](https://github.com/habib-analyst/organcnn) · [Hugging Face](https://huggingface.co/habib-analyst/organcnn) · [Live demo](https://huggingface.co/spaces/habib-analyst/organcnn-demo) |

The model cards also list the weak spots — PathoCNN's weakest class, Nuqta-Nano's systematic letter errors — rather than only the headline numbers.

## Research & background

- Healthcare-AI research at the Center of Data Science, GCUF: transformer-based medical imaging pipelines (Vision Transformers with Perceiver IO), experiments, documentation, and manuscript writing.
- Other 2025 research: multimodal deepfake detection (Multimodal-FNet — Fourier token mixing for audio-video inconsistency) and a federated-learning security framework combining GANs, blockchain, and differential privacy.
- I maintain a reproducibility benchmark for MedMNIST v2 — 12 2D and 6 3D datasets across multiple seeds — because reported numbers should be repeatable: [MedMNIST_v2_Reproducibility_Benchmark](https://github.com/habib-analyst/MedMNIST_v2_Reproducibility_Benchmark).

## Contact

- GitHub: [github.com/habib-analyst](https://github.com/habib-analyst)
- LinkedIn: [linkedin.com/in/hur-dev](https://www.linkedin.com/in/hur-dev)
- Website: [habib.top](https://habib.top)
- Email: [habib.gcuf.edu@gmail.com](mailto:habib.gcuf.edu@gmail.com)
