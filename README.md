# Synthetic Face Forensics — AI-Generated / Deepfake Image Detector

A CNN-based system that classifies whether a face photo is **real** or
**AI-generated** (GAN/diffusion synthesis), with frequency-domain (FFT)
analysis and Grad-CAM explainability, served through a Streamlit app.

This README is your step-by-step guide from "unzipped folder" to "trained
model running in a live app." Follow it in order.

---

## 0. What's in this zip

```
deepfake-detector/
├── data/
│   ├── train/{real,fake}/    <- put real dataset here
│   ├── valid/{real,fake}/
│   ├── test/{real,fake}/
│   └── sample/{real,fake}/   <- a few synthetic placeholder images (already included)
├── src/
│   ├── config.py             <- all paths & hyperparameters live here
│   ├── dataset.py            <- dataloaders, transforms
│   ├── models.py             <- BaselineCNN, TransferLearningNet (EfficientNet-B0), FFTAwareNet
│   ├── fft_utils.py          <- frequency-domain analysis utilities
│   ├── gradcam.py            <- Grad-CAM explainability (hand-written, not a library)
│   ├── train.py              <- training loop (baseline / transfer)
│   ├── evaluate.py           <- model comparison + cross-dataset generalization test
│   ├── inference.py          <- shared prediction logic used by the Streamlit app
│   └── make_sample_data.py   <- generates synthetic placeholder images (already run once)
├── notebooks/
│   └── 03_fft_analysis.ipynb <- average-spectrum figure + FFTAwareNet training
├── app/
│   └── app.py                <- the Streamlit app (3 tabs: single image, batch, comparison)
├── outputs/
│   ├── checkpoints/          <- trained model weights land here
│   ├── logs/                 <- training history CSVs, comparison tables
│   ├── gradcam/, fft/        <- saved figures
├── requirements.txt
└── README.md                 <- you are here
```

**Important — what's real vs placeholder right now:**
The `data/` folders currently contain **synthetic procedurally-generated
images** (colored blobs with/without a faint grid pattern), NOT real faces.
They exist only so you can run the entire pipeline once and confirm nothing
is broken before downloading the real ~140k-image dataset, which is several
GB and not something I can include in this zip. Step 2 below replaces them.

---

## 1. Set up your environment

You're on an Intel i5 / 16GB RAM machine — this is enough for CPU training
at the settings already configured in `src/config.py` (128×128 images,
capped dataset sizes). Don't try to train at 224×224 on the full 140k
images on CPU — it'll take unreasonably long. The defaults are already
tuned for your hardware.

```bash
# unzip and enter the project
cd deepfake-detector

# create a virtual environment
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# install dependencies
pip install -r requirements.txt
```

This installs PyTorch (CPU build), torchvision, Streamlit, OpenCV, and the
usual data-science stack. On a fresh machine this download is a few hundred
MB and may take 5–10 minutes depending on your connection.

**Sanity check:**
```bash
python -c "import torch; print('torch OK, version', torch.__version__)"
python -c "import streamlit; print('streamlit OK')"
```

---

## 2. Get the real dataset

1. Go to Kaggle and search **"140k Real and Fake Faces"** (by user `xhlulu`).
   You'll need a free Kaggle account to download.
2. Download and unzip it. You'll get folders roughly like:
   ```
   real_vs_fake/
     train/real/*.jpg
     train/fake/*.jpg
     valid/real/*.jpg
     valid/fake/*.jpg
     test/real/*.jpg
     test/fake/*.jpg
   ```
   (exact naming may vary slightly by version — check the unzipped structure)
3. **Replace** the synthetic placeholders with the real images:
   ```bash
   # from inside deepfake-detector/
   rm -rf data/train/real/* data/train/fake/*
   rm -rf data/valid/real/* data/valid/fake/*
   rm -rf data/test/real/*  data/test/fake/*

   # copy the real dataset in (adjust source paths to wherever you unzipped it)
   cp /path/to/real_vs_fake/train/real/* data/train/real/
   cp /path/to/real_vs_fake/train/fake/* data/train/fake/
   cp /path/to/real_vs_fake/valid/real/* data/valid/real/
   cp /path/to/real_vs_fake/valid/fake/* data/valid/fake/
   cp /path/to/real_vs_fake/test/real/*  data/test/real/
   cp /path/to/real_vs_fake/test/fake/*  data/test/fake/
   ```
4. You do **not** need to use all 140k images. `src/config.py` already caps
   how many images per class get loaded during training:
   ```python
   MAX_TRAIN_IMAGES_PER_CLASS = 6000
   MAX_VALID_IMAGES_PER_CLASS = 1200
   MAX_TEST_IMAGES_PER_CLASS = 1200
   ```
   You can drop the *entire* dataset into the folders — the loader randomly
   samples up to this cap, so copying everything is fine and actually
   simpler than hand-picking a subset. Lower these numbers further (e.g.
   3000/600/600) if training feels too slow.

---

## 3. (Optional) Re-run the synthetic data generator

Already done once, but if you ever want to regenerate the placeholder
images (e.g. after clearing `data/` to load the real dataset, then want to
go back to a quick smoke test):

```bash
python -m src.make_sample_data
```

This writes ~40 synthetic real-like and ~40 synthetic fake-like images into
`data/train`, `data/valid`, `data/test`. It runs in seconds and needs no
GPU. Use it any time you want to confirm the code runs before committing to
a full real-data training run.

---

## 4. Train the baseline CNN

```bash
python -m src.train --model baseline
```

What happens:
- Loads data from `data/train`, `data/valid`
- Trains a from-scratch 5-block CNN for up to 12 epochs (early stops if
  validation loss stops improving for 4 epochs)
- Saves the best checkpoint to `outputs/checkpoints/baseline_best.pt`
- Saves epoch-by-epoch metrics to `outputs/logs/baseline_history.csv`
- Reports final test-set accuracy/F1/AUC at the end

**Expected time on your hardware:** with the synthetic smoke-test data,
seconds. With the real dataset at the default caps (6000 images/class),
expect roughly 1–3 minutes per epoch on an i5 — so perhaps 15–35 minutes
total depending on how many epochs run before early stopping. This varies
by exact CPU, so treat it as a rough guide, not a guarantee.

---

## 5. Train the transfer-learning model

```bash
python -m src.train --model transfer
```

This fine-tunes an EfficientNet-B0 (pretrained on ImageNet) on your data.
It will download the pretrained weights the first time (~20MB, needs
internet). Transfer learning usually needs fewer epochs to converge — the
default is 8.

**Why both models?** This comparison is one of the strongest parts of your
project. Your baseline shows you can build a CNN from first principles;
the transfer-learning model shows you understand when and how to leverage
pretrained features. Presenting both, with their accuracy/F1/AUC compared,
is exactly the kind of depth a project defense rewards.

---

## 6. Compare the two models on the test set

```bash
python -m src.evaluate --models baseline transfer
```

This prints accuracy, precision, recall, F1, AUC, and a confusion matrix
for each model, and saves a tidy comparison table to
`outputs/logs/model_comparison.csv`. This table is also what the
Streamlit app's "Model Comparison" tab displays.

---

## 7. Frequency-domain (FFT) analysis — the differentiator

Open the notebook:
```bash
jupyter notebook notebooks/03_fft_analysis.ipynb
```
(install jupyter if needed: `pip install jupyter`)

Run it top to bottom. It will:
1. Compute and plot the **average FFT spectrum** across real images vs
   fake images side by side, plus their difference — this is a genuinely
   compelling figure for your report, showing the periodic artifacts GAN
   upsampling leaves behind in the frequency domain.
2. Build a 4-channel (RGB + FFT-magnitude) version of your dataset.
3. Train `FFTAwareNet` on that 4-channel input.
4. Report its test accuracy/F1/AUC so you can directly compare: did adding
   explicit frequency information help, hurt, or make no difference versus
   the plain-RGB baseline/transfer models? All three outcomes are
   interesting and worth discussing — you don't need it to "win" for the
   experiment to be a strong part of your project.

---

## 8. Cross-dataset generalization test (optional but strong)

If you want the standout "does this generalize to *other* fake-generation
methods" experiment:

1. Gather a small set (50-200 images) of diffusion-model-generated faces
   (e.g. from a Stable Diffusion face dataset, or generate your own with
   any text-to-image tool using prompts like "portrait photo of a person")
2. Put them in a new folder: `data/cross_test_diffusion/fake/` and add an
   equal number of real photos to `data/cross_test_diffusion/real/`
3. Run:
   ```bash
   python -m src.evaluate --models baseline transfer --cross_test data/cross_test_diffusion
   ```
4. Compare the accuracy on this set vs the standard GAN-based test set.
   A noticeable accuracy drop is the expected, interesting result — it
   demonstrates that detectors trained on one generator family don't
   automatically generalize to another, which is a real, current problem
   in this research area and a great talking point for your viva.

---

## 9. Launch the Streamlit app

```bash
streamlit run app/app.py
```

This opens in your browser (usually `http://localhost:8501`). Features:
- **Single Image tab:** upload a face photo, get a verdict (Real / AI-Generated),
  confidence score, class probability bar chart, Grad-CAM heatmap overlay,
  and FFT spectrum visualization.
- **Batch Analysis tab:** upload many images at once, get a results table
  you can download as CSV.
- **Model Comparison tab:** run one image through every trained model side
  by side, plus a table of your saved test-set comparison metrics.

The app automatically detects which models you've trained (by checking
`outputs/checkpoints/`) and only offers those in the dropdown.

---

## 10. Suggested write-up structure for your report

1. **Problem statement** — why deepfake/AI-image detection matters right now
2. **Dataset** — 140k Real and Fake Faces, class balance, preprocessing
3. **Methodology**
   - Baseline CNN architecture and rationale
   - Transfer learning (EfficientNet-B0) and rationale
   - Frequency-domain hypothesis and FFTAwareNet
4. **Results** — comparison table, ROC curves, confusion matrices, the
   average-spectrum figure from the notebook
5. **Explainability** — Grad-CAM examples (include both a correctly and
   an incorrectly classified example — discussing failure cases is valued)
6. **Generalization test** — GAN-trained model vs diffusion-generated test
   set, and what the accuracy drop implies
7. **Limitations & future work** — dataset bias (mostly frontal, well-lit
   faces), generalization to other generators, real-time/video deepfakes
   as a natural extension
8. **Demo** — the Streamlit app

---

## Troubleshooting

- **`ModuleNotFoundError: No module named 'src'`** — make sure you're
  running commands from the project root (`deepfake-detector/`), not from
  inside `src/`. Commands use `python -m src.train`, not `python train.py`.
- **Training feels extremely slow** — lower `MAX_TRAIN_IMAGES_PER_CLASS` in
  `src/config.py`, or lower `IMG_SIZE` from 128 to 96.
- **`No trained model checkpoints found` in the Streamlit app** — you need
  to run at least `python -m src.train --model baseline` before the app
  has anything to load.
- **EfficientNet download fails** — the transfer-learning model needs
  internet access the first time to fetch pretrained ImageNet weights. If
  you're offline, train `baseline` only.
- **Out-of-memory / system freezing** — lower `BATCH_SIZE` in
  `src/config.py` from 32 to 16 or 8.

---

## A note on scope and honesty for your viva

Be upfront that:
- This is trained primarily on **GAN-generated** faces (StyleGAN via the
  140k dataset). It is one detector for one family of generation methods,
  not a universal deepfake detector.
- The cross-dataset test (Step 8) is exactly the right way to demonstrate
  you understand this limitation rather than overselling the tool.
- "Real-world deployment" would need continual retraining as generation
  methods evolve — this is an active research problem, and your project
  demonstrating awareness of it (rather than claiming a solved problem)
  will read as more credible, not less.
