# Fake News Headline Classification

A small Streamlit demo serving a trained TextCNN that predicts whether a headline matches the `fake` or `real` label in the historical WELFake dataset. It is a university project demonstration, not a fact-checker.

## Run locally

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

The app loads `cnn_checkpoint.npz` and `vocabulary.npz`. It does not train when launched. Open the localhost URL printed by Streamlit.

## Deploy on Streamlit Community Cloud

1. Open Streamlit Community Cloud and sign in with GitHub.
2. Select **Create app** and choose this repository, branch `main`, and file `app.py`.
3. Deploy. Streamlit installs the packages from `requirements.txt` and launches the app.

## Train again

The trained app does not need the training dataset. To retrain, download the WELFake parquet file from the dataset page below, save it beside `train_fake_news.py` as `WELFake.parquet`, install `requirements.txt`, then run:

```powershell
python train_fake_news.py
```

Dataset: [WELFake on Hugging Face](https://huggingface.co/datasets/davanstrien/WELFake). The dataset has 72,134 records before cleaning; this experiment uses 63,633 unique usable records and a stratified 70/15/15 train/validation/test split. Label mapping is `0 = fake`, `1 = real`.

## Model results

The app uses headline text only (up to 48 tokens). The TextCNN has 32-dimensional embeddings and parallel convolution widths 3, 4, and 5, with 24 filters per width. Eight epochs completed. The best validation loss was 0.2428 at epoch 1; validation loss increased afterward, indicating overfitting. On the held-out test split, the best-validation checkpoint achieved 90.65% accuracy, 0.9006 F1, and 0.9674 ROC-AUC. See `results.json`, `training_curve.png`, and `confusion_matrices.png` for details.

## Important limitation

This model predicts labels from a historical dataset, not factual accuracy. It can learn source, topic, and writing-style patterns and may perform poorly on current news or new publishers. Its probability is a model score, not a calibrated chance that a claim is true. Verify information with reliable sources.
