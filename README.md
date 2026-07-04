# Heart Disease Risk Predictor

A Streamlit app that serves a Random Forest pipeline (`StandardScaler` → `RandomForestClassifier`) to estimate heart disease risk from 13 clinical inputs.

## Files
- `app.py` — the Streamlit application
- `rf_pipeline.pkl` — trained sklearn pipeline (scikit-learn 1.6.1)
- `requirements.txt` — pinned dependencies matching the training environment

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Streamlit Community Cloud (free)

1. Push these files to a public (or private, with Streamlit connected) GitHub repo.
2. Go to https://share.streamlit.io → **New app**.
3. Pick the repo, branch, and set the main file path to `app.py`.
4. Click **Deploy**. Streamlit installs `requirements.txt` automatically and the app is live at a `*.streamlit.app` URL within a couple of minutes.

**Important:** `rf_pipeline.pkl` is ~1.1 MB — well under GitHub's 100 MB limit, so a normal `git add`/`commit`/`push` works; no Git LFS needed.

## Deploy on Hugging Face Spaces (alternative)

1. Create a new Space → SDK: **Streamlit**.
2. Upload `app.py`, `rf_pipeline.pkl`, and `requirements.txt` to the Space repo.
3. The Space builds and launches automatically.

## Notes on the model
- `requirements.txt` pins `scikit-learn==1.6.1` because that's the exact version the pipeline was trained/pickled with. Using a different sklearn version will still run, but you may see `InconsistentVersionWarning` and, in rarer cases, subtly different predictions — pinning avoids both.
- The app validates that `rf_pipeline.pkl` sits next to `app.py` and shows a clear error if it's missing, rather than crashing.
- Inputs are grouped in a form so the app only recomputes once, on submit, instead of on every keystroke/click.

## Disclaimer
This tool is for educational purposes only and is **not** a substitute for professional medical advice, diagnosis, or treatment.
