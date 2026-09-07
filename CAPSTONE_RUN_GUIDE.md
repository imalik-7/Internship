# Capstone finish guide

The remaining capstone notebooks are already populated. Run them in this order so every metric is real and reproducible.

## Before you start

In Google Colab, make sure the private Hugging Face read token is stored in **Secrets** as `HF_TOKEN` and notebook access is enabled. Never paste the token into a cell or commit it.

## 1. Week 5 — model

Open:

https://colab.research.google.com/github/imalik-7/Internship/blob/main/work/notebooks/w05_model.ipynb

Then:

1. Runtime -> Restart session
2. Runtime -> Run all
3. Confirm there are no red errors.
4. Confirm the output shows the held-out base rate, baseline + model comparison table, best learned model, feature importance, and three wrong cases.
5. File -> Save a copy in GitHub -> `imalik-7/Internship` -> overwrite `work/notebooks/w05_model.ipynb`.

## 2. Week 6 — validation audit

Open:

https://colab.research.google.com/github/imalik-7/Internship/blob/main/work/notebooks/w06_validation_audit.ipynb

Run all and save it back to the same repo/path. Confirm the random-vs-grouped table, client overlap = 0, leakage audit, and honest claim are visible.

## 3. Week 7 — action playbook

Open:

https://colab.research.google.com/github/imalik-7/Internship/blob/main/work/notebooks/w07_action_playbook.ipynb

Run all and save it back. Confirm the top ranked queue, reason codes, action labels, human checks, monitoring reference, and paper exports are visible.

## 4. Capstone notebook

Open:

https://colab.research.google.com/github/imalik-7/Internship/blob/main/work/notebooks/capstone.ipynb

Run all and save it back. This is the notebook that produces the final same-split model comparison, charts, feature importance, top recommendations, reproducibility receipts, and communication cuts.

## Finish line

After the four notebooks have been executed and saved back to GitHub, give ChatGPT the repo URL again. ChatGPT can read the real notebook outputs, build the final public research-paper page, deploy-ready GitHub Pages files, and create `submission/paper_url.txt` with the real paper URL.

Final capstone submission will be the repository URL:

https://github.com/imalik-7/Internship
