# A3 – Predicting Car Price (classification) · ML August 2026

**Submitted by:** Anish Lal Manandhar (st125973)
**Live app:** http://192.41.170.105:8051/ *(A3 container – the A2 app stays on port 8050)*
**MLflow:** experiment `st125973-a3` on http://mlflow.ml.brain.cs.ait.ac.th/ · registered model `st125973-a3-model` (Staging)

The Car Price dataset from A1/A2 is turned into a **4-class classification problem**. A multinomial
**Logistic Regression written from scratch** gets classification metrics (Task 1) and an optional
**ridge (L2) penalty** (Task 2). The experiments are logged to the course **MLflow** server, the best model is
registered, and the web app is tested and deployed automatically with **GitHub Actions** (Task 3).

---

## Repository structure

```
A3_Predicting_Car_Price_st125973.ipynb   notebook: cleaning → model → experiments → export
Cars.csv                                 dataset
app/                                     the web application
├── code/app.py                          Dash web app
├── code/car_price_model.py              loads the exported model; preprocessing + prediction
├── model/car_price_a3.json              exported best model (weights + fitted encoders / scaler)
├── tests/test_model.py                  unit tests (pytest)
├── Dockerfile
├── docker-compose.yaml
└── requirements.txt
.github/workflows/ci-cd.yml              CI/CD: run the tests → deploy if they pass
README.md
```

---

## 1. Data and ML pipeline

`Load → Clean → EDA → Label → Split → Encode + Scale (fit on train only) → Model → CV experiments (MLflow) → Test once → Export → Register`

| Step | What was done |
|---|---|
| Cleaning (as A1/A2) | 1202 duplicates removed; CNG/LPG cars and *Test Drive Cars* removed; units stripped from `mileage`, `engine`, `max_power`; `torque` dropped; `brand` = first word of `name`; 201 rows (2.9 %) with missing values dropped → **6626 cars** |
| Label | `selling_price` → 4 classes with `pd.cut` at the quartiles: **0** ≤ ₹250,000 < **1** ≤ ₹422,000 < **2** ≤ ₹650,000 < **3** (1705 / 1608 / 1771 / 1542 cars) |
| Split | stratified 80 / 20 → 5300 train, 1326 test (same class shares in both) |
| Encoding (after the split) | `owner` ordinal map (1–4), `transmission` and `fuel` 0/1 maps, `seller_type` `LabelEncoder` (fit on train), `brand` `OneHotEncoder(handle_unknown='ignore')` fit on train – the only *Ashok* car is in the test set and becomes all zeros instead of crashing |
| Scaling | `StandardScaler` fit on the train set for `year, km_driven, mileage, engine, max_power`; the test set is only transformed |
| Model input | 40 features + intercept = 41 inputs, Y one-hot `(m, 4)` |

The test set is used **once**, for the final model. The first run and the Task 1 checks use a validation split
taken from the training data; model selection uses stratified 3-fold cross-validation on the training data.

---

## 2. Task 1 – Classification metrics from scratch

Added to the `LogisticRegression` class:

| Method | Formula |
|---|---|
| `accuracy(yhat, y)` | correct predictions / all predictions |
| `confusion_matrix(yhat, y, c)` | TP, FP, TN, FN of class *c* (one-vs-rest) |
| `precision(yhat, y, c)` | TP / (TP + FP) |
| `recall(yhat, y, c)` | TP / (TP + FN) |
| `f1(yhat, y, c)` | 2·P·R / (P + R) |
| `macro_precision / macro_recall / macro_f1` | plain average over the 4 classes |
| `weighted_precision / weighted_recall / weighted_f1` | Σ (support_c / total) · score_c |
| `classification_report(yhat, y)` | the same table as sklearn |

* **Comparison with sklearn:** identical numbers (absolute difference 0) on mock-up data (300 random labels,
  20/30/20/30 % classes), on the validation predictions and on the test predictions.
* **Weighted average:** the weights `support_c / total` already sum to 1, so the weighted score is
  `0.2·p0 + 0.3·p1 + 0.2·p2 + 0.3·p3` – dividing by 4 again (as in the PDF's example) gives a value about
  4× too small (0.19 instead of sklearn's 0.77 on the mock-up data).
* **What does *support* mean?** The number of samples that **truly belong to each class** in the evaluated data
  (`y_true`). It does not depend on the predictions; in the average rows it is the total number of samples. It shows
  how reliable each per-class score is and it is the weight used in the weighted average.

---

## 3. Task 2 – Ridge Logistic Regression

$$J(\theta) = -\sum_{i=1}^{m} y^{(i)}\log(h^{(i)}) + \lambda\sum_{j=1}^{n}\theta_j^2 \qquad \frac{\partial J}{\partial\theta} = X^\top(H-Y) + 2\lambda\theta$$

* Same design as `03 - Regularization.ipynb`: a `RidgePenalty` class with `__call__` (added to the loss) and
  `derivation` (added to the gradient), and a `NoPenalty` class for the plain model.
* **The user chooses:** `LogisticRegression(k, n, "batch", alpha=0.1, use_penalty=True, l=0.01)` or the shortcut
  `RidgeLogisticRegression(k, n, "batch", alpha=0.1, l=0.01)`; `use_penalty=False` (default) is plain logistic regression.
* The intercept row of `W` is **not** penalised (the sum starts at j = 1).
* The data term is averaged over the batch (÷ m), as in the class code, so the code's λ equals the PDF's λ / m.

Other fixes to the class code: gradient divided by m, numerically stable softmax, `log` clipped, seeded weights,
minibatch draws `batch_size` random rows (the old slice could be 1–2 rows), stochastic GD visits every row once per
pass (the old check compared `i` instead of `idx`).

---

## 4. Experiments (logged to MLflow)

36 runs: method {batch, minibatch, sto} × learning rate {0.1, 0.01, 0.001} × penalty {none, ridge λ = 0.001, 0.01, 0.1},
5000 iterations each, scored with stratified 3-fold CV on the training set, then refit on all training data
(the refit model is saved with each run). Logged per run: parameters, `cv_accuracy`, `cv_accuracy_std`, `cv_macro_f1`,
`cv_weighted_f1`, `train_accuracy`, `final_train_loss` and the model. The dataset is not logged.

| Run | CV accuracy | CV macro F1 | Train accuracy |
|---|---|---|---|
| **minibatch-lr0.1-nopenalty** (best) | **0.7341** | **0.7364** | 0.7374 |
| batch-lr0.1-nopenalty | 0.7321 | 0.7340 | 0.7385 |
| batch-lr0.1-ridge0.001 | 0.7274 | 0.7292 | 0.7385 |
| minibatch-lr0.1-ridge0.001 | 0.7270 | 0.7291 | 0.7370 |
| batch-lr0.01-nopenalty | 0.7121 | 0.7123 | 0.7166 |
| batch-lr0.1-ridge0.01 | 0.7087 | 0.7079 | 0.7192 |
| sto-lr0.01-nopenalty (best sto) | 0.6719 | 0.6547 | 0.7028 |
| batch-lr0.1-ridge0.1 | 0.6402 | 0.6165 | 0.6426 |
| batch-lr0.001-nopenalty | 0.6151 | 0.5997 | 0.6143 |

* **Learning rate** matters most: α = 0.1 ≈ 0.73, α = 0.01 ≈ 0.71, α = 0.001 ≈ 0.60 (5000 small steps do not reach the minimum).
* **Batch ≈ minibatch**; **stochastic** is worse (one row per step, only ~1.4 passes over the data, noisy weights).
* **Ridge** does not help here: the plain model does not overfit (train ≈ CV accuracy), so the penalty only adds bias
  (CV macro F1 change: λ = 0.001 → −0.005 to −0.007, λ = 0.01 → about −0.03, λ = 0.1 → −0.12 to −0.15).

### Final model on the test set (used once)

| | precision | recall | f1-score | support |
|---|---|---|---|---|
| class 0 | 0.8765 | 0.8534 | 0.8648 | 341 |
| class 1 | 0.6718 | 0.6801 | 0.6759 | 322 |
| class 2 | 0.6546 | 0.7175 | 0.6846 | 354 |
| class 3 | 0.8607 | 0.7799 | 0.8183 | 309 |
| **accuracy** | | | **0.7579** | 1326 |
| macro avg | 0.7659 | 0.7577 | 0.7609 | 1326 |
| weighted avg | 0.7639 | 0.7579 | 0.7600 | 1326 |

Almost all mistakes are between neighbouring classes (only 10 of 321 skip a class). Most important features:
`year`, `max_power`, `fuel`, `engine`, then the brands.

---

## 5. Task 3 – Deployment

### Objective 1 – log the experiment to the CSIM MLflow server
1. Be on the AIT network (or VPN) – the server is not reachable from outside.
2. `pip install mlflow`. The notebook prints the server's and your MLflow version; if they have a different
   major version and logging the model fails, install the server's version: `pip install mlflow==<server version>`
   (the server version is also shown at http://mlflow.ml.brain.cs.ait.ac.th/version).
3. If the server asks for a login, set `MLFLOW_TRACKING_USERNAME` / `MLFLOW_TRACKING_PASSWORD` in the MLflow
   configuration cell (do not commit real passwords).
4. *Restart & Run All*. The experiment cell (about 5 minutes) logs the 36 runs to the experiment **`st125973-a3`**;
   the best run also gets the `test_*` metrics and the tag `best_model = true`.

### Objective 2 – register the best model (Staging)
The *Register* cell registers the best run's model as **`st125973-a3-model`** and moves the new version to **Staging**
(it also sets the alias `staging`, which newer MLflow versions show instead of stages). In the UI:
*Experiments → st125973-a3 →* best run *→ Artifacts → model → Register model →* name `st125973-a3-model`,
then *Models → st125973-a3-model →* version *→ Stage: Staging*.

The web app uses the same best model, exported by the notebook to `app/model/car_price_a3.json` together with the
fitted encoders and scaler, so the container does not depend on the MLflow server being reachable.

### Objective 3 – CI/CD with GitHub Actions
**Unit tests** (`app/tests/test_model.py`):
1. `test_model_takes_expected_input` – a raw car becomes a `(1, 41)` array (intercept + 40 features) without missing
   values, and the model accepts it.
2. `test_model_output_has_expected_shape` – for 3 cars the model returns 3 labels in {0, 1, 2, 3} and a `(3, 4)`
   probability table whose rows sum to 1.
3. Two extra checks: an input with the wrong number of features is refused; a car with every field empty is filled
   with the training defaults and still gets one valid prediction.

**Workflow** (`.github/workflows/ci-cd.yml`): on every push to `main`
* job **test** – installs `app/requirements.txt` + pytest on Python 3.10 and runs `pytest -v app/tests`;
* job **deploy** – runs only if *test* passed: connects to the VM over SSH, clones / pulls the repository and runs
  `docker compose up -d --build` in `app/`.

**One-time setup**
1. Create a key pair for GitHub Actions (on your computer): `ssh-keygen -t ed25519 -C "github-actions-a3" -f gha_a3 -N ""`
2. Add the **public** key to the VM: append the content of `gha_a3.pub` to `~/.ssh/authorized_keys` of your VM user.
3. GitHub → *Settings → Secrets and variables → Actions → New repository secret*:
   `SSH_HOST` (e.g. `192.41.170.105`), `SSH_USER` (your VM user), `SSH_PRIVATE_KEY` (the content of `gha_a3`),
   optionally `SSH_PORT`.
4. On the VM your user must be able to run Docker without `sudo` (`sudo usermod -aG docker $USER`, then log in again).
   The repository must be public (or the VM must have access to it) so the VM can `git pull`.
5. Push to `main` → *Actions* tab: **test** ✓ → **deploy** ✓ → the app runs on port **8051**.

*If GitHub's servers cannot reach the VM over SSH* (e.g. SSH is only open inside AIT), install a **self-hosted runner**
on the VM (*Settings → Actions → Runners → New self-hosted runner → Linux*, run the shown commands on the VM) and
replace the deploy job with:

```yaml
  deploy:
    needs: test
    runs-on: self-hosted
    steps:
      - uses: actions/checkout@v4
      - name: Rebuild and restart the container
        run: |
          cd app
          docker compose up -d --build
```

---

## 6. Run it yourself

```bash
# notebook
pip install numpy pandas matplotlib seaborn scikit-learn mlflow
jupyter notebook A3_Predicting_Car_Price_st125973.ipynb

# unit tests
pip install -r app/requirements.txt pytest
pytest -v app/tests

# web app without Docker -> http://localhost:8050
python app/code/app.py

# web app with Docker -> http://localhost:8051
cd app
docker compose up -d --build
```

## Screenshots
*(add after running on the AIT network)*
* MLflow – runs of the experiment `st125973-a3`
* MLflow – `st125973-a3-model` in Staging
* GitHub Actions – successful *test* and *deploy* jobs
* The deployed web app
