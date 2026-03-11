# Extended EDA and Modeling Workflow

This workflow turns the ideas in `ANALYSIS_AND_IDEAS.md` into a repeatable `uv`-based script:

```bash
cd /home/runner/work/ceu-ml/ceu-ml
python -m uv run python comps/uni_students/scripts/run_extended_analysis.py
```

This default command is optimized for a quick submission run:

- generates the markdown EDA summary
- runs a compact **3-fold tuned XGBoost** search
- runs the neural-network baseline as well
- writes `submission_xgb_tuned.csv` and `submission_mlp.csv`

If you only want the faster XGBoost-only path:

```bash
cd /home/runner/work/ceu-ml/ceu-ml
python -m uv run python comps/uni_students/scripts/run_extended_analysis.py --skip-mlp
```

## EDA steps and why they matter

1. **Outcome mix by module**
   - The competition mixes modules with very different assessment setups.
   - Checking the per-module outcome distribution helps decide whether module-aware features or models are warranted.

2. **Assessment missingness by module**
   - Missing scores are not all the same signal.
   - `EEE`/`GGG` rows can be structurally missing because no assessment was due, while missing scores in other modules can indicate non-submission behavior.

3. **Weekly click trajectories**
   - Looking at week-by-week engagement captures decline vs. steady activity better than total clicks alone.
   - The quick workflow keeps this in the EDA report so it can inform model choice without adding extra feature-engineering overhead.

4. **`last_access_day` and submission behavior**
   - Previous model sheets showed `last_access_day` as the strongest signal.
   - The extended report keeps this visible and pairs it with `num_assessments`, score, and lateness summaries by outcome.

5. **Correlation review**
   - Highly redundant click features can inflate instability in linear and neural models.
   - The report surfaces the strongest numeric correlations so future pruning stays evidence-based.

## Modeling workflow

- **Tuned XGBoost**
  - Uses the cleaned, unscaled feature matrix.
  - Runs a compact grid search with **3-fold stratified CV** by default for speed.
  - Writes `models_info/TunedXGBoost_info.md` and `submissions/submission_xgb_tuned.csv`.

- **Neural network (MLPClassifier)**
  - Uses the same cleaned feature matrix but applies scaling inside a sklearn pipeline.
  - Runs a compact hyperparameter search with stratified CV and early stopping.
  - It is included by default so the standard workflow produces both submission files.
  - Use `--skip-mlp` when you want the faster XGBoost-only path.
  - Writes `models_info/NeuralNetwork_info.md` and `submissions/submission_mlp.csv`.

- **Bootstrap summary**
  - Each model sheet includes a bootstrap 95% confidence interval from the fold accuracies.
  - This gives a more stable sense of uncertainty than a single random validation split.

## Outputs

- `reports/EXTENDED_EDA_REPORT.md`
- `models_info/TunedXGBoost_info.md`
- `models_info/NeuralNetwork_info.md`
- `submissions/submission_xgb_tuned.csv`
- `submissions/submission_mlp.csv`

If the Kaggle train/test CSVs are not present under `data/uni_students/`, the script fails fast with a clear message so the data can be downloaded first.
