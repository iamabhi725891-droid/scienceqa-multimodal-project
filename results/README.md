# ScienceQA-IMG experimental results

These files are *measured outputs* produced by the notebook, not manually entered values.

- model_metrics.csv: accuracy, cross entropy, Brier, ECE, timing and memory by model and seed
- learning_history.csv and learning_curves.png: train/validation learning curves
- test_predictions.csv: paired held-out predictions by question
- paired_group_bootstrap.csv: paired differences resampling image groups
- image_shuffle_diagnostic.csv: impact of breaking image-question correspondence
- subgroup_accuracy.csv: accuracy by subject, grade, and number of options
- high_confidence_errors_for_review.csv: errors requiring manual interpretation
- experiment_manifest.json, environment_versions.json: reproducibility and audit metadata

Check quick_run in the manifest. QUICK RUN results are pilot subset results, not full cleaned-dataset results.
Large model checkpoints and embedding caches are separate from this ZIP.
