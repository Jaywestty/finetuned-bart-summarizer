# 📰 BART News Summarizer

A fine-tuned **BART-base** model for **abstractive news summarization**.

Trained on the **CNN/DailyMail dataset** using Hugging Face **Transformers** and **PyTorch** in Google Colab, the model generates **concise and coherent summaries** of lengthy news articles — ideal for quick news consumption
# News Text Summarizer: Final Evaluation Report

Model: `facebook/bart-base` fine-tuned on CNN/DailyMail (version 3.0.0)
Report date: 2026-10-06
Source of numbers: the Colab run outputs and `results.json` from the final run. Nothing in this report is estimated.

## 1. Summary

The final model was trained on 16,000 CNN/DailyMail articles for 5 epochs (83 minutes on a free T4 GPU). It was evaluated on 800 validation and 800 test articles, against two references: a Lead-3 baseline and the untrained BART-base model decoded the same way.

| Model | Split | ROUGE-1 | ROUGE-2 | ROUGE-L | ROUGE-Lsum |
|---|---|---|---|---|---|
| Untrained BART-base | val | 30.59 | 10.72 | 20.33 | 26.89 |
| Lead-3 (boilerplate stripped) | val | 29.33 | 10.97 | 19.49 | 26.10 |
| **Fine-tuned** | val | **33.14** | **13.27** | **23.69** | **30.42** |
| Untrained BART-base | test | 31.42 | 11.85 | 21.31 | 27.78 |
| Lead-3 (boilerplate stripped) | test | 30.15 | 11.79 | 20.46 | 26.87 |
| **Fine-tuned** | test | **32.82** | **12.96** | **23.33** | **30.11** |

Key findings:

1. On the held-out test set the fine-tuned model beats both Lead-3 and the untrained model on all four ROUGE metrics, and every 95 percent confidence interval for the gains excludes zero.
2. Validation and test scores agree to within about 0.3 points, so checkpoint and decoding selection did not overfit the validation set.
3. The gain over the untrained model is real but modest (+1.4 ROUGE-1, +2.3 ROUGE-Lsum). The untrained model already copies the start of the article, which scores well on this metric.
4. The model was still improving at epoch 5, so it is undertrained rather than overfit.
5. The decoding fix (removing the forced 56-token minimum) mattered, but the final tuned decoding is not better than plain beam search on the full validation set (section 4.3).

## 2. Setup

| Item | Value |
|---|---|
| Base model | `facebook/bart-base` |
| Dataset | `abisee/cnn_dailymail`, config 3.0.0 |
| Training set | 16,000 articles, shuffled with seed 42 (about 5.6 percent of the full training split) |
| Validation / test | First 800 articles of each split |
| Input / target length | 1024 / 142 tokens, dynamic padding |
| Epochs and steps | 5 epochs, 1,000 steps per epoch, 5,000 steps |
| Batch size | 4 per device, 4 accumulation steps, effective 16 |
| Learning rate | 5e-5, 250 warmup steps (5 percent), weight decay 0.01, label smoothing 0.1 |
| Precision | fp16 |
| Checkpoint selection | `load_best_model_at_end=True`, metric `rougeLsum` on validation |
| Seed | 42 |
| Runtime | 4,986 seconds (83 minutes), about 16 samples per second |
| Final decoding | beam search 4, `max_length=64`, `min_length=0`, `length_penalty=2.0`, `no_repeat_ngram_size=3`, early stopping |

Evaluation protocol: the test set was scored once, after the checkpoint and decoding settings were chosen on validation data. Lead-3 variant and decoding were both selected on validation, never on test.

## 3. How to read the metrics

All ROUGE values are F1 scores multiplied by 100. F1 balances precision (how much of the generated text appears in the reference) with recall (how much of the reference appears in the generated text), so summaries that are too long or too short are both penalized.

| Metric | Meaning |
|---|---|
| ROUGE-1 | Overlap of single words |
| ROUGE-2 | Overlap of two-word sequences. Lower in value, and a better signal for fluent phrasing |
| ROUGE-L | Longest common subsequence over the whole text treated as one sequence |
| ROUGE-Lsum | Longest common subsequence computed sentence by sentence and combined. Requires newline sentence splitting, which this notebook does with NLTK. Used to pick the best checkpoint |
| Lead-3 | A baseline, not a metric: the first three sentences of the article used as the summary |
| Validation loss | Average cross-entropy per token. It includes the label smoothing term here, so it is not comparable with losses from runs without smoothing |
| gen_len | Average generated length in tokens during in-training evaluation |

## 4. Results in detail

### 4.1 Training curve

In-training evaluation on the 800 validation articles (beam 2, `max_length=142`):

| Epoch | Train loss | Val loss | ROUGE-1 | ROUGE-2 | ROUGE-L | ROUGE-Lsum | gen_len |
|---|---|---|---|---|---|---|---|
| 1 | 3.441 | 3.331 | 31.77 | 12.19 | 22.43 | 29.09 | 59.5 |
| 2 | 3.197 | 3.304 | 31.28 | 11.87 | 21.82 | 28.67 | 68.9 |
| 3 | 3.106 | 3.276 | 31.88 | 12.34 | 22.21 | 29.10 | 70.2 |
| 4 | 2.988 | 3.266 | 32.61 | 12.69 | 23.14 | 29.94 | 63.6 |
| 5 | 2.927 | 3.270 | 33.05 | 13.24 | 23.52 | 30.47 | 67.1 |

- The selected checkpoint is step 5,000 (epoch 5), best validation ROUGE-Lsum 30.47.
- ROUGE-Lsum rose from 29.09 to 30.47 over training. Epoch 2 dipped below epoch 1. Single-epoch differences smaller than about 0.5 are within evaluation noise (see 4.5), so the upward trend is reliable but any one step-to-step change is not.
- Validation loss flattened at epochs 4 and 5 (3.266 and 3.270) while ROUGE still rose. There is no sign of overfitting. The training loss is lower than the validation loss, which is expected because it is averaged during training with dropout active.
- After the best checkpoint was reloaded, the log shows a warning about missing embedding keys. These are tied weights that are re-tied on load. The validation scores after reload (33.14 / 30.42) match the epoch 5 in-training scores (33.05 / 30.47), so the weights were restored correctly.

### 4.2 Lead-3 baseline selection

Three Lead-3 variants were scored on validation. The one with the highest ROUGE-Lsum was used for all comparisons.

| Variant | ROUGE-1 | ROUGE-2 | ROUGE-L | ROUGE-Lsum | Avg words |
|---|---|---|---|---|---|
| Boilerplate stripped (selected) | 29.33 | 10.97 | 19.49 | 26.10 | 68.9 |
| Original | 28.95 | 10.76 | 19.23 | 25.72 | 67.6 |
| Regex sentence split | 28.83 | 10.73 | 19.29 | 25.64 | 67.4 |

Stripping `(CNN)` prefixes, datelines and Daily Mail bylines helped by about 0.4 points. That is small, so it does not explain why Lead-3 scores near 30 ROUGE-1 here when published Lead-3 results on the full CNN/DailyMail test set are commonly reported near 40. The reference summaries average 34 words while Lead-3 outputs average 67 to 69 words, which lowers precision and so F1. That length gap is a plausible contributor but was not tested (see section 7).

### 4.3 Decoding sweep (the minimum-length fix)

The first full run forced a minimum of 56 tokens and a length penalty of 2.0. The sweep scored the selected checkpoint on the first 300 validation articles under 9 alternative settings with no forced minimum, plus the old settings.

| Config | ROUGE-1 | ROUGE-2 | ROUGE-L | ROUGE-Lsum | Avg words |
|---|---|---|---|---|---|
| max64, penalty 2.0 (selected) | 34.25 | 14.41 | 24.93 | 31.58 | 46.7 |
| max64, penalty 0.8 | 34.16 | 14.48 | 25.00 | 31.57 | 44.8 |
| max64, penalty 1.2 | 34.10 | 14.30 | 24.87 | 31.53 | 45.9 |
| max80, penalty 0.8 | 33.92 | 14.41 | 24.58 | 31.16 | 48.3 |
| max80, penalty 1.2 | 33.78 | 14.23 | 24.37 | 31.15 | 50.8 |
| max100, penalty 0.8 | 33.75 | 14.38 | 24.46 | 31.10 | 49.0 |
| max100, penalty 1.2 | 33.65 | 14.25 | 24.22 | 31.08 | 52.6 |
| max80, penalty 2.0 | 33.76 | 14.19 | 24.40 | 31.02 | 52.7 |
| max100, penalty 2.0 | 33.60 | 14.20 | 24.20 | 30.92 | 55.0 |
| Old settings (min 56, penalty 2.0, max 142) | 33.12 | 14.05 | 23.60 | 30.51 | 59.4 |

What this shows:

- Removing the forced minimum and shortening the cap raised ROUGE-Lsum by 1.07 on this 300-article sample. Shorter outputs match the references (about 34 words) better. Scores fall steadily as `max_length` grows from 64 to 80 to 100.
- The length penalty has almost no effect. The top three configurations are within 0.05 of each other, which is noise. `max64, penalty 2.0` was selected only because it ranked first.
- Selecting the best of 10 settings on 300 articles is mildly optimistic, and the first 300 articles score higher than the full 800. Do not compare the 31.58 here with the 30.42 validation score below.
- Important caveat: on the full 800-article validation set, the tuned decoding scored ROUGE-Lsum 30.42, while the plain beam-2 decoding used during training evaluation scored 30.47. The two are equivalent. The tuning therefore repaired the damage done by the forced minimum, and did not find anything better than simple defaults.

### 4.4 Final results

See the table in section 1. Differences between fine-tuned and each reference, on the test set:

| Comparison | ROUGE-1 | ROUGE-2 | ROUGE-L | ROUGE-Lsum |
|---|---|---|---|---|
| Fine-tuned minus Lead-3 | +2.67 | +1.17 | +2.87 | +3.24 |
| Fine-tuned minus untrained | +1.40 | +1.11 | +2.02 | +2.33 |

The untrained BART-base, decoded with the same settings, already beats Lead-3 on test (31.42 vs 30.15 ROUGE-1). It is a denoising model that largely reproduces the start of the article, and the 64-token cap does the rest of the work.

### 4.5 Confidence intervals (paired bootstrap over test articles)

2,000 resamples of the 800 test articles. The gain columns are means of per-example differences, so they can differ by up to about 0.03 from subtracting the headline scores, which come from the evaluation library's own aggregation.

| Comparison | Metric | Gain | 95 percent CI | Share of resamples not better |
|---|---|---|---|---|
| vs Lead-3 | ROUGE-1 | +2.66 | 1.86 to 3.48 | 0.0000 |
| vs Lead-3 | ROUGE-2 | +1.18 | 0.48 to 1.88 | 0.0000 |
| vs Lead-3 | ROUGE-L | +2.85 | 2.16 to 3.56 | 0.0000 |
| vs Lead-3 | ROUGE-Lsum | +3.24 | 2.48 to 4.03 | 0.0000 |
| vs untrained | ROUGE-1 | +1.37 | 0.54 to 2.22 | 0.0005 |
| vs untrained | ROUGE-2 | +1.12 | 0.37 to 1.87 | 0.0005 |
| vs untrained | ROUGE-L | +2.00 | 1.28 to 2.75 | 0.0000 |
| vs untrained | ROUGE-Lsum | +2.33 | 1.51 to 3.15 | 0.0000 |

Absolute test scores with 95 percent intervals:

| Metric | Mean | 95 percent CI |
|---|---|---|
| ROUGE-1 | 32.81 | 31.98 to 33.67 |
| ROUGE-2 | 12.97 | 12.20 to 13.77 |
| ROUGE-L | 23.33 | 22.58 to 24.09 |
| ROUGE-Lsum | 30.12 | 29.34 to 30.93 |

Reading these: each interval is roughly plus or minus 0.8 points. A share of 0.0000 means no resample out of 2,000 failed to favor the model, not that the true probability is zero. These intervals reflect which 800 articles were sampled. They do not include training randomness, because only one seed was run.

### 4.6 Length analysis (test set, average words)

| Source | Words |
|---|---|
| Reference | 34.3 |
| Fine-tuned | 46.7 |
| Untrained BART-base | 46.9 |
| Lead-3 | 66.8 |

The fine-tuned and untrained models produce almost identical lengths, which indicates the 64-token cap, rather than learned behavior, is setting the length. Both are about 36 percent longer than the references. Lead-3 is almost double the reference length.

### 4.7 Progression across runs

| Run | Setup | Split | ROUGE-1 | ROUGE-2 | ROUGE-L | ROUGE-Lsum |
|---|---|---|---|---|---|---|
| Original notebook | 8k examples, 3 epochs, evaluation cut near 20 tokens, unsplit Lsum | val | 24.6 | 9.5 | not reported | 22.48 |
| Run 1 | 8k examples, 3 epochs, forced minimum length 56 | test | 31.22 | 12.14 | 21.73 | 28.56 |
| Run 2 (final) | 16k examples, 5 epochs, tuned decoding | test | 32.82 | 12.96 | 23.33 | 30.11 |

Run 2 improved over Run 1 by +1.60 ROUGE-1, +0.82 ROUGE-2, +1.60 ROUGE-L and +1.55 ROUGE-Lsum on test. Data size, epochs and decoding all changed together, so the gain cannot be assigned to any single change. The original notebook's numbers use a different Lsum computation and truncated generation, so they are not a like-for-like comparison.

## 5. Claims that are safe to make

1. "Fine-tuned BART-base outperforms Lead-3 and the untrained model on all four ROUGE metrics on a held-out test set of 800 articles, with 95 percent confidence intervals that exclude zero."
2. "Validation and test scores are within about 0.3 points, so there is no sign of overfitting to the validation set."
3. "The model is undertrained: ROUGE was still rising at epoch 5 while validation loss had flattened."
4. "Forcing a 56-token minimum hurt. Removing it and capping length at 64 improved ROUGE-Lsum by about 1 point on a 300-article validation sample."
5. "Most of the absolute score comes from the model reproducing the lead of the article. Fine-tuning adds about 1.4 ROUGE-1 and 2.3 ROUGE-Lsum on top of that."

## 6. Claims to avoid

- Comparing these scores to published results. Published BART numbers use the full training set, and the Lead-3 baseline here is about 10 ROUGE-1 points below commonly reported Lead-3 numbers for reasons not yet explained.
- Saying the model "significantly beats Lead-3" without noting that Lead-3 outputs are about twice as long as the references and as the model's output.
- Claiming the tuned decoding is better than default beam search. On the full validation set it is not.
- Attributing the improvement over the original notebook to any one fix.
- Saying anything about factual accuracy. ROUGE measures word overlap only, and no factuality check was run on the final model. In the first full run, one sample summary misattributed a statement to the wrong person, which ROUGE cannot detect. The qualitative samples from the final run were not reviewed for this report.

## 7. Limitations

- One training run, one seed. Run-to-run variation is not measured.
- Training used 16,000 of roughly 287,000 available articles.
- Validation and test are the first 800 articles of each split, not random samples.
- The decoding sweep used only 300 articles and 10 configurations, and the 300 are part of the 800 validation articles.
- Lead-3 is not length-matched to the model. A Lead-1, Lead-2 or word-truncated lead baseline could score higher F1 than Lead-3 and would be a stricter comparison. This was not run.
- The untrained BART-base was decoded with settings tuned on the fine-tuned model, which may favor it slightly.
- The best checkpoint was the last epoch, so the trend suggests further gains are available but they were not measured.
- Confidence intervals cover test-article sampling only.

## 8. Issues fixed during the project

| Issue | Fix |
|---|---|
| Evaluation generated about 20 tokens | Set `generation_max_length=142` and beam search |
| `rougeLsum` equal to `rougeL` | Split sentences with NLTK before scoring |
| Test split never used | Used once at the end |
| Manual checkpoint choice | `load_best_model_at_end` on `rougeLsum` |
| No baseline | Lead-3 variants and untrained model added |
| Padding to 1024 tokens, deprecated tokenizer call, tiny batch | Dynamic padding, `text_target`, effective batch 16, warmup |
| Forced minimum length of 56 | Decoding sweep with no forced minimum |
| `model.config.max_length` error in newer transformers | Use `model.generation_config.max_length` |
| `cnn_dailymail` dataset id rejected | Use `abisee/cnn_dailymail` |
| `warmup_ratio` rejected | Use `warmup_steps`, computed from dataset size and epochs |

## 9. Recommended next steps

1. Add length-matched Lead baselines (Lead-1, Lead-2, first 47 words) so the baseline comparison cannot be challenged on length.
2. Train on more data (the full 287,000 articles if time allows) and for more epochs, since ROUGE is still rising.
3. Repeat the run with two or three seeds and report the mean and spread.
4. Add a factuality check (for example, a manual review of 30 summaries or an entailment-based metric).
5. Compare against `facebook/bart-large-cnn` as an upper reference.
6. Publish the final model to the Hugging Face Hub after changing `HF_REPO_ID`, so the earlier epoch 2 model is not overwritten.

## 10. Reproducibility

All settings are in the configuration cell of `Text_summarizer_improved.ipynb`: seed 42, `TRAIN_SIZE=16000`, `NUM_EPOCHS=5`, effective batch 16, bootstrap 2,000 resamples. The run outputs are stored in `results.json`, which also holds the decoding sweep and the Lead-3 variant tables.
