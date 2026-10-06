# News Text Summarizer

An AI model that reads a news article and writes a short summary of it.

The model is `facebook/bart-base`, a general-purpose language model from Meta, which I fine-tuned on about 16,000 CNN and Daily Mail news articles so that it learns to write news-style highlights. The whole project runs in a free Google Colab notebook.

## At a glance

| | |
|---|---|
| **What it does** | Turns a long news article into a short summary (about 45 words) |
| **Base model** | `facebook/bart-base` (about 140 million parameters) |
| **Training data** | 16,000 articles from the CNN/DailyMail dataset |
| **Training time** | 83 minutes on a free Colab T4 GPU |
| **Main result** | Beats two simple reference methods on every score, and the gains are statistically reliable |
| **Honest caveat** | The improvement over the best simple reference is modest (see "What the results mean") |

## The problem

Reading the news takes time. A summarizer gives you the main point of an article in a few seconds. The hard part is not only building one, but measuring whether it is actually good. This project does both: it trains a model and checks it carefully against simple baselines, with confidence ranges, so the numbers can be trusted.

## How it works

1. **Start with a pre-trained model.** BART already understands English. It was not taught to summarize.
2. **Show it examples.** Each example is a news article paired with the human-written highlights that came with it. The model practices writing those highlights.
3. **Repeat 5 times.** One full pass through the 16,000 examples is called an epoch. After each epoch, the model is tested on 800 articles it has not trained on.
4. **Keep the best version.** The version that scored best on the test articles is kept automatically.
5. **Choose how it writes.** The way the model builds a summary word by word can be adjusted, for example how long it is allowed to go. These settings were tuned on separate validation data.
6. **Final exam.** The finished model is scored once on 800 other articles it has never seen (the test set).

## How the model is scored

The score is called **ROUGE**. It compares the model's summary with the human-written summary and measures how many words and phrases they share. A higher score means more overlap. Four versions are reported:

| Score | In plain words |
|---|---|
| **ROUGE-1** | How many single words match the human summary |
| **ROUGE-2** | How many two-word phrases match. Harder, so numbers are lower |
| **ROUGE-L** | How well the words match in the same order |
| **ROUGE-Lsum** | The same idea, checked sentence by sentence. This is the score used to pick the best model |

All scores are out of 100. They are not percentages of correctness, and a perfect 100 is impossible because two people would never write the same summary. ROUGE only measures overlap with the human summary. It cannot tell whether a summary is factually right.

## Who the model is compared with

A score on its own means little, so the model is compared with two simple references:

- **Lead-3.** Copy the first three sentences of the article. This is a classic news baseline, because news articles put the key facts first.
- **Untrained BART-base.** The same model before any training, writing summaries with the same settings. This shows how much the training actually added.

## Results

Scored on 800 test articles the model had never seen:

| Method | ROUGE-1 | ROUGE-2 | ROUGE-L | ROUGE-Lsum |
|---|---|---|---|---|
| Lead-3 | 30.15 | 11.79 | 20.46 | 26.87 |
| Untrained BART-base | 31.42 | 11.85 | 21.31 | 27.78 |
| **This model** | **32.82** | **12.96** | **23.33** | **30.11** |

The same comparison on the 800 validation articles gave 33.14, 13.27, 23.69 and 30.42, which is within about 0.3 points of the test scores. That agreement is a good sign: it means the model was not just tuned to look good on one set of articles.

### How sure are we?

Scores depend partly on which articles happen to be in the test set. To measure that, the test articles were resampled 2,000 times. The result is a range in which the true gain very likely lies. If the range does not include zero, the gain is reliable.

| Model versus | ROUGE-1 gain | ROUGE-2 gain | ROUGE-L gain | ROUGE-Lsum gain |
|---|---|---|---|---|
| Lead-3 | +2.66 (1.86 to 3.48) | +1.18 (0.48 to 1.88) | +2.85 (2.16 to 3.56) | +3.24 (2.48 to 4.03) |
| Untrained BART-base | +1.37 (0.54 to 2.22) | +1.12 (0.37 to 1.87) | +2.00 (1.28 to 2.75) | +2.33 (1.51 to 3.15) |

The numbers in brackets are the 95 percent range. None of them include zero, so the model reliably beats both references on every score.

### Training progress

The model improved steadily over the five epochs, and it was still improving at the end.

| Epoch | ROUGE-1 | ROUGE-2 | ROUGE-L | ROUGE-Lsum |
|---|---|---|---|---|
| 1 | 31.77 | 12.19 | 22.43 | 29.09 |
| 2 | 31.28 | 11.87 | 21.82 | 28.67 |
| 3 | 31.88 | 12.34 | 22.21 | 29.10 |
| 4 | 32.61 | 12.69 | 23.14 | 29.94 |
| 5 | 33.05 | 13.24 | 23.52 | 30.47 |

(Validation articles, quick evaluation settings used during training.) The model did not start to memorize the training data. It is undertrained, so more data or more epochs would likely help.

## What the results mean

**What is solid:**

- The trained model reliably beats both Lead-3 and the untrained model on every ROUGE score.
- Results on the validation and test sets agree, so the numbers are not a fluke of one set.
- Training clearly helps: the model gains about 1.4 points on ROUGE-1 and 2.3 points on ROUGE-Lsum over the untrained model.

**What to keep in mind:**

- **The untrained model is already decent.** Untrained BART tends to copy the beginning of the article, and that scores reasonably well on ROUGE. Training adds a real but modest improvement on top.
- **Lead-3 is not a perfectly fair opponent.** It writes about 67 words, the model writes about 47, and the human summaries average 34. Length affects ROUGE, so a shorter lead baseline (just the first sentence or two) might score higher than Lead-3. This has not been tested yet.
- **Scores here are lower than published results.** Published results for similar models use the full training set (about 287,000 articles) and larger models. This project uses about 5.6 percent of the data. The Lead-3 score here is also lower than commonly reported figures, and the cause has not been found, so these numbers should not be compared directly with other papers.
- **ROUGE does not check facts.** A summary can score well and still get a detail wrong. In an earlier version of this project, one sample summary attributed a statement to the wrong person. No fact-checking was done on the final model.

## Things learned along the way

- **Forcing a minimum length hurt.** The first version made the model write at least 56 tokens, but human summaries are short (34 words on average). Removing that rule improved the score by about 1 point in a test on 300 articles. The final settings cap the summary length at 64 tokens.
- **Tuning the writing settings did not beat the defaults.** On the full validation set, the tuned settings scored the same as simple settings (30.42 versus 30.47 on ROUGE-Lsum). The tuning mainly repaired the minimum-length mistake.
- **Fixing the measurement mattered as much as fixing the model.** The first version cut summaries off at about 20 words during testing, which made the scores look much worse than they were, and it used a sentence-level score without splitting sentences. Fixing the testing process was a large part of the improvement.

## Limitations

- One training run with one random seed, so run-to-run variation is not measured.
- Trained on 16,000 of about 287,000 available articles.
- Validation and test sets are the first 800 articles of each split, not random samples.
- The writing settings were tuned on only 300 validation articles.
- Works on English news articles. It has not been tested on other kinds of text or on other languages.
- Summaries are not checked for factual accuracy.

## Run it yourself

1. Open `Text_summarizer_improved.ipynb` in [Google Colab](https://colab.research.google.com/).
2. Set the runtime to GPU: Runtime, then Change runtime type, then T4 GPU.
3. Run all cells from top to bottom. The full run takes roughly 1.5 to 2 hours.

All settings are in one configuration cell near the top:

| Setting | Value used | What it controls |
|---|---|---|
| `TRAIN_SIZE` | 16000 | Number of training articles |
| `NUM_EPOCHS` | 5 | Number of passes through the training data |
| `SEED` | 42 | Makes the run repeatable |
| `PUSH_TO_HUB` | False | Set to True to upload the model to Hugging Face |

If Colab disconnects or runs short on time, lower `TRAIN_SIZE` (for example to 10000).

At the end the notebook writes `results.json`, which holds every score in this README.

## Use the trained model

If you have uploaded the model to the Hugging Face Hub, replace `YOUR_USERNAME/bart-summarizer-best` with your model name:

```python
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

model_id = "YOUR_USERNAME/bart-summarizer-best"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForSeq2SeqLM.from_pretrained(model_id)

article = "Paste a news article here."

inputs = tokenizer(article, return_tensors="pt", max_length=1024, truncation=True)
summary_ids = model.generate(
    **inputs,
    num_beams=4,
    max_length=64,
    min_length=0,
    length_penalty=2.0,
    no_repeat_ngram_size=3,
    early_stopping=True,
)
print(tokenizer.decode(summary_ids[0], skip_special_tokens=True))
```

## Repository contents

| File | Purpose |
|---|---|
| `Text_summarizer_improved.ipynb` | The full project: training, tuning, evaluation and saving |
| `results.json` | All scores from the final run |
| `README.md` | This document |

## Possible next steps

1. Add shorter Lead baselines (first sentence, first two sentences) for a fairer comparison.
2. Train on more articles and for more epochs, since the model was still improving.
3. Repeat the run with a few different seeds and report the average.
4. Add a fact-checking step, for example a manual review of 30 summaries.
5. Compare against `facebook/bart-large-cnn`, a larger model already trained for this task.

## Credits

- Base model: BART by Meta AI (Lewis et al., 2019), `facebook/bart-base`.
- Data: the CNN/DailyMail dataset, loaded from Hugging Face as `abisee/cnn_dailymail`.
- Built with Hugging Face Transformers, Datasets and Evaluate.

## License

Add the license of your choice here. Check the licenses of the base model and the dataset before reusing the trained model commercially.
