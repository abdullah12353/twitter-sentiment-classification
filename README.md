# Twitter Sentiment Classification: Classical ML vs LSTM vs Quantized BERT

End-to-end sentiment classification on the SemEval 2017 Twitter/X dataset, comparing classical machine learning, word embeddings, recurrent neural networks, transformer fine-tuning and few-shot prompting.

The project evaluates multiple approaches for classifying tweets as positive, negative or neutral, with the best full-test model being a fine-tuned and dynamically quantized BERT classifier achieving 0.713 mean macro-F1 across three SemEval test sets.

## Highlights

- Built and compared TF-IDF SVM, MaxEnt, GloVe-feature models, LSTM, quantized BERT and few-shot FLAN-T5 prompting.
- Fine-tuned BERT and applied dynamic quantization for faster local inference.
- Evaluated models across three independent SemEval test sets using macro-F1.
- Performed error analysis using confusion matrices and validation-loss curves.

## Results

| Classifier | Test 1 | Test 2 | Test 3 | Mean | Std |
|---|---:|---:|---:|---:|---:|
| BoW + SVM | 0.601 | 0.608 | 0.567 | 0.592 | 0.018 |
| Avg-GloVe + SVM | 0.418 | 0.414 | 0.449 | 0.427 | 0.016 |
| BoW + MaxEnt | 0.538 | 0.538 | 0.519 | 0.532 | 0.009 |
| Avg-GloVe + MaxEnt | 0.562 | 0.575 | 0.533 | 0.557 | 0.018 |
| LSTM | 0.586 | 0.592 | 0.546 | 0.575 | 0.020 |
| Quantized BERT | 0.721 | 0.714 | 0.704 | 0.713 | 0.007 |
| Few-shot FLAN-T5* | 0.750 | 0.727 | 0.656 | 0.711 | 0.040 |

\*Few-shot prompting was evaluated on stratified subsets for faster experimentation, so it should not be treated as directly equivalent to the full-test BERT result.
