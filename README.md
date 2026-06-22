# Twitter Sentiment Classification: Classical ML vs LSTM vs Quantized BERT

End-to-end sentiment classification on the SemEval 2017 Twitter/X dataset, comparing classical machine learning, word embeddings, recurrent neural networks, transformer fine-tuning and few-shot prompting.

The project evaluates multiple approaches for classifying tweets as positive, negative or neutral, with the best full-test model being a fine-tuned and dynamically quantized BERT classifier achieving 0.713 mean macro-F1 across three SemEval test sets.

## Highlights

- Built and compared TF-IDF SVM, MaxEnt, GloVe-feature models, LSTM, quantized BERT and few-shot FLAN-T5 prompting.
- Fine-tuned BERT and applied dynamic quantization for faster local inference.
- Evaluated models across three independent SemEval test sets using macro-F1.
- Performed error analysis using confusion matrices and validation-loss curves.
