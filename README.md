# Abstractive Text Summarization Using Transformer BART Model

## Overview

Abstractive text summarization is a challenging task in Natural Language Processing (NLP) where the goal is to generate a concise summary of a text while preserving its meaning. Unlike extractive summarization, which involves selecting and rearranging sentences from the source text, abstractive summarization creates new sentences, potentially rephrasing or condensing information.

This approach leverages the power of deep learning through the BART model (Bidirectional and Auto-Regressive Transformer). BART combines the bidirectional encoder of BERT and the autoregressive decoder of GPT. The BART model is fine-tuned to generate high-quality summaries that capture the essence of long articles in a few sentences.

The task of text summarization has numerous applications across fields, including journalism, financial research, legal document analysis, scientific literature, and many more.

## Aim

The goal is to implement an abstractive text summarizer using the BART model from the Hugging Face Transformers library to generate summaries of news articles. The summarizer is trained to produce human-like summaries from a large dataset of news articles.

## Dataset

The dataset used in this implementation includes around 40,000 professionally written summaries of news articles. Each record contains:
- **Article titles**: The title of the article.
- **Summaries**: The professionally written summary of the article.
- **URLs**: The link to the full article.
- **Dates**: Publication date of the article.
- **Article content**: The full content of the article.

The dataset is preprocessed, tokenized, and split into training and testing subsets to build and evaluate the model.

## Key Concepts

### BART Model

BART (Bidirectional and Auto-Regressive Transformer) is a powerful model designed for sequence-to-sequence tasks like text summarization. It merges the advantages of BERT's bidirectional encoder and GPT's autoregressive decoder into a unified framework. The encoder processes the entire input sequence (such as an article), and the decoder generates the output (a summary) one token at a time, autoregressively.

BART is particularly effective for summarization because:
- **Bidirectional encoding** helps the model understand the entire context of an input article.
- **Autoregressive decoding** ensures fluency and coherence in the generated summaries.
- **Attention mechanisms** enable the model to focus on relevant parts of the input article during summary generation.

### Pretraining and Fine-tuning in BART

BART is pretrained on tasks where parts of the input sequence are corrupted, and the model is trained to reconstruct the original sequence. During fine-tuning, the model is adapted to specific tasks, such as text summarization, making use of its powerful encoder-decoder architecture.

The pretraining tasks include:
- **Token masking**: Randomly masking a small number of input tokens.
- **Token deletion**: Deleting certain tokens from the document.
- **Text infilling**: Replacing multiple tokens with a single mask token.
- **Sentence permutation**: Shuffling the order of sentences for training.
- **Document rotation**: Selecting a token randomly and rotating the sequence so that the document starts with the chosen token.

### Abstractive Summarization

Abstractive summarization generates new sentences based on the meaning of the source text. Unlike extractive summarization, where key sentences are extracted from the original document, abstractive summarization generates its own unique phrases. The BART model excels at this task due to its sophisticated architecture that allows it to paraphrase, condense, and even generate novel language structures.

### Evaluation Metric – ROUGE

The **ROUGE** (Recall-Oriented Understudy for Gisting Evaluation) metric is commonly used for evaluating summarization tasks. It compares the overlap between n-grams (a contiguous sequence of n items) in the generated summaries and reference summaries. ROUGE measures recall, precision, and F1-score, providing insights into the quality of the generated summaries.

Key ROUGE metrics include:
- **ROUGE-1**: Measures the overlap of unigrams (single words) between the generated summary and reference summary.
- **ROUGE-2**: Measures the overlap of bigrams (two consecutive words) between the generated summary and reference summary.
- **ROUGE-L**: Measures the longest common subsequence (LCS) between the generated summary and reference summary.
- **ROUGE-Lsum**: Specifically tailored for summarization tasks, this metric evaluates the LCS at the summary level.

ROUGE scores provide a quantifiable method to evaluate the quality of generated summaries. For example, scores of:
- `rouge1: 0.21875`
- `rouge2: 0.129`
- `rougeL: 0.21875`
- `rougeLsum: 0.21875`

These scores indicate a moderate level of overlap between the generated summary and the reference summary. ROUGE is an essential metric in determining how well the model captures the key ideas of the source text.

### Conclusion

Using the BART model for abstractive summarization allows the generation of coherent and concise summaries, leveraging a deep learning architecture that understands and rephrases the content of an article. The evaluation using ROUGE metrics provides a clear measure of the summarizer’s performance, showing how effectively it can replicate human-like summarization.
