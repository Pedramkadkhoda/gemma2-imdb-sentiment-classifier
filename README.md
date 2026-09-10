# Gemma-2b IMDB Sentiment Classifier

Fine-tuning `google/gemma-2b-it` for sentiment classification (positive/negative) on the IMDB dataset, using LoRA and 4-bit quantization (QLoRA) to run on GPUs with limited memory (e.g., a T4 on Google Colab).

## Features
- Base model: Gemma-2b-it (Google)
- Fine-tuning method: LoRA (Low-Rank Adaptation)
- Quantization: 4-bit (NF4) via bitsandbytes
- Dataset: [stanfordnlp/imdb](https://huggingface.co/datasets/stanfordnlp/imdb)
- Evaluation: Accuracy, F1, Precision, Recall

## Setup
Requirements:
```bash
pip install transformers datasets evaluate rouge_score loralib bitsandbytes scikit-learn peft
```

To run:
1. Get a Hugging Face Access Token and save it as `HF_TOKEN` in Colab Secrets
2. Open `Untitled11.ipynb` in Google Colab (requires a GPU, e.g., T4)
3. Run the cells in order

## Result
The final model is published on the Hugging Face Hub:
[account/gemma2b-imdb-sentiment-lora](https://huggingface.co/account/gemma2b-imdb-sentiment-lora)

## Notes
This project was built for learning and practice purposes.
