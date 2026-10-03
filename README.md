# Resume-Classifier-NLP
Resume classifier using *longformer-base*. Classifies into 24 distinct Job-type based on the huge form text resume.

url: https://huggingface.co/Kami0867/logformer-resume-classifier/commit/a361ba729fff3cd8c4a916a95c608ef2446553b5

# Loading model:

**model_id = "Kami0867/logformer-resume-classifier"**

**tokenizer = AutoTokenizer.from_pretrained(model_id)**

**model = AutoModelForSequenceClassification.from_pretrained(model_id)**

# Dataset:

dataset : https://www.kaggle.com/datasets/snehaanbhawal/resume-dataset

# Achieved accuracy:

train-set: 95.57%

test-set: 90.14%

Tokenizing resumes: 100%|███████████████████████████████████████████████████████████████████████████| 1986/1986 [00:02<00:00, 825.91 examples/s]
[transformers] Initializing global attention on CLS token...
[transformers] Input ids are automatically padded to be a multiple of `config.attention_window`: 512
Train resume-level accuracy: 95.57%
Tokenizing resumes: 100%|█████████████████████████████████████████████████████████████████████████████| 497/497 [00:00<00:00, 803.98 examples/s]
Held-out validation/test resume-level accuracy: 90.14%
Note: the held-out split was used during checkpoint tuning.