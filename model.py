from transformers import RobertaTokenizer, RobertaForSequenceClassification, TrainingArguments, Trainer
from datasets import load_dataset
import numpy as np
from sklearn.metrics import accuracy_score, f1_score

# Загрузка датасета 
dataset = load_dataset("snli")
print("Пример данных:", dataset["train"][0])

# Загрузка токенизатора и модели roberta
model_name = "roberta-base"
tokenizer = RobertaTokenizer.from_pretrained(model_name)
model = RobertaForSequenceClassification.from_pretrained(model_name, num_labels=3)  # 3 класса

# Токенизация
def tokenize_function(examples):
    return tokenizer(
        examples["premise"],
        examples["hypothesis"],
        padding="max_length",
        truncation=True,
        max_length=128,
    )

tokenized_datasets = dataset.map(tokenize_function, batched=True)

# Вычисление метрик
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, predictions),
        "f1": f1_score(labels, predictions, average="macro"),
    }

# Обучение (на уменьшенном датасете)
training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=16,
    num_train_epochs=3,
    eval_strategy="epoch",
    logging_steps=100,
    learning_rate=2e-5,
    warmup_steps=500,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"].shuffle(seed=42).select(range(10000)),
    eval_dataset=tokenized_datasets["validation"].shuffle(seed=42).select(range(1000)),
    compute_metrics=compute_metrics,
)

trainer.train()

model.save_pretrained("./roberta_snli_model")
tokenizer.save_pretrained("./roberta_snli_model")

# Проверка на своих примерах
def predict(premise, hypothesis):
    inputs = tokenizer(premise, hypothesis, return_tensors="pt", truncation=True, padding=True)
    outputs = model(**inputs)
    predicted_class = outputs.logits.argmax().item()
    labels = ["entailment", "neutral", "contradiction"]
    return labels[predicted_class]

print("\nРезультаты:")
print(predict("A cat sits on the mat.", "The mat is under the cat."))  # entailment
print(predict("It's raining outside.", "The sun is shining."))  # contradiction
print(predict("The book is on the table.", "The table is made of wood."))  # neutral