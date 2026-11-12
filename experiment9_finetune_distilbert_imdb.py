"""EXPERIMENT 9: Fine-tuning a Transformer Model.

AIM:
    To fine-tune a pre-trained DistilBERT transformer model on movie
    reviews and evaluate its classification performance.

ALGORITHM:
    1. Load the IMDb movie-review dataset.
    2. Select training and testing subsets.
    3. Load the DistilBERT tokenizer and tokenize the reviews.
    4. Load DistilBERT with two classification labels.
    5. Train the model using the Trainer API and evaluate its performance.

Sample Output:
    Training started...
    Training completed.
    Evaluation:
    eval_loss: (calculated value)
    eval_accuracy: (calculated value)

RESULT:
    Thus, the DistilBERT transformer model was fine-tuned on the IMDb
    dataset and evaluated using the Trainer API.
"""

from datasets import load_dataset
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          Trainer, TrainingArguments)


def main() -> None:
    ds = load_dataset("stanfordnlp/imdb")
    train = ds["train"].shuffle(seed=42).select(range(2000))
    test = ds["test"].shuffle(seed=42).select(range(500))
    name = "distilbert/distilbert-base-uncased"
    tok = AutoTokenizer.from_pretrained(name)

    def encode(batch):
        return tok(batch["text"], truncation=True, max_length=256)

    train = train.map(encode, batched=True).remove_columns(["text"])
    test = test.map(encode, batched=True).remove_columns(["text"])
    model = AutoModelForSequenceClassification.from_pretrained(name, num_labels=2)

    args = TrainingArguments(output_dir="results", num_train_epochs=1,
                             per_device_train_batch_size=8,
                             per_device_eval_batch_size=8,
                             eval_strategy="epoch", report_to="none")
    trainer = Trainer(model=model, args=args,
                      train_dataset=train, eval_dataset=test)
    print("Training started...")
    trainer.train()
    print("Training completed.")
    print("Evaluation:")
    print(trainer.evaluate())


if __name__ == "__main__":
    main()
