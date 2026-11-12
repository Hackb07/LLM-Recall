# EXPERIMENT 8: Fine-Tune BERT for Custom Dataset

## AIM

To fine-tune a pre-trained BERT model on a labelled text dataset for binary sentiment classification.

## ALGORITHM

1. Prepare a labelled text dataset.
2. Tokenize the text and create attention masks.
3. Load BERT with a two-class classification layer.
4. Train the model using a suitable optimizer and loss function.
5. Evaluate the model and predict the class of a new sentence.

## CODE

```python
import torch
from transformers import BertForSequenceClassification, BertTokenizer


def main() -> None:
    texts = ["excellent product", "great quality",
             "terrible product", "poor quality"]
    labels = torch.tensor([1, 1, 0, 0])
    tok = BertTokenizer.from_pretrained("bert-base-uncased")
    x = tok(texts, padding=True, truncation=True, return_tensors="pt")
    model = BertForSequenceClassification.from_pretrained(
        "bert-base-uncased", num_labels=2)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-5)

    model.train()
    for _ in range(2):
        opt.zero_grad()
        loss = model(**x, labels=labels).loss
        loss.backward()
        opt.step()

    model.eval()
    with torch.no_grad():
        test = tok("fantastic quality", return_tensors="pt")
        pred = model(**test).logits.argmax(-1).item()
    print("Prediction:", "Positive" if pred == 1 else "Negative")


if __name__ == "__main__":
    main()
```

## OUTPUT

```
Prediction: Positive or Negative
```

(The predicted class depends on the trained model weights.)

## RESULT

Thus, BERT was configured and trained on a small labelled dataset for binary text classification.
