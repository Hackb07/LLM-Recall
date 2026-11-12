# EXPERIMENT 3: Sentiment Classification using BERT

## AIM

To classify text reviews into positive, negative or neutral sentiments using a pre-trained BERT sentiment model.

## ALGORITHM

1. Install Transformers and load the pre-trained BERT sentiment pipeline.
2. Prepare review sentences as input.
3. Pass each review to the model for classification.
4. Obtain the predicted star rating and confidence score.
5. Map the star rating to positive, negative or neutral sentiment and display the result.

## CODE

```python
from transformers import pipeline


def main() -> None:
    model = pipeline(
        "sentiment-analysis",
        model="nlptown/bert-base-multilingual-uncased-sentiment",
    )
    reviews = [
        "The movie was fantastic!",
        "The product is terrible.",
        "The food was okay.",
    ]

    for review in reviews:
        r = model(review)[0]
        stars = int(r["label"].split()[0])
        sentiment = ("NEGATIVE" if stars <= 2 else
                     "NEUTRAL" if stars == 3 else "POSITIVE")
        print(review, "->", sentiment, round(r["score"], 3))


if __name__ == "__main__":
    main()
```

## OUTPUT

```
The movie was fantastic! -> POSITIVE (confidence score)
The product is terrible. -> NEGATIVE (confidence score)
The food was okay. -> NEUTRAL (confidence score)
```

Note: Exact confidence scores are produced by the model.

## RESULT

Thus, sentiment classification using the pre-trained BERT model was implemented successfully.
