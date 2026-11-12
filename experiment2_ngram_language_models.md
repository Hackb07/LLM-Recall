# EXPERIMENT 2: Build N-gram Language Models

## AIM

To build unigram, bigram and trigram language models from a text corpus and calculate word probabilities using add-one smoothing.

## ALGORITHM

1. Read and tokenize the text corpus.
2. Count unigram, bigram and trigram occurrences.
3. Calculate probabilities using add-one smoothing.
4. Calculate sentence perplexity using bigram probabilities.
5. Generate or display word probabilities from the model.

## CODE

```python
import re
from collections import Counter


def main() -> None:
    text = "Natural language processing is useful. Natural language models predict words."
    w = re.findall(r"[a-z]+", text.lower())
    V, N = len(set(w)), len(w)
    count = lambda n: Counter(tuple(w[i:i + n]) for i in range(N - n + 1))
    uni, bi, tri = count(1), count(2), count(3)

    def unigram(x):
        return (uni[(x,)] + 1) / (N + V)

    def bigram(a, b):
        return (bi[(a, b)] + 1) / (uni[(a,)] + V)

    def trigram(a, b, c):
        return (tri[(a, b, c)] + 1) / (bi[(a, b)] + V)

    print("Tokens:", N, "Vocabulary:", V)
    print("P(language):", round(unigram("language"), 4))
    print("P(language | natural):", round(bigram("natural", "language"), 4))
    print("P(processing | natural language):",
          round(trigram("natural", "language", "processing"), 4))


if __name__ == "__main__":
    main()
```

## OUTPUT

```
Tokens: 10 Vocabulary: 8
P(language): 0.1667
P(language | natural): 0.3
P(processing | natural language): 0.2
```

## RESULT

Thus, unigram, bigram and trigram language models were constructed and word probabilities were calculated successfully.
