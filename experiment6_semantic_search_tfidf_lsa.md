# EXPERIMENT 6: Implementing a Semantic Search Engine

## AIM

To implement a semantic search engine using TF-IDF, Latent Semantic Analysis and cosine similarity to rank documents for a query.

## ALGORITHM

1. Prepare a collection of documents.
2. Convert documents into TF-IDF vectors.
3. Apply Truncated SVD to obtain a lower-dimensional representation.
4. Convert the query using the same TF-IDF and SVD transformations.
5. Rank documents by cosine similarity and display the top results.

## CODE

```python
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize


def main() -> None:
    docs = ["Deep learning uses neural networks",
            "Doctors diagnose diseases using images",
            "Banks manage interest rates"]
    tfidf = TfidfVectorizer(stop_words="english")
    X = tfidf.fit_transform(docs)
    svd = TruncatedSVD(n_components=2, random_state=42)
    D = normalize(svd.fit_transform(X))

    query = "medical disease diagnosis"
    q = normalize(svd.transform(tfidf.transform([query])))
    scores = (D @ q.T).ravel()
    print("Search query:", query)
    print("Ranked results:")
    for rank, i in enumerate(np.argsort(-scores), start=1):
        print(f"{rank}. {docs[i]} (score: {round(float(scores[i]), 3)})")


if __name__ == "__main__":
    main()
```

## OUTPUT

```
Search query: medical disease diagnosis
Ranked results:
1. Deep learning uses neural networks (score: 0.0)
2. Doctors diagnose diseases using images (score: 0.0)
3. Banks manage interest rates (score: 0.0)
```

## RESULT

Thus, a semantic search engine was implemented using TF-IDF, SVD and cosine similarity.
