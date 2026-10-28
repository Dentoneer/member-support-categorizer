"""TF-IDF + logistic regression baseline: the cheap model every LLM must beat."""

from __future__ import annotations

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


def train_baseline(texts: list[str], labels: list[str]) -> Pipeline:
    model = Pipeline(
        [
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)),
            ("lr", LogisticRegression(max_iter=1000, C=4.0)),
        ]
    )
    return model.fit(texts, labels)


def predict_baseline(model: Pipeline, texts: list[str]) -> list[str]:
    return [str(x) for x in model.predict(texts)]
