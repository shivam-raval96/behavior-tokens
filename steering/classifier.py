"""Concept probe: a linear classifier over pooled activations.

Trains logistic regression to separate positive (concept) from negative
activations at the steering layer. Used both to validate the concept is
linearly decodable and to score generations along the steering curve.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from steering.config import Config


@dataclass
class ConceptClassifier:
    clf: LogisticRegression
    layer: int
    pooling: str
    train_acc: float
    test_acc: float

    def predict(self, activations: torch.Tensor) -> np.ndarray:
        """+1 / -1 predictions for [N, hidden] activations."""
        return self.clf.predict(_np(activations))

    def prob_pos(self, activations: torch.Tensor) -> np.ndarray:
        """P(concept present) for [N, hidden] activations."""
        idx = list(self.clf.classes_).index(1)
        return self.clf.predict_proba(_np(activations))[:, idx]


def _np(x: torch.Tensor) -> np.ndarray:
    return x.detach().cpu().float().numpy()


def train_classifier(cfg: Config, A_pos: torch.Tensor, A_neg: torch.Tensor) -> ConceptClassifier:
    X = np.concatenate([_np(A_pos), _np(A_neg)])
    y = np.concatenate([np.ones(len(A_pos), int), -np.ones(len(A_neg), int)])

    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=cfg.clf_test_frac, random_state=cfg.seed, stratify=y
    )
    clf = LogisticRegression(C=cfg.clf_C, max_iter=1000)
    clf.fit(Xtr, ytr)
    return ConceptClassifier(
        clf=clf, layer=cfg.layer, pooling=cfg.pooling,
        train_acc=float(clf.score(Xtr, ytr)),
        test_acc=float(clf.score(Xte, yte)),
    )


if __name__ == "__main__":
    # Synthetic separable data — validates training logic without the model.
    torch.manual_seed(0)
    A_pos = torch.randn(50, 2048) + 0.6
    A_neg = torch.randn(50, 2048) - 0.6
    cfg = Config()
    cc = train_classifier(cfg, A_pos, A_neg)
    print(f"train_acc={cc.train_acc:.3f} test_acc={cc.test_acc:.3f}")
    print("prob_pos sample:", cc.prob_pos(A_pos[:3]).round(3))
