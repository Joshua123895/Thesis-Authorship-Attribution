"""Skema pembagian data.

leave_one_topic_out  -> skema UTAMA
random_split         -> HANYA pembanding, dijalankan dengan fitur, model,
                        hyperparameter dan seed yang identik
"""
import numpy as np
from sklearn.model_selection import StratifiedKFold

import config as C


def leave_one_topic_out(topics):
    """Satu fold per topik: latih di semua topik lain, uji di topik ini."""
    topics = np.asarray(topics)
    folds = []
    for t in sorted(set(topics)):
        test = np.where(topics == t)[0]
        train = np.where(topics != t)[0]
        folds.append((train, test, t))
    return folds


def random_split(authors, n_splits=4):
    """Stratified k-fold berdasarkan penulis, tanpa memperhatikan topik."""
    authors = np.asarray(authors)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=C.SEED)
    return [(tr, te, f"fold{i+1}")
            for i, (tr, te) in enumerate(skf.split(np.zeros(len(authors)), authors))]
