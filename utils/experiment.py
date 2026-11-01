import itertools
import time

import numpy as np
import polars as pl
import torch
from scipy import sparse

from models.ease import EASE
from models.fm import FM
from models.ials import IALS, base_confidence
from models.ials_custom import fit_custom
from models.item_knn import ItemKNN
from . import data as D
from . import features as F
from .evaluate import evaluate


class Experiment:
    def __init__(self, data, seed=42):
        """
        Подготавливает общие индексы, матрицы взаимодействий и признаки для экспериментов.
        :param data: словарь таблиц interactions, items и users
        :param seed: зерно случайной инициализации моделей
        """
        self.data, self.seed = data, seed
        self.user_index, self.item_index = D.build_index(data["interactions"].filter(pl.col("split") == "train"))
        self.n_users, self.n_items = self.user_index.height, self.item_index.height
        self.inter = D.add_index(data["interactions"], self.user_index, self.item_index)
        self.item_ids = self.item_index["item_id"].to_numpy()
        assert self.inter.height == data["interactions"].height
        histories = {"T1": self.inter.filter(pl.col("timestamp") < D.T1),
                     "T2": self.inter.filter(pl.col("timestamp") < D.T2)}
        item_long, item_vocab = F.add_vocab(F.item_features(data["items"], histories), "item_id")
        user_long, user_vocab = F.add_vocab(F.user_features(data["users"], data["items"], histories), "user_id")
        self.stage = {}
        for part, cutoff, as_of in [("val", D.T1, "T1"), ("test", D.T2, "T2")]:
            h = histories[as_of]
            rel = self.inter.filter((pl.col("split") == part) & (pl.col("rating") >= 4))
            users = rel.select("user_id", "user_idx").unique().sort("user_idx")
            self.stage[part] = dict(
                cutoff=cutoff, seen=D.to_csr(h, self.n_users, self.n_items),
                pos=D.to_csr(h.filter(pl.col("rating") >= 4), self.n_users, self.n_items),
                relevant=rel, eval_users=users,
                user_x=F.feature_matrix(user_long, self.user_index, "user_id", "user_idx", as_of, user_vocab.height),
                item_x=F.feature_matrix(item_long, self.item_index, "item_id", "item_idx", as_of, item_vocab.height))

    def evaluate(self, score, part):
        """
        Оценивает скоры по общему протоколу на указанной части выборки.
        :param score: функция, возвращающая скоры всех объектов для массива пользователей
        :param part: часть выборки (val/test)
        :return: кортеж (результаты по пользователям, сводка, рекомендации)
        """
        s = self.stage[part]
        return evaluate(score, s["seen"], s["relevant"], s["eval_users"], k=10, batch_size=512, item_ids=self.item_ids)

    def fit(self, name, part, config, early_stop=False):
        """
        Обучает модель на указанной части выборки с заданной конфигурацией.
        :param name: имя модели (ItemKNN, EASE, iALS, iALS custom, FM)
        :param part: часть выборки (val/test)
        :param config: конфигурация модели (словарь)
        :param early_stop: параметр для FM, использовать ли раннюю остановку по валидации
        :return: обученная модель
        """
        c, s = dict(config), self.stage[part]
        if name == "ItemKNN":
            signal = c.pop("signal", "pos")
            return ItemKNN(**c).fit(s[signal])
        if name == "EASE":
            signal = c.pop("signal", "pos")
            return EASE(**c).fit(s[signal])
        if name == "iALS":
            alpha = c.pop("alpha")
            return IALS(**c, seed=self.seed).fit(base_confidence(s["pos"], alpha))
        if name == "iALS custom":
            return fit_custom(self.data["interactions"], s["cutoff"], self.user_index, self.item_index,
                              **c, seed=self.seed)
        if name in {"FM", "FM (ID)"}:
            epochs = c.pop("epochs", 15)
            ux = sparse.identity(self.n_users, format="csr", dtype=np.float32)
            ix = sparse.identity(self.n_items, format="csr", dtype=np.float32)
            if name == "FM":
                ux = sparse.hstack([ux, s["user_x"]]).tocsr()
                ix = sparse.hstack([ix, s["item_x"]]).tocsr()
            callback = (lambda m: self.evaluate(m.score, "val")[1]["NDCG@10"]) if early_stop else None
            return FM(ux.shape[1], ix.shape[1], **c, seed=self.seed).fit(
                ux, ix, s["pos"], s["seen"], epochs=epochs, eval_fn=callback, patience=3)
        raise ValueError(name)

    def run(self, name, part, config, early_stop=False):
        """
        Запускает обучение и оценку модели на указанной части выборки (val/test) с заданной конфигурацией.

        :param name: имя модели (ItemKNN, EASE, iALS, iALS custom, FM)
        :param part: часть выборки (val/test)
        :param config: конфигурация модели (словарь)
        :param early_stop: параметр для FM, использовать ли раннюю остановку по валидации
        :return: кортеж (результаты по пользователям, сводка, рекомендации)
        """
        started = time.perf_counter()
        if name == "Popular":
            popularity = np.asarray(self.stage[part]["seen"].sum(axis=0)).ravel()
            score = lambda users: np.broadcast_to(popularity, (len(users), self.n_items))
            model = None
        else:
            model = self.fit(name, part, config, early_stop)
            score = model.score
        per_user, summary, recs = self.evaluate(score, part)
        summary["seconds"] = time.perf_counter() - started
        if name.startswith("FM"):
            summary["epochs"] = model.best_epoch
        return per_user, summary, recs

    def search(self, name, configs):
        """
        :param name: имя модели (ItemKNN, EASE, iALS, iALS custom, FM)
        :param configs: список конфигураций для перебора
        :return: список словарей с результатами по каждой конфигурации и словарь с лучшей конфигурацией
        """
        rows, best = [], None
        for config in configs:
            _, summary, _ = self.run(name, "val", config, early_stop=name.startswith("FM"))
            chosen = dict(config)
            if "epochs" in summary:
                chosen["epochs"] = summary["epochs"]
            rows.append(dict(model=name, config=chosen, **summary))
            if best is None or summary["NDCG@10"] > best["NDCG@10"]:
                best = rows[-1]
        return rows, best["config"]


def search_spaces():
    """
    :return: словарь {имя модели: список конфигураций для перебора}
    """
    return {
        "ItemKNN": [dict(signal=s, k=k) for s, k in itertools.product(["pos", "seen"], [50, 100, 200, 400])],
        "EASE": [dict(signal=s, reg=r) for s, r in itertools.product(["pos", "seen"], [10, 20, 50, 200, 500, 1000, 3000])],
        "iALS": [dict(factors=f, reg=r, alpha=a, iterations=12)
                 for f, r, a in itertools.product([32, 64, 128], [0.001, 0.01, 0.1, 1], [5, 20, 50])],
        "iALS custom": [dict(strength=s, temperature=0.5, k=5, alpha=a, reg=r, factors=32, iterations=12)
                        for s, a, r in itertools.product([0.1, 0.3, 0.5], [10, 20, 30, 40], [0.01, 0.1, 0.3, 1])],
        "FM": [dict(dim=d, lr=lr, reg=r) for d, lr, r in itertools.product([32, 64], [0.001, 0.005], [1e-5, 1e-3])],
    }
