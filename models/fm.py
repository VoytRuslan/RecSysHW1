import copy

import numpy as np
import torch
import torch.nn.functional as F
from scipy import sparse


def bag(matrix, rows, weight, square=False):
    """
    Суммирует эмбеддинги признаков выбранных строк с учётом значений признаков.
    :param matrix: CSR-матрица признаков
    :param rows: массив индексов строк
    :param weight: тензор эмбеддингов признаков
    :param square: использовать ли квадраты эмбеддингов и значений признаков
    :return: тензор сумм эмбеддингов для выбранных строк
    """
    sub = matrix[rows]
    x = torch.from_numpy(sub.data)
    return F.embedding_bag(torch.from_numpy(sub.indices.astype(np.int64)), weight ** 2 if square else weight,
                           torch.from_numpy(sub.indptr[:-1].astype(np.int64)),
                           per_sample_weights=x ** 2 if square else x, mode="sum")


def is_seen(sorted_keys, keys):
    """
    Проверяет наличие закодированных пар в отсортированной истории взаимодействий.
    :param sorted_keys: непустой отсортированный массив ключей известных пар
    :param keys: массив проверяемых ключей user_idx*n_items+item_idx
    :return: булев массив наличия каждой пары в истории
    """
    pos = np.minimum(np.searchsorted(sorted_keys, keys), len(sorted_keys) - 1)
    return sorted_keys[pos] == keys


class FM(torch.nn.Module):
    """Факторизационная машина с BPR (семинар 05) на PyTorch.

    Вход пользователя — [one-hot id | признаки пользователя], объекта — [one-hot id | признаки объекта].
    Члены, зависящие только от пользователя, в BPR сокращаются, поэтому скор пары:
    s(u, i) = w_i^T x_i + 1/2 (|V_i^T x_i|^2 - sum v^2 x^2) + <V_u^T x_u, V_i^T x_i>.
    """

    def __init__(self, n_user_features, n_item_features, dim=64, lr=0.005, reg=1e-5, batch_size=4096, seed=42):
        """
        Создаёт параметры факторизационной машины и задаёт настройки обучения.
        :param n_user_features: количество признаков пользователя, включая ID
        :param n_item_features: количество признаков объекта, включая ID
        :param dim: размерность эмбеддингов признаков
        :param lr: скорость обучения Adam
        :param reg: коэффициент регуляризации в функции потерь
        :param batch_size: количество обучающих троек в пакете
        :param seed: зерно инициализации и выборки негативов
        """
        super().__init__()
        torch.manual_seed(seed)
        self.rng = np.random.default_rng(seed)
        self.lr, self.reg, self.batch_size = lr, reg, batch_size
        self.Vu = torch.nn.Parameter(0.01 * torch.randn(n_user_features, dim))
        self.Vi = torch.nn.Parameter(0.01 * torch.randn(n_item_features, dim))
        self.wi = torch.nn.Parameter(torch.zeros(n_item_features, 1))

    def item_part(self, items):
        """
        Вычисляет эмбеддинги объектов и зависящие только от объекта слагаемые FM.
        :param items: массив индексов объектов
        :return: кортеж (эмбеддинги объектов, линейные и внутренние слагаемые скоров)
        """
        e = bag(self.item_x, items, self.Vi)
        bias = bag(self.item_x, items, self.wi)[:, 0] + 0.5 * ((e ** 2).sum(1) - bag(self.item_x, items, self.Vi, True).sum(1))
        return e, bias

    def fit(self, user_x, item_x, positives, seen, epochs=10, eval_fn=None, patience=2):
        """
        Обучает FM по BPR с выборкой неоценённых объектов в качестве негативов.
        :param user_x: матрица признаков пользователей
        :param item_x: матрица признаков объектов
        :param positives: CSR-матрица позитивных пар (пользователи × объекты)
        :param seen: CSR-матрица всех оценённых пар для исключения из негативов
        :param epochs: максимальное количество эпох обучения
        :param eval_fn: функция оценки модели на validation; None отключает раннюю остановку
        :param patience: число эпох без улучшения validation-метрики до остановки
        :return: обученная модель; при eval_fn восстановлены параметры лучшей эпохи
        """
        self.user_x = sparse.csr_matrix(user_x, dtype=np.float32)
        self.item_x = sparse.csr_matrix(item_x, dtype=np.float32)
        pos = positives.tocoo()
        pos_u, pos_i = pos.row.astype(np.int64), pos.col.astype(np.int64)
        n_items = seen.shape[1]
        seen_keys = np.sort(seen.tocoo().row.astype(np.int64) * n_items + seen.tocoo().col)
        opt = torch.optim.Adam(self.parameters(), lr=self.lr)
        self.history, best, best_state, bad = [], -np.inf, None, 0
        for epoch in range(epochs):
            self.train()
            order = self.rng.permutation(len(pos_u))
            losses = []
            for start in range(0, len(order), self.batch_size):
                b = order[start:start + self.batch_size]
                users, items = pos_u[b], pos_i[b]
                neg = self.rng.integers(0, n_items, len(b))
                collide = is_seen(seen_keys, users * n_items + neg)
                while collide.any():  # негатив — объект, который пользователь не оценивал
                    neg[collide] = self.rng.integers(0, n_items, collide.sum())
                    collide = is_seen(seen_keys, users * n_items + neg)
                u = bag(self.user_x, users, self.Vu)
                ep, bp = self.item_part(items)
                en, bn = self.item_part(neg)
                diff = (u * ep).sum(1) + bp - (u * en).sum(1) - bn
                loss = -F.logsigmoid(diff).mean() + self.reg * ((u ** 2).sum() + (ep ** 2).sum() + (en ** 2).sum()) / len(b)
                opt.zero_grad()
                loss.backward()
                opt.step()
                losses.append(loss.item())
            row = {"epoch": epoch + 1, "loss": float(np.mean(losses))}
            if eval_fn is not None:
                self.prepare()
                row["val"] = eval_fn(self)
                if row["val"] > best:
                    best, best_state, bad = row["val"], copy.deepcopy(self.state_dict()), 0
                else:
                    bad += 1
            self.history.append(row)
            if eval_fn is not None and bad >= patience:
                break
        if best_state is not None:
            self.load_state_dict(best_state)
        self.best_epoch = int(np.argmax([r.get("val", 0) for r in self.history]) + 1) if eval_fn else epochs
        self.prepare()
        return self

    @torch.no_grad()
    def prepare(self):
        """
        Переводит модель в режим оценки и сохраняет эмбеддинги и смещения всех объектов.
        :return: None
        """
        self.eval()
        items = np.arange(self.item_x.shape[0])
        e, bias = self.item_part(items)
        self.item_emb, self.item_bias = e.numpy(), bias.numpy()

    @torch.no_grad()
    def score(self, users):
        """
        Вычисляет скоры всех объектов с использованием сохранённых представлений.
        :param users: массив индексов пользователей
        :return: матрица скоров (пользователи × объекты)
        """
        return bag(self.user_x, users, self.Vu).numpy() @ self.item_emb.T + self.item_bias[None, :]
