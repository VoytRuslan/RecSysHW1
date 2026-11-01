import numpy as np
from scipy import sparse


class ItemKNN:
    """Item-based CF: косинусное сходство объектов, k ближайших соседей, скор — сумма сходств по истории."""

    def __init__(self, k=100):
        """
        Задаёт число соседей для каждого объекта.
        :param k: количество ближайших соседей
        """
        self.k = k

    def fit(self, X):
        """
        Вычисляет косинусные сходства и оставляет ближайших соседей каждого объекта.
        :param X: матрица взаимодействий (пользователи × объекты)
        :return: обученная модель
        """
        self.X = sparse.csr_matrix(X, dtype=np.float32)
        gram = (self.X.T @ self.X).toarray()
        norms = np.sqrt(np.diag(gram))
        sim = gram / np.maximum(np.outer(norms, norms), 1e-12)
        np.fill_diagonal(sim, 0)
        # у каждого объекта оставляем k самых похожих
        n = sim.shape[0]
        neighbors = np.argpartition(-sim, self.k, axis=1)[:, :self.k]
        rows = np.repeat(np.arange(n), self.k)
        cols = neighbors.ravel()
        self.S = sparse.csr_matrix((sim[rows, cols], (rows, cols)), shape=(n, n))
        return self

    def score(self, users):
        """
        Вычисляет скоры объектов для выбранных пользователей.
        :param users: массив индексов пользователей
        :return: матрица скоров (пользователи × объекты)
        """
        return (self.X[users] @ self.S).toarray()
