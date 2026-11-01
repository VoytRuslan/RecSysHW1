import numpy as np
from scipy import sparse


class EASE:
    """EASE (Steck, 2019): B = I - P diag(1/diag(P)), P = (X^T X + lambda I)^(-1)."""

    def __init__(self, reg=500.0):
        """
        Задаёт регуляризацию модели.
        :param reg: коэффициент L2-регуляризации
        """
        self.reg = reg

    def fit(self, X):
        """
        Находит матрицу весов EASE с нулевой диагональю.
        :param X: матрица взаимодействий (пользователи × объекты)
        :return: обученная модель
        """
        self.X = sparse.csr_matrix(X, dtype=np.float32)
        gram = (self.X.T @ self.X).toarray().astype(np.float64)
        gram[np.diag_indices_from(gram)] += self.reg
        P = np.linalg.inv(gram)
        B = -P / np.diag(P)[None, :]
        np.fill_diagonal(B, 0)
        self.B = B.astype(np.float32)
        return self

    def score(self, users):
        """
        Вычисляет скоры объектов по истории выбранных пользователей.
        :param users: массив индексов пользователей
        :return: матрица скоров (пользователи × объекты)
        """
        return self.X[users] @ self.B
