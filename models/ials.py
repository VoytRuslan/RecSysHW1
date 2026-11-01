import numpy as np
from scipy import sparse


def update_rows(C, Y, reg, preferences=None, batch_size=512):
    """
    Выполняет полушаг ALS при фиксированных факторах другой стороны.
    :param C: CSR-матрица confidence; у пропущенных пар c=1 и p=0
    :param Y: матрица фиксированных факторов
    :param reg: коэффициент L2-регуляризации
    :param preferences: CSR-матрица таргетов с той же структурой, что C; None означает p=1 у сохранённых пар
    :param batch_size: количество строк в одном пакете решения систем
    :return: матрица обновлённых факторов
    """
    d = Y.shape[1]
    base = Y.T @ Y + reg * np.eye(d)
    result = np.empty((C.shape[0], d), dtype=np.float64)
    for begin in range(0, C.shape[0], batch_size):
        end = min(begin + batch_size, C.shape[0])
        A = np.broadcast_to(base, (end - begin, d, d)).copy()
        b = np.zeros((end - begin, d))
        for offset, u in enumerate(range(begin, end)):
            start, stop = C.indptr[u:u + 2]
            local = Y[C.indices[start:stop]]
            c = C.data[start:stop]
            p = np.ones(len(c)) if preferences is None else preferences.data[start:stop]
            A[offset] += local.T @ ((c - 1)[:, None] * local)
            b[offset] = local.T @ (c * p)
        result[begin:end] = np.linalg.solve(A, b[..., None])[..., 0]
    return result


def base_confidence(positives, alpha):
    """
    Формирует confidence для базового iALS.
    :param positives: разреженная матрица позитивных взаимодействий
    :param alpha: добавочный вес позитивной пары, c=1+alpha
    :return: CSR-матрица confidence
    """
    C = sparse.csr_matrix(positives, dtype=np.float64)
    C.data[:] = 1 + alpha
    return C


class IALS:
    def __init__(self, factors=32, reg=0.1, iterations=12, seed=42):
        """
        Задаёт параметры обучения iALS.
        :param factors: размерность латентных факторов
        :param reg: коэффициент L2-регуляризации
        :param iterations: количество полных итераций ALS
        :param seed: зерно случайной инициализации
        """
        self.factors, self.reg, self.iterations, self.seed = factors, reg, iterations, seed

    def fit(self, confidence, preferences=None):
        """
        Обучает факторы пользователей и объектов чередующимися полушагами ALS.
        :param confidence: разреженная матрица confidence, значения не меньше 1
        :param preferences: матрица таргетов с той же структурой; None означает бинарные позитивы
        :return: обученная модель
        """
        C = sparse.csr_matrix(confidence, dtype=np.float64)
        C.sort_indices()
        P = Pt = None
        if preferences is not None:
            P = sparse.csr_matrix(preferences, dtype=np.float64)
            P.sort_indices()
            if (P.shape != C.shape or not np.array_equal(P.indptr, C.indptr)
                    or not np.array_equal(P.indices, C.indices)):
                raise ValueError("C and P must have the same sparse structure")
            if not np.isfinite(P.data).all():
                raise ValueError("preferences must be finite")
            Pt = P.T.tocsr()
        if not np.isfinite(C.data).all() or np.any(C.data < 1):
            raise ValueError("confidence must be finite and >= 1")
        Ct = C.T.tocsr()
        rng = np.random.default_rng(self.seed)
        self.item_factors = 0.01 * rng.standard_normal((C.shape[1], self.factors))
        for _ in range(self.iterations):
            self.user_factors = update_rows(C, self.item_factors, self.reg, P)
            self.item_factors = update_rows(Ct, self.user_factors, self.reg, Pt)
        self.user_factors = self.user_factors.astype(np.float32)
        self.item_factors = self.item_factors.astype(np.float32)
        return self

    def score(self, users):
        """
        Вычисляет скалярные произведения факторов пользователей и объектов.
        :param users: массив индексов пользователей
        :return: матрица скоров (пользователи × объекты)
        """
        return self.user_factors[users] @ self.item_factors.T
