import numpy as np


def top_k(scores, seen=None, k=10):
    """
    Выбирает top-k объектов, исключая известные пары и разрешая ничьи по меньшему индексу.
    :param scores: матрица скоров (пользователи × объекты)
    :param seen: CSR-матрица известных пар того же размера или None
    :param k: количество объектов в выдаче
    :return: матрица индексов top-k; недостающие позиции заполнены значением -1
    """
    scores = np.array(scores, dtype=np.float64)  # копия: маскирование не портит вход
    if seen is not None:
        rows = np.repeat(np.arange(seen.shape[0]), np.diff(seen.indptr))
        scores[rows, seen.indices] = -np.inf
    n_rows, n_items = scores.shape
    m = min(k + 1, n_items)
    # k+1 лучших кандидатов, затем точный порядок (-score, item_idx)
    part = np.argpartition(-scores, m - 1, axis=1)[:, :m] if m < n_items else np.tile(np.arange(n_items), (n_rows, 1))
    part_scores = np.take_along_axis(scores, part, axis=1)
    order = np.lexsort((part, -part_scores), axis=1)
    top = np.take_along_axis(part, order, axis=1)
    top_scores = np.take_along_axis(part_scores, order, axis=1)

    # Ничья на границе: k-й и (k+1)-й скоры равны — argpartition мог выбрать не те объекты
    if m > k:
        for r in np.flatnonzero(top_scores[:, k - 1] == top_scores[:, k]):
            cand = np.flatnonzero(scores[r] >= top_scores[r, k - 1])
            top[r, :k] = cand[np.lexsort((cand, -scores[r, cand]))][:k]
            top_scores[r, :k] = scores[r, top[r, :k]]
    top, top_scores = top[:, :k], top_scores[:, :k]
    top[np.isneginf(top_scores)] = -1
    if top.shape[1] < k:
        top = np.hstack([top, -np.ones((n_rows, k - top.shape[1]), dtype=top.dtype)])
    return top


def hit_matrix(top, relevant):
    """
    Определяет попадания рекомендаций в релевантные объекты каждого пользователя.
    :param top: матрица индексов рекомендованных объектов
    :param relevant: список массивов релевантных item_idx в порядке строк top
    :return: булева матрица попаданий; позиции с отрицательным индексом не засчитываются
    """
    return np.array([np.isin(row, rel) & (row >= 0) for row, rel in zip(top, relevant)], dtype=bool)


def dcg_discounts(k=10):
    """
    Вычисляет логарифмические скидки для позиций в выдаче.
    :param k: количество позиций
    :return: массив значений 1/log2(position+1) для позиций от 1 до k
    """
    return 1.0 / np.log2(np.arange(2, k + 2))


def ndcg_at_k(hits, n_relevant):
    """
    Считает бинарный NDCG отдельно для каждого пользователя.
    :param hits: матрица попаданий (пользователи × позиции выдачи)
    :param n_relevant: массив положительных чисел релевантных объектов у пользователей
    :return: массив значений NDCG, нормированных на идеальную выдачу длины min(k, n_relevant)
    """
    k = hits.shape[1]
    disc = dcg_discounts(k)
    idcg = np.cumsum(disc)[np.minimum(n_relevant, k) - 1]
    return (hits * disc).sum(axis=1) / idcg


def recall_at_k(hits, n_relevant):
    """
    Считает долю найденных релевантных объектов для каждого пользователя.
    :param hits: матрица попаданий в выдаче
    :param n_relevant: массив положительных чисел релевантных объектов у пользователей
    :return: массив значений Recall
    """
    return hits.sum(axis=1) / np.asarray(n_relevant)


def hit_rate_at_k(hits):
    """
    Проверяет наличие хотя бы одного попадания в выдаче каждого пользователя.
    :param hits: матрица попаданий в выдаче
    :return: массив значений 0 или 1 для расчёта среднего HitRate
    """
    return hits.any(axis=1).astype(np.float64)


def bootstrap_ci(values, n=1000, seed=42, alpha=0.05):
    """
    Оценивает перцентильный bootstrap-интервал среднего по пользователям.
    :param values: непустой массив индивидуальных значений метрики
    :param n: количество bootstrap-повторов
    :param seed: зерно выборки пользователей с возвращением
    :param alpha: уровень ошибки интервала; 0.05 соответствует доверительному уровню 95%
    :return: кортеж (среднее значение, нижняя граница, верхняя граница)
    """
    values = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    means = values[rng.integers(0, len(values), size=(n, len(values)))].mean(axis=1)
    low, high = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return values.mean(), low, high


def paired_bootstrap(metric_a, metric_b, n=1000, seed=42, alpha=0.05):
    """
    Оценивает bootstrap-интервал средней парной разности метрик a-b.
    :param metric_a: массив метрик первой модели по пользователям
    :param metric_b: массив метрик второй модели по тем же пользователям в том же порядке
    :param n: количество bootstrap-повторов
    :param seed: зерно выборки пользователей с возвращением
    :param alpha: уровень ошибки интервала
    :return: кортеж (средняя разность, нижняя граница, верхняя граница)
    """
    a, b = np.asarray(metric_a, dtype=np.float64), np.asarray(metric_b, dtype=np.float64)
    assert a.shape == b.shape
    return bootstrap_ci(a - b, n=n, seed=seed, alpha=alpha)
