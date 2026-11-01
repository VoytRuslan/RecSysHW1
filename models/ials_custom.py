import numpy as np
import polars as pl
from scipy import sparse

from .ials import IALS


def build_target(history, cutoff, strength=0.5, temperature=0.5, k=5):
    """
    Смешивает абсолютную релевантность и оценку относительно личной нормы пользователя.
    :param history: таблица взаимодействий с user_id, item_id, rating и timestamp
    :param cutoff: временная отсечка; используются только взаимодействия строго до неё
    :param strength: доля относительного сигнала в таргете, от 0 до 1
    :param temperature: масштаб отклонения оценки в сигмоиде, положительное число
    :param k: вес глобального среднего при сглаживании среднего пользователя
    :return: таблица оценённых пар со средним пользователя mu_u и таргетом p
    """
    if not 0 <= strength <= 1 or temperature <= 0 or k < 0:
        raise ValueError("invalid target parameters")
    hist = history.filter(pl.col("timestamp") < cutoff)
    mu_global = hist["rating"].mean()
    means = hist.group_by("user_id").agg(
        ((pl.col("rating").sum() + k * mu_global) / (pl.len() + k)).alias("mu_u"))
    target = hist.select("user_id", "item_id", "rating").join(means, on="user_id")
    base = (pl.col("rating") >= 4).cast(pl.Float64)
    relative = 1 / (1 + (-(pl.col("rating") - pl.col("mu_u")) / temperature).exp())
    return target.with_columns(((1 - strength) * base + strength * relative).alias("p")).sort("user_id", "item_id")


def target_matrices(target, user_index, item_index, alpha=20):
    """
    Переводит кастомные таргеты и confidence в разреженные матрицы.
    :param target: таблица пар с user_id, item_id, rating и p
    :param user_index: таблица соответствия user_id и user_idx
    :param item_index: таблица соответствия item_id и item_idx
    :param alpha: добавочный confidence для оценок не ниже 4
    :return: кортеж CSR-матриц (confidence, таргеты) с одинаковой структурой
    """
    t = target.join(user_index, on="user_id").join(item_index, on="item_id")
    rows, cols = t["user_idx"].to_numpy(), t["item_idx"].to_numpy()
    shape = (user_index.height, item_index.height)
    c = np.where(t["rating"].to_numpy() >= 4, 1 + alpha, 1.0)
    C = sparse.csr_matrix((c, (rows, cols)), shape=shape, dtype=np.float64)
    P = sparse.csr_matrix((t["p"].to_numpy(), (rows, cols)), shape=shape, dtype=np.float64)
    return C, P


def fit_custom(history, cutoff, user_index, item_index, strength=0.5, temperature=0.5,
               k=5, alpha=20, factors=32, reg=0.1, iterations=12, seed=42):
    """
    Строит мягкий таргет по доступной истории и обучает общий решатель iALS.
    :param history: таблица взаимодействий с оценками и временем
    :param cutoff: временная отсечка истории
    :param user_index: таблица соответствия user_id и user_idx
    :param item_index: таблица соответствия item_id и item_idx
    :param strength: доля относительного сигнала в таргете
    :param temperature: масштаб отклонения оценки в сигмоиде
    :param k: вес глобального среднего при сглаживании
    :param alpha: добавочный confidence для оценок не ниже 4
    :param factors: размерность латентных факторов
    :param reg: коэффициент L2-регуляризации
    :param iterations: количество полных итераций ALS
    :param seed: зерно случайной инициализации
    :return: обученная модель iALS с кастомным таргетом
    """
    target = build_target(history, cutoff, strength, temperature, k)
    C, P = target_matrices(target, user_index, item_index, alpha)
    return IALS(factors, reg, iterations, seed).fit(C, P)
