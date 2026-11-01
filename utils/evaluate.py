import numpy as np
import polars as pl

from .metrics import top_k, hit_matrix, ndcg_at_k, recall_at_k, hit_rate_at_k


def evaluate(score_fn, history_csr, relevant, eval_users, k=10, batch_size=4096, item_ids=None):
    """
    Формирует top-k рекомендации и считает метрики по единому протоколу для всех моделей.
    :param score_fn: функция, возвращающая матрицу скоров всех объектов для массива user_idx
    :param history_csr: CSR-матрица оценённых в истории пар, которые исключаются из выдачи
    :param relevant: таблица релевантных пар с user_idx и item_idx
    :param eval_users: таблица оцениваемых пользователей с user_id и user_idx
    :param k: количество рекомендаций для пользователя
    :param batch_size: количество пользователей в пакете оценки
    :param item_ids: массив исходных item_id по item_idx; None не добавляет их в выдачу
    :return: кортеж (метрики по пользователям, сводка средних метрик, таблица рекомендаций)
    """
    users = eval_users.select("user_id", "user_idx").sort("user_idx")
    rel = dict(relevant.group_by("user_idx").agg(pl.col("item_idx")).iter_rows())
    user_idx = users["user_idx"].to_numpy()
    n_items = history_csr.shape[1]

    tops, ndcg, recall, hit, n_rel = [], [], [], [], []
    for start in range(0, len(user_idx), batch_size):
        batch = user_idx[start:start + batch_size]
        scores = np.asarray(score_fn(batch))
        assert scores.shape == (len(batch), n_items), scores.shape
        top = top_k(scores, history_csr[batch], k)
        rel_batch = [np.asarray(rel[u]) for u in batch]
        n = np.array([len(r) for r in rel_batch])
        hits = hit_matrix(top, rel_batch)
        tops.append(top)
        ndcg.append(ndcg_at_k(hits, n))
        recall.append(recall_at_k(hits, n))
        hit.append(hit_rate_at_k(hits))
        n_rel.append(n)

    per_user = users.with_columns(
        pl.Series("n_relevant", np.concatenate(n_rel)).cast(pl.Int32),
        pl.Series(f"ndcg{k}", np.concatenate(ndcg)),
        pl.Series(f"recall{k}", np.concatenate(recall)),
        pl.Series(f"hit{k}", np.concatenate(hit)),
    )
    summary = {f"NDCG@{k}": per_user[f"ndcg{k}"].mean(), f"Recall@{k}": per_user[f"recall{k}"].mean(),
               f"HitRate@{k}": per_user[f"hit{k}"].mean(), "users": per_user.height}

    top = np.vstack(tops)
    recs = pl.DataFrame({
        "user_id": np.repeat(users["user_id"].to_numpy(), k),
        "rank": np.tile(np.arange(1, k + 1), len(user_idx)).astype(np.int32),
        "item_idx": top.ravel().astype(np.int32),
    }).filter(pl.col("item_idx") >= 0)
    if item_ids is not None:
        recs = recs.with_columns(pl.Series("item_id", np.asarray(item_ids)[recs["item_idx"].to_numpy()]))
    return per_user, summary, recs
