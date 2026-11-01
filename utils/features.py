import numpy as np
import polars as pl
from scipy import sparse

# Признаки для FM. Всё, что зависит от оценок, считается только по истории до отсечки.
# Средняя оценка объекта и snapshot-агрегаты сайта не используются (утечка таргета / будущего).
MIN_PUBLISHER_ITEMS = 5
MIN_AUTHOR_ITEMS = 2
MIN_LANGUAGE_ITEMS = 20
N_TOP_AUTHORS_USER = 200
POP_EDGES = [20, 40, 80, 160, 320, 640, 1280, 2560]
POP_LABELS = ["<20", "20-39", "40-79", "80-159", "160-319", "320-639", "640-1279", "1280-2559", ">=2560"]
YEAR_EDGES = [1800, 1900, 1950, 1980, 1990, 2000, 2005, 2010]
YEAR_LABELS = ["<1800", "1800-1899", "1900-1949", "1950-1979", "1980-1989", "1990-1999",
               "2000-2004", "2005-2009", ">=2010"]
PAGE_EDGES = [49, 81, 121, 201, 401]
PAGE_LABELS = ["<=48", "49-80", "81-120", "121-200", "201-400", ">400"]
ENGLISH = {"eng", "en-US", "en-GB", "en-CA", "en"}


def cut(values, edges, labels):
    """
    Разбивает числовой признак на интервалы, выделяя пропуски в unknown.
    :param values: выражение Polars со значениями признака
    :param edges: упорядоченный список границ интервалов
    :param labels: названия интервалов, на одно больше числа границ
    :return: выражение Polars с категорией для каждого значения
    """
    expr = pl.when(values.is_null()).then(pl.lit("unknown"))
    for e, lab in zip(edges, labels[:-1]):
        expr = expr.when(values < e).then(pl.lit(lab))
    return expr.otherwise(pl.lit(labels[-1]))


def edge_labels(edges, fmt):
    """
    Формирует текстовые названия интервалов по их границам.
    :param edges: упорядоченный список конечных границ
    :param fmt: строка форматирования числовой границы
    :return: список названий интервалов, включая крайние с бесконечными границами
    """
    bounds = [-np.inf, *edges, np.inf]
    return [f"[{fmt.format(lo)},{fmt.format(hi)})" for lo, hi in zip(bounds[:-1], bounds[1:])]


def norm_format(f):
    """
    Приводит исходное название формата книги к общей категории.
    :param f: строка с форматом книги или None
    :return: нормализованная категория формата; unknown для пустого значения
    """
    if f is None or not f.strip():
        return "unknown"
    f = f.lower()
    if "audio" in f:
        return "audio"
    if any(k in f for k in ["ebook", "e-book", "kindle", "html", "e-chap", "nook", "pdf"]):
        return "ebook"
    if "chapbook" in f:
        return "chapbook"
    if any(k in f for k in ["hardcover", "library binding", "leather", "board book", "hardback"]):
        return "hardcover"
    return "paperback" if "paper" in f else "other"


def to_long(df, id_col, as_of, fields):
    """
    Преобразует категориальные признаки в длинную таблицу с единичными весами.
    :param df: таблица сущностей и признаков
    :param id_col: имя столбца с идентификатором сущности
    :param as_of: метка временной отсечки (T1/T2)
    :param fields: список столбцов признаков для преобразования
    :return: таблица с идентификатором, as_of, field, value и weight
    """
    return pl.concat([df.select(pl.col(id_col), pl.lit(as_of).alias("as_of"), pl.lit(f).alias("field"),
                                pl.col(f).cast(pl.Utf8).alias("value"), pl.lit(1.0, pl.Float32).alias("weight"))
                      for f in fields])


def item_features(items, histories):
    """
    Формирует категориальные признаки объектов и популярность на временных отсечках.
    :param items: таблица метаданных произведений
    :param histories: словарь историй {T1: оценки до T1, T2: оценки до T2}
    :return: длинная таблица признаков объектов для T1 и T2
    """
    pub_keep = items.group_by("publisher").len().filter(pl.col("len") >= MIN_PUBLISHER_ITEMS).drop_nulls()["publisher"]
    auth_keep = items.group_by("author_id").len().filter(pl.col("len") >= MIN_AUTHOR_ITEMS)["author_id"]
    lang = items["language_code"].map_elements(lambda c: "en" if c in ENGLISH else c, return_dtype=pl.Utf8)
    lang_keep = set(lang.value_counts().filter(pl.col("count") >= MIN_LANGUAGE_ITEMS)["language_code"].drop_nulls())
    static = items.select(
        "item_id",
        pl.when(pl.col("author_id").is_in(auth_keep.implode())).then(pl.col("author_id"))
        .otherwise(pl.lit("single")).alias("i_author"),
        pl.when(pl.col("publisher").is_null()).then(pl.lit("unknown"))
        .when(pl.col("publisher").is_in(pub_keep.implode())).then(pl.col("publisher"))
        .otherwise(pl.lit("other")).alias("i_publisher"),
        cut(pl.col("publication_year"), YEAR_EDGES, YEAR_LABELS).alias("i_year"),
        cut(pl.when(pl.col("num_pages") > 0).then(pl.col("num_pages")), PAGE_EDGES, PAGE_LABELS).alias("i_pages"),
        pl.col("format").map_elements(norm_format, return_dtype=pl.Utf8, skip_nulls=False).alias("i_format"),
        pl.col("language_code").map_elements(
            lambda c: "unknown" if c is None else "en" if c in ENGLISH else c if c in lang_keep else "other",
            return_dtype=pl.Utf8, skip_nulls=False).alias("i_language"),
    )
    parts = []
    for as_of, hist in histories.items():
        pop = hist.group_by("item_id").len().with_columns(pl.col("len").map_batches(
            lambda s: pl.Series([POP_LABELS[i] for i in np.searchsorted(POP_EDGES, s.to_numpy(), side="right")]))
            .alias("i_pop"))
        x = static.join(pop.select("item_id", "i_pop"), on="item_id", how="left")
        parts.append(to_long(x, "item_id", as_of, ["i_author", "i_publisher", "i_year", "i_pages",
                                                   "i_format", "i_language", "i_pop"]))
    return pl.concat(parts)


def user_features(users, items, histories):
    """
    Формирует категории пользовательского профиля и доли оценок по авторам и эпохам.
    :param users: таблица профилей с for_split=val для T1 и for_split=test для T2
    :param items: таблица метаданных произведений
    :param histories: словарь историй {T1: оценки до T1, T2: оценки до T2}
    :return: длинная таблица пользовательских признаков для T1 и T2
    """
    edges = {"n_ratings": [7, 10, 15, 25, 50, 100],
             "share_5": [0.001, 0.2, 0.4, 0.6, 0.8, 0.999], "share_le2": [0.001, 0.1, 0.2, 0.4],
             "share_reviewed": [0.001, 0.1, 0.3, 0.6], "days_since_first": [180, 365, 2 * 365, 4 * 365, 7 * 365],
             "days_since_last": [30, 90, 365, 2 * 365, 5 * 365]}
    u1 = users.filter(pl.col("for_split") == "val")
    for col in ["mean_rating", "std_rating"]:  # квантильные границы по T1
        edges[col] = list(np.unique(np.quantile(u1[col].to_numpy(), np.linspace(0, 1, 7)[1:-1])))
    edges = {c: edges[c] for c in ["n_ratings", "mean_rating", "std_rating", "share_5", "share_le2",
                                   "share_reviewed", "days_since_first", "days_since_last"]}
    labels = {c: edge_labels(e, "{:g}" if c.startswith(("n_", "days")) else "{:.3f}") for c, e in edges.items()}

    item_meta = items.select("item_id", "author_id", cut(pl.col("publication_year"), YEAR_EDGES, YEAR_LABELS).alias("era"))
    top_authors = (histories["T1"].join(item_meta, on="item_id").group_by("author_id").len()
                   .sort(["len", "author_id"], descending=[True, False]).head(N_TOP_AUTHORS_USER)["author_id"])
    parts = []
    for as_of, for_split in [("T1", "val"), ("T2", "test")]:
        u = users.filter(pl.col("for_split") == for_split)
        cats = u.select("user_id", *[pl.col(c).map_batches(
            lambda s, c=c: pl.Series([labels[c][i] for i in np.searchsorted(edges[c], s.to_numpy(), side="right")]))
            .alias("u_" + c) for c in edges])
        parts.append(to_long(cats, "user_id", as_of, ["u_" + c for c in edges]))
        h = histories[as_of].join(item_meta, on="item_id")
        n_u = h.group_by("user_id").len().rename({"len": "n_u"})
        # доли оценок пользователя по топ-авторам и по эпохам публикации
        for field, col, frame in [("u_author_share", "author_id", h.filter(pl.col("author_id").is_in(top_authors.implode()))),
                                  ("u_era_share", "era", h)]:
            parts.append(frame.group_by("user_id", col).len().join(n_u, on="user_id").select(
                "user_id", pl.lit(as_of).alias("as_of"), pl.lit(field).alias("field"), pl.col(col).alias("value"),
                (pl.col("len") / pl.col("n_u")).cast(pl.Float32).alias("weight")))
    return pl.concat(parts)


def add_vocab(long, id_col):
    """
    Строит общий словарь признаков и добавляет их числовые индексы.
    :param long: длинная таблица признаков с field, value, as_of и weight
    :param id_col: имя столбца с идентификатором сущности
    :return: кортеж (таблица с feature_idx, словарь соответствий field/value → feature_idx)
    """
    vocab = (long.select("field", "value").unique().sort("field", "value")
             .with_row_index("feature_idx").with_columns(pl.col("feature_idx").cast(pl.Int32)))
    long = long.join(vocab, on=["field", "value"]).select(id_col, "as_of", "field", "value", "feature_idx", "weight")
    return long.sort("as_of", id_col, "feature_idx"), vocab


def feature_matrix(long, index, id_col, idx_col, as_of, n_features):
    """
    Собирает разреженную матрицу признаков в общем порядке индексов сущностей.
    :param long: длинная таблица признаков с feature_idx и weight
    :param index: таблица соответствия идентификаторов и числовых индексов сущностей
    :param id_col: имя столбца с исходным идентификатором
    :param idx_col: имя столбца с числовым индексом
    :param as_of: метка временной отсечки (T1/T2)
    :param n_features: общее количество признаков в словаре
    :return: CSR-матрица размера количество сущностей × n_features
    """
    x = long.filter(pl.col("as_of") == as_of).join(index, on=id_col)
    return sparse.csr_matrix((x["weight"].to_numpy(), (x[idx_col].to_numpy(), x["feature_idx"].to_numpy())),
                             shape=(index.height, n_features), dtype=np.float32)
