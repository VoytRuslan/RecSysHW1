import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import polars as pl
import requests
from scipy import sparse

T1 = datetime(2016, 11, 1, tzinfo=timezone.utc)
T2 = datetime(2017, 5, 1, tzinfo=timezone.utc)
T3 = datetime(2017, 11, 1, tzinfo=timezone.utc)
START = datetime(2007, 1, 1, tzinfo=timezone.utc)
BASE_URL = "https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/"
FILES = {
    "byGenre/goodreads_interactions_poetry.json.gz": "65fb9777f5e482695f0e4b71ef9884f6a7e5ab31c224dabc7ae0c2dcd60e49de",
    "byGenre/goodreads_reviews_poetry.json.gz": "f0f1c8d403cb67328c801bde6ae5c54cccd937b80979fefb042def044df82b78",
    "byGenre/goodreads_books_poetry.json.gz": "11a71d924f245323af8e0c16d9cc7b6beefb18cf155cb58434f8dc120febaaab",
    "goodreads_book_authors.json.gz": "85e6628dee5102c70a4a296f6d68c10ff60b0cf72714444b1e721abeffe9f22d",
    "goodreads_book_works.json.gz": "1166d532fe4e39cb68402b22b97668b9336442e6ba22ab946374e1bda10ae4fd",
}


def sha256(path):
    """
    Вычисляет контрольную сумму файла.
    :param path: путь к файлу
    :return: SHA-256 в виде шестнадцатеричной строки
    """
    with open(path, "rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def download(raw_dir):
    """
    Скачивает исходные файлы Goodreads и проверяет их контрольные суммы.
    :param raw_dir: директория для исходных файлов
    :return: None; корректные существующие файлы не скачиваются повторно
    """
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    for remote, digest in FILES.items():
        path = raw_dir / Path(remote).name
        if path.exists() and sha256(path) == digest:
            continue
        tmp = path.with_suffix(path.suffix + ".part")
        with requests.get(BASE_URL + remote, stream=True, timeout=120) as r, tmp.open("wb") as f:
            r.raise_for_status()
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)
        if sha256(tmp) != digest:
            raise ValueError(f"SHA-256 mismatch: {path.name}")
        tmp.replace(path)


def date_column(name):
    """
    Создаёт выражение для разбора строковых дат Goodreads в UTC.
    :param name: имя столбца с датами
    :return: выражение Polars для преобразования столбца; пустые и некорректные даты становятся null
    """
    return pl.col(name).replace("", None).str.strptime(
        pl.Datetime("us", "UTC"), "%a %b %d %H:%M:%S %z %Y", strict=False)


def read_items(raw_dir):
    """
    Читает справочники изданий, произведений и авторов и собирает метаданные произведений.
    :param raw_dir: путь к директории с исходными файлами в формате Path
    :return: кортеж (таблица изданий, таблица метаданных произведений)
    """
    books = pl.read_ndjson(raw_dir / "goodreads_books_poetry.json.gz", infer_schema_length=None).with_columns(
        pl.col("book_id").cast(pl.Int64),
        pl.col("ratings_count").cast(pl.Int64, strict=False).fill_null(0),
        pl.col("num_pages").replace("", None).cast(pl.Int64, strict=False),
        pl.col("publication_year").replace("", None).cast(pl.Int64, strict=False))
    works = pl.read_ndjson(raw_dir / "goodreads_book_works.json.gz", infer_schema_length=None).select(
        "work_id", "original_title",
        pl.col("original_publication_year").replace("", None).cast(pl.Int64, strict=False))
    authors = pl.read_ndjson(raw_dir / "goodreads_book_authors.json.gz", infer_schema_length=None).select(
        "author_id", pl.col("name").alias("author_name"))
    rep = (books.sort(["work_id", "ratings_count", "book_id"], descending=[False, True, False])
           .unique("work_id", keep="first", maintain_order=True))
    sizes = books.group_by("work_id").agg(pl.col("num_pages").median(),
                                          pl.col("publication_year").min().alias("min_year"))
    items = (rep.select("work_id", "title", "publisher", "format", "language_code", "description",
                        pl.col("authors").list.first().struct.field("author_id").alias("author_id"))
             .join(sizes, on="work_id").join(works, on="work_id", how="left")
             .join(authors, on="author_id", how="left")
             .with_columns(pl.coalesce(pl.col("original_title").replace("", None), pl.col("title")).alias("title"),
                           pl.coalesce("original_publication_year", "min_year").alias("publication_year"),
                           *[pl.col(c).replace("", None) for c in ["publisher", "format", "language_code", "description"]])
             .drop("original_title", "original_publication_year", "min_year").rename({"work_id": "item_id"})
             .sort("item_id"))
    return books, items


def user_profiles(interactions, cutoff, part):
    """
    Вычисляет признаки пользователей по истории строго до временной отсечки.
    :param interactions: таблица взаимодействий с оценками, временем и признаком has_review
    :param cutoff: временная отсечка истории
    :param part: часть выборки, для которой строятся признаки (val/test)
    :return: таблица пользовательских признаков с метками for_split и as_of
    """
    h = interactions.filter(pl.col("timestamp") < cutoff)
    cut = pl.lit(cutoff).cast(pl.Datetime("us", "UTC"))
    return h.group_by("user_id").agg(
        pl.len().alias("n_ratings"), pl.col("rating").mean().alias("mean_rating"),
        pl.col("rating").std().fill_null(0).alias("std_rating"),
        (pl.col("rating") == 5).mean().alias("share_5"),
        (pl.col("rating") <= 2).mean().alias("share_le2"), pl.col("has_review").mean().alias("share_reviewed"),
        (cut - pl.col("timestamp").min()).dt.total_days().alias("days_since_first"),
        (cut - pl.col("timestamp").max()).dt.total_days().alias("days_since_last"),
    ).with_columns(pl.lit(part).alias("for_split"), pl.lit(cutoff).alias("as_of")).sort("user_id")


def prepare(raw_dir, processed_dir, rebuild=False):
    """
    Очищает данные, формирует временные сплиты и признаки либо читает готовый результат.
    :param raw_dir: директория с исходными файлами Goodreads
    :param processed_dir: директория для подготовленных таблиц и сводки очистки
    :param rebuild: пересоздавать ли подготовленные файлы при наличии сохранённого результата
    :return: кортеж (словарь таблиц interactions, items и users, сводка подготовки)
    """
    raw_dir, processed_dir = Path(raw_dir), Path(processed_dir)
    names = ["interactions", "items", "users"]
    if not rebuild and (processed_dir / "summary.json").exists() and all(
            (processed_dir / f"{name}.parquet").exists() for name in names):
        summary = json.loads((processed_dir / "summary.json").read_text(encoding="utf-8"))
        return {n: pl.read_parquet(processed_dir / f"{n}.parquet") for n in names}, summary
    raw = pl.read_ndjson(raw_dir / "goodreads_interactions_poetry.json.gz", infer_schema_length=None).with_columns(
        date_column("date_added"), date_column("date_updated"), date_column("read_at"))
    books, all_items = read_items(raw_dir)
    reviews = pl.read_ndjson(raw_dir / "goodreads_reviews_poetry.json.gz", infer_schema_length=None).select(
        "review_id", (pl.col("review_text").str.len_chars() > 0).alias("has_review"))
    funnel = []

    def record(label, frame, item="item_id"):
        """
        Добавляет размеры текущей выборки в журнал очистки.
        :param label: название шага подготовки
        :param frame: таблица взаимодействий после шага
        :param item: имя столбца с идентификатором объекта
        :return: None
        """
        funnel.append(dict(step=label, interactions=frame.height, users=frame["user_id"].n_unique(),
                           items=frame[item].n_unique()))

    record("Исходные полки", raw, "book_id")
    missing = {c: int(raw[c].null_count() + (raw[c].cast(pl.Utf8) == "").sum())
               for c in ["user_id", "book_id", "rating", "date_added", "read_at"]}
    df = raw.filter(pl.col("user_id").is_not_null() & (pl.col("user_id") != "")
                    & pl.col("book_id").is_not_null() & (pl.col("book_id") != "")
                    & pl.col("date_added").is_not_null())
    record("Корректные ID и дата", df, "book_id")
    df = df.filter(pl.col("rating").is_between(1, 5))
    record("Только оценки 1–5", df, "book_id")
    mapping = books.select(pl.col("book_id").cast(pl.Utf8), pl.col("work_id").alias("item_id"))
    df = (df.join(mapping, on="book_id").filter(pl.col("item_id").is_not_null())
          .join(reviews, on="review_id", how="left").with_columns(pl.col("has_review").fill_null(False)))
    duplicates = df.height - df.select("user_id", "item_id").n_unique()
    first = df.group_by("user_id", "item_id").agg(pl.col("date_added").min().alias("first_added"))
    df = (df.sort(["user_id", "item_id", "date_updated", "date_added", "book_id"])
          .unique(["user_id", "item_id"], keep="last", maintain_order=True).join(first, on=["user_id", "item_id"])
          .with_columns(pl.col("first_added").alias("date_added"), pl.col("first_added").alias("timestamp")))
    record("Издания → произведения, без дублей", df)
    df = df.filter((pl.col("timestamp") >= START) & (pl.col("timestamp") < T3))
    record("Окно 2007-01-01 … 2017-10-31", df)
    good_items = all_items.filter(pl.col("title").is_not_null() & pl.col("author_id").is_not_null())["item_id"]
    df = df.filter(pl.col("item_id").is_in(good_items.implode()))
    record("Есть название и автор", df)
    profile = df.group_by("user_id", "rating").len().group_by("user_id").agg(
        pl.col("len").sum().alias("n"), (pl.col("len").max() / pl.col("len").sum()).alias("mode_share"))
    bad = profile.filter((pl.col("n") >= 20) & (pl.col("mode_share") >= 0.95))["user_id"]
    df = df.filter(~pl.col("user_id").is_in(bad.implode()))
    record("Без длинных однообразных историй", df)
    df = df.with_columns(pl.when(pl.col("timestamp") < T1).then(pl.lit("train"))
                         .when(pl.col("timestamp") < T2).then(pl.lit("val")).otherwise(pl.lit("test")).alias("split"))
    train = df.filter(pl.col("split") == "train")
    while True:
        size = train.height
        train = train.filter(pl.len().over("user_id") >= 5).filter(pl.len().over("item_id") >= 10)
        if train.height == size:
            break
    rest = df.filter((pl.col("split") != "train")
                     & pl.col("user_id").is_in(train["user_id"].unique().implode())
                     & pl.col("item_id").is_in(train["item_id"].unique().implode()))
    df = pl.concat([train, rest])
    record("Train k-core: user≥5, item≥10", df)
    interactions = df.select("user_id", "item_id", "rating", "timestamp", "split", "has_review",
                             "date_added", "date_updated", "read_at").sort("timestamp", "user_id", "item_id")
    items = all_items.filter(pl.col("item_id").is_in(df["item_id"].unique().implode()))
    users = pl.concat([user_profiles(interactions, T1, "val"), user_profiles(interactions, T2, "test")])
    assert 100_000 <= interactions.height <= 2_000_000
    assert interactions["user_id"].n_unique() >= 1000 and items.height >= 1000
    assert interactions.select("user_id", "item_id").is_duplicated().sum() == 0
    tables = dict(interactions=interactions, items=items, users=users)
    summary = dict(funnel=funnel, missing=missing, work_duplicates=duplicates,
                   raw_ratings=raw.group_by("rating").len().sort("rating").to_dicts())
    processed_dir.mkdir(parents=True, exist_ok=True)
    for name, frame in tables.items():
        frame.write_parquet(processed_dir / f"{name}.parquet")
    (processed_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return tables, summary


def build_index(train):
    """
    Строит индексы пользователей и объектов по отсортированным идентификаторам train.
    :param train: таблица обучающих взаимодействий с user_id и item_id
    :return: кортеж таблиц соответствий (user_id → user_idx, item_id → item_idx)
    """
    users = train.select("user_id").unique().sort("user_id").with_row_index("user_idx")
    items = train.select("item_id").unique().sort("item_id").with_row_index("item_idx")
    return users, items


def add_index(frame, user_index, item_index):
    """
    Добавляет числовые индексы пользователей и объектов к таблице взаимодействий.
    :param frame: таблица с user_id и item_id
    :param user_index: таблица соответствия user_id и user_idx
    :param item_index: таблица соответствия item_id и item_idx
    :return: таблица с добавленными индексами; пары без соответствий исключены
    """
    return frame.join(user_index, on="user_id").join(item_index, on="item_id")


def to_csr(frame, n_users, n_items):
    """
    Преобразует индексированные взаимодействия в разреженную матрицу.
    :param frame: таблица пар с user_idx и item_idx
    :param n_users: общее количество пользователей
    :param n_items: общее количество объектов
    :return: CSR-матрица размера n_users × n_items с единичным вкладом каждой пары
    """
    return sparse.csr_matrix((np.ones(frame.height, dtype=np.float32),
                             (frame["user_idx"].to_numpy(), frame["item_idx"].to_numpy())),
                            shape=(n_users, n_items))
