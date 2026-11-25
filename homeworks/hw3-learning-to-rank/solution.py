#!/usr/bin/env python
# ДЗ 3. Обучение ранжированию
#
# Обучение:  ./solution.py --train --model_file=model.bin PATH-TO-DATA
# Сабмишн:   ./solution.py --model_file=model.bin --submission_file=submission.csv PATH-TO-DATA

import argparse
import os
from timeit import default_timer as timer

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool, sum_models


THREADS = 6
MODEL_SETTINGS = (
    (42, 500, 1.0),
    (72, 450, 0.75),
)
QUERY_ID = "QueryId"
TARGET = "Label"
OBJECT_ID = "ObjectId"
RELATIVE_FEATURES = (
    "title_len", "body_len", "url_len", "url_tokens", "url_depth",
    "domain_docs", "body_cyr_share", "body_digit_share", "body_uniq_ratio",
    "title_in_body", "bm25_title", "tf_sum_title", "idf_coverage_title",
    "bm25_body", "tf_sum_body", "idf_coverage_body", "bm25_url", "tf_sum_url",
    "idf_coverage_url", "lm_title", "lm_body", "bm25f", "bm25_stem_title",
    "coverage_stem_title", "bm25_stem_body", "coverage_stem_body",
    "body_min_window", "body_first_pos", "body_matched_first50",
    "title_trigram_jaccard", "query_words_in_domain", "query_translit_in_url",
    "domain_prior", "domain_train_pairs",
)
QUERY_CONTEXT = (
    "bm25f", "bm25_title", "bm25_body", "bm25_stem_title", "bm25_stem_body",
    "bm25_url", "lm_title", "lm_body", "title_trigram_jaccard",
    "coverage_title", "coverage_body", "domain_prior",
)


def make_features(data):
    raw = data.drop(columns=[QUERY_ID, TARGET, OBJECT_ID], errors="ignore").astype(np.float32)
    grouped = raw.groupby(data[QUERY_ID], sort=False)

    relative = list(RELATIVE_FEATURES)
    means = grouped[relative].transform("mean")
    std = grouped[relative].transform("std").fillna(0).replace(0, 1)
    standardized = raw[relative].sub(means).div(std).add_suffix("_query_z")

    rank_columns = [column for column in relative if f"{column}_rank" not in raw.columns]
    ranks = (
        grouped[rank_columns].rank(method="average", pct=True)
        .astype(np.float32)
        .add_suffix("_query_percentile")
    )
    context = grouped[list(QUERY_CONTEXT)].transform("max").add_suffix("_query_max")

    derived = {}
    query_idf = raw["q_idf_sum"].clip(lower=1)
    for zone, length in (("title", "title_len"), ("body", "body_len"), ("url", "url_tokens")):
        derived[f"bm25_{zone}_per_idf"] = raw[f"bm25_{zone}"] / query_idf
        derived[f"tf_{zone}_per_token"] = raw[f"tf_sum_{zone}"] / raw[length].clip(lower=1)
    for zone in ("title", "body"):
        derived[f"stem_coverage_gain_{zone}"] = raw[f"coverage_stem_{zone}"] - raw[f"coverage_{zone}"]
    ratios = pd.DataFrame(derived, index=data.index)
    return pd.concat([raw, standardized, ranks, context, ratios], axis=1)


def fit_ranker(data):
    data = data.sort_values(QUERY_ID, kind="stable")

    labels = data[TARGET] - data.groupby(QUERY_ID, sort=False)[TARGET].transform("mean")
    train_pool = Pool(make_features(data), label=labels, group_id=data[QUERY_ID])

    models = []
    for seed, iterations, feature_fraction in MODEL_SETTINGS:
        model = CatBoostRegressor(
            loss_function="RMSE",
            iterations=iterations,
            learning_rate=0.05,
            depth=7,
            rsm=feature_fraction,
            random_seed=seed,
            thread_count=THREADS,
            allow_writing_files=False,
            verbose=100,
        )
        models.append(model.fit(train_pool))

    # Both models predict on the same scale, so the merged model can average their scores.
    return sum_models(models, weights=[0.5, 0.5])


def train(data_dir, model_file):
    data = pd.read_csv(os.path.join(data_dir, "train.csv"))
    model = fit_ranker(data)
    model.save_model(model_file)


def predict(data_dir, model_file, submission_file):
    test = pd.read_csv(os.path.join(data_dir, "test.csv"))
    model = CatBoostRegressor().load_model(model_file)
    features = make_features(test)
    scores = model.predict(features[model.feature_names_], thread_count=THREADS)
    submission = pd.DataFrame({OBJECT_ID: test[OBJECT_ID], "Score": scores})
    submission.to_csv(submission_file, index=False)


def main():
    parser = argparse.ArgumentParser(description='Learning to rank homework solution')
    parser.add_argument('--train', action='store_true', help='обучить модель и сохранить её в model_file')
    parser.add_argument('--model_file', required=True, help='файл модели')
    parser.add_argument('--submission_file', help='куда записать сабмишн для Kaggle')
    parser.add_argument('data_dir', help='папка с данными соревнования')
    args = parser.parse_args()

    start = timer()

    if args.train:
        train(args.data_dir, args.model_file)
    else:
        if args.submission_file is None:
            parser.error('без --train нужен --submission_file')
        predict(args.data_dir, args.model_file, args.submission_file)

    print(f'finished, elapsed = {timer() - start:.3f}')


if __name__ == '__main__':
    main()
