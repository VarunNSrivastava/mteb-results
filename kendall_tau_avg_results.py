import json
import shutil

import pandas as pd
from datasets import load_dataset
from huggingface_hub import get_hf_file_metadata, HfApi, hf_hub_download, hf_hub_url
from huggingface_hub.repocard import metadata_load
from huggingface_hub.utils._errors import GatedRepoError

from dataset_list import TASK_LIST_EN, TASK_LIST_CLASSIFICATION, \
    TASK_LIST_PAIR_CLASSIFICATION, TASK_LIST_SUMMARIZATION, TASK_LIST_CLUSTERING, \
    TASK_LIST_RERANKING, TASK_LIST_RETRIEVAL, TASK_LIST_STS, add_task
from model_list import EXTERNAL_MODELS, EXTERNAL_MODEL_TO_LINK, EXTERNAL_MODEL_TO_SIZE, EXTERNAL_MODEL_TO_SEQLEN, \
    EXTERNAL_MODEL_TO_DIM, MODELS_TO_SKIP


def get_dim_seq_size(model):
    filenames = [sib.rfilename for sib in model.siblings]
    dim, seq, size = "", "", ""
    if "1_Pooling/config.json" in filenames:
        st_config_path = hf_hub_download(model.modelId, filename="1_Pooling/config.json")
        dim = json.load(open(st_config_path)).get("word_embedding_dimension", "")
    elif "2_Pooling/config.json" in filenames:
        st_config_path = hf_hub_download(model.modelId, filename="2_Pooling/config.json")
        dim = json.load(open(st_config_path)).get("word_embedding_dimension", "")
    if "config.json" in filenames:
        config_path = hf_hub_download(model.modelId, filename="config.json")
        config = json.load(open(config_path))
        if not dim:
            dim = config.get("hidden_dim", config.get("hidden_size", config.get("d_model", "")))
        seq = config.get("n_positions",
                         config.get("max_position_embeddings", config.get("n_ctx", config.get("seq_length", ""))))
    # Get model file size without downloading
    if "pytorch_model.bin" in filenames:
        url = hf_hub_url(model.modelId, filename="pytorch_model.bin")
        meta = get_hf_file_metadata(url)
        size = round(meta.size / 1e9, 2)
    elif "pytorch_model.bin.index.json" in filenames:
        index_path = hf_hub_download(model.modelId, filename="pytorch_model.bin.index.json")
        """
        {
        "metadata": {
            "total_size": 28272820224
        },....
        """
        size = json.load(open(index_path))
        if ("metadata" in size) and ("total_size" in size["metadata"]):
            size = round(size["metadata"]["total_size"] / 1e9, 2)
    return dim, seq, size


TASK_TO_METRIC = {
    "BitextMining": "f1",
    "Clustering": "v_measure",
    "Classification": "accuracy",
    "PairClassification": "cos_sim_ap",
    "Reranking": "map",
    "Retrieval": "ndcg_at_10",
    "STS": "cos_sim_spearman",
    "Summarization": "cos_sim_spearman",
}

EXTERNAL_MODEL_RESULTS = {model: {k: {v: []} for k, v in TASK_TO_METRIC.items()} for model in EXTERNAL_MODELS}


def make_clickable_model(model_name, link=None):
    if link is None:
        link = "https://huggingface.co/" + model_name
    # Remove user from model name
    return (
        f'<a target="_blank" style="text-decoration: underline" href="{link}">{model_name.split("/")[-1]}</a>'
    )


def add_lang(examples):
    if not (examples["eval_language"]):
        examples["mteb_dataset_name_with_lang"] = examples["mteb_dataset_name"]
    else:
        examples["mteb_dataset_name_with_lang"] = examples["mteb_dataset_name"] + f' ({examples["eval_language"]})'
    return examples


def add_rank(df):
    cols_to_rank = [col for col in df.columns if
                    col not in ["Model", "Model Size (GB)", "Embedding Dimensions", "Sequence Length"]]
    if len(cols_to_rank) == 1:
        df.sort_values(cols_to_rank[0], ascending=False, inplace=True)
    else:
        df.insert(1, "Average", df[cols_to_rank].mean(axis=1, skipna=False))
        df.sort_values("Average", ascending=False, inplace=True)
    df.insert(0, "Rank", list(range(1, len(df) + 1)))
    df = df.round(2)
    # Fill NaN after averaging
    df.fillna("", inplace=True)
    return df


for model in EXTERNAL_MODELS:
    ds = load_dataset("mteb/results", model)
    # For local debugging:
    # , download_mode='force_redownload', verification_mode="no_checks")
    ds = ds.map(add_lang)
    ds = ds.map(add_task)
    base_dict = {"Model": make_clickable_model(model, link=EXTERNAL_MODEL_TO_LINK.get(model,
                                                                                      "https://huggingface.co/spaces/mteb/leaderboard"))}
    # For now only one metric per task - Could add more metrics later on
    for task, metric in TASK_TO_METRIC.items():
        ds_dict = ds.filter(lambda x: (x["mteb_task"] == task) and (x["metric"] == metric))["test"].to_dict()
        ds_dict = {k: round(v, 2) for k, v in zip(ds_dict["mteb_dataset_name_with_lang"], ds_dict["score"])}
        EXTERNAL_MODEL_RESULTS[model][task][metric].append({**base_dict, **ds_dict})


def get_mteb_data(tasks=["Clustering"], langs=[], datasets=[], fillna=True, add_emb_dim=False,
                  task_to_metric=TASK_TO_METRIC, rank=True):
    api = HfApi()
    models = api.list_models(filter="mteb")
    # Initialize list to models that we cannot fetch metadata from
    df_list = []
    for model in EXTERNAL_MODEL_RESULTS:
        results_list = [res for task in tasks for res in EXTERNAL_MODEL_RESULTS[model][task][task_to_metric[task]]]
        if len(datasets) > 0:
            res = {k: v for d in results_list for k, v in d.items() if
                   (k == "Model") or any([x in k for x in datasets])}
        elif langs:
            # Would be cleaner to rely on an extra language column instead
            langs_format = [f"({lang})" for lang in langs]
            res = {k: v for d in results_list for k, v in d.items() if
                   any([k.split(" ")[-1] in (k, x) for x in langs_format])}
        else:
            res = {k: v for d in results_list for k, v in d.items()}
        # Model & at least one result
        if len(res) > 1:
            if add_emb_dim:
                res["Model Size (GB)"] = EXTERNAL_MODEL_TO_SIZE.get(model, "")
                res["Embedding Dimensions"] = EXTERNAL_MODEL_TO_DIM.get(model, "")
                res["Sequence Length"] = EXTERNAL_MODEL_TO_SEQLEN.get(model, "")
            df_list.append(res)

    for model in models:
        if model.modelId in MODELS_TO_SKIP: continue
        try:
            readme_path = hf_hub_download(model.modelId, filename="README.md")
            meta = metadata_load(readme_path)
        except GatedRepoError as e:
            print(f"Ignoring model {model} because they are perverts.")
            continue
        # meta['model-index'][0]["results"] is list of elements like:
        # {
        #    "task": {"type": "Classification"},
        #    "dataset": {
        #        "type": "mteb/amazon_massive_intent",
        #        "name": "MTEB MassiveIntentClassification (nb)",
        #        "config": "nb",
        #        "split": "test",
        #    },
        #    "metrics": [
        #        {"type": "accuracy", "value": 39.81506388702084},
        #        {"type": "f1", "value": 38.809586587791664},
        #    ],
        # },
        # Use "get" instead of dict indexing to skip incompat metadata instead of erroring out
        if len(datasets) > 0:
            task_results = [sub_res for sub_res in meta["model-index"][0]["results"] if
                            (sub_res.get("task", {}).get("type", "") in tasks) and any(
                                [x in sub_res.get("dataset", {}).get("name", "") for x in datasets])]
        elif langs:
            task_results = [sub_res for sub_res in meta["model-index"][0]["results"] if
                            (sub_res.get("task", {}).get("type", "") in tasks) and (
                                    sub_res.get("dataset", {}).get("config", "default") in ("default", *langs))]
        else:
            task_results = [sub_res for sub_res in meta["model-index"][0]["results"] if
                            (sub_res.get("task", {}).get("type", "") in tasks)]
        out = [{res["dataset"]["name"].replace("MTEB ", ""): [round(score["value"], 2) for score in res["metrics"] if
                                                              score["type"] == task_to_metric.get(res["task"]["type"])][
            0]} for res in task_results]
        out = {k: v for d in out for k, v in d.items()}
        out["Model"] = make_clickable_model(model.modelId)
        # Model & at least one result
        if len(out) > 1:
            if add_emb_dim:
                out["Embedding Dimensions"], out["Sequence Length"], out["Model Size (GB)"] = get_dim_seq_size(model)
            df_list.append(out)
    df = pd.DataFrame(df_list)
    # If there are any models that are the same, merge them
    # E.g. if out["Model"] has the same value in two places, merge & take whichever one is not NaN else just take the first one
    df = df.groupby("Model", as_index=False).first()
    # Put 'Model' column first
    cols = sorted(list(df.columns))
    cols.insert(0, cols.pop(cols.index("Model")))
    df = df[cols]
    if rank:
        df = add_rank(df)
    if fillna:
        df.fillna("", inplace=True)
    return df


def get_mteb_average():
    global DATA_OVERALL, DATA_CLASSIFICATION_EN, DATA_CLUSTERING, DATA_PAIR_CLASSIFICATION, DATA_RERANKING, \
        DATA_RETRIEVAL, DATA_STS_EN, DATA_SUMMARIZATION, COLUMNS
    DATA_OVERALL = get_mteb_data(
        tasks=[
            "Classification",
            "Clustering",
            "PairClassification",
            "Reranking",
            "Retrieval",
            "STS",
            "Summarization",
        ],
        datasets=TASK_LIST_EN,
        fillna=False,
        add_emb_dim=True,
        rank=False,
    )
    # Debugging:
    # DATA_OVERALL.to_csv("overall.csv")

    DATA_OVERALL.insert(1, f"Average ({len(TASK_LIST_EN)} datasets)",
                        DATA_OVERALL[TASK_LIST_EN].mean(axis=1, skipna=False))
    DATA_OVERALL.insert(2, f"Classification Average ({len(TASK_LIST_CLASSIFICATION)} datasets)",
                        DATA_OVERALL[TASK_LIST_CLASSIFICATION].mean(axis=1, skipna=False))
    DATA_OVERALL.insert(3, f"Clustering Average ({len(TASK_LIST_CLUSTERING)} datasets)",
                        DATA_OVERALL[TASK_LIST_CLUSTERING].mean(axis=1, skipna=False))
    DATA_OVERALL.insert(4, f"Pair Classification Average ({len(TASK_LIST_PAIR_CLASSIFICATION)} datasets)",
                        DATA_OVERALL[TASK_LIST_PAIR_CLASSIFICATION].mean(axis=1, skipna=False))
    DATA_OVERALL.insert(5, f"Reranking Average ({len(TASK_LIST_RERANKING)} datasets)",
                        DATA_OVERALL[TASK_LIST_RERANKING].mean(axis=1, skipna=False))
    DATA_OVERALL.insert(6, f"Retrieval Average ({len(TASK_LIST_RETRIEVAL)} datasets)",
                        DATA_OVERALL[TASK_LIST_RETRIEVAL].mean(axis=1, skipna=False))
    DATA_OVERALL.insert(7, f"STS Average ({len(TASK_LIST_STS)} datasets)",
                        DATA_OVERALL[TASK_LIST_STS].mean(axis=1, skipna=False))
    DATA_OVERALL.insert(8, f"Summarization Average ({len(TASK_LIST_SUMMARIZATION)} dataset)",
                        DATA_OVERALL[TASK_LIST_SUMMARIZATION].mean(axis=1, skipna=False))
    DATA_OVERALL.sort_values(f"Average ({len(TASK_LIST_EN)} datasets)", ascending=False, inplace=True)
    # Start ranking from 1
    DATA_OVERALL.insert(0, "Rank", list(range(1, len(DATA_OVERALL) + 1)))

    DATA_OVERALL = DATA_OVERALL.round(2)

    DATA_CLASSIFICATION_EN = add_rank(DATA_OVERALL[["Model"] + TASK_LIST_CLASSIFICATION])
    # Only keep rows with at least one score in addition to the "Model" & rank column
    DATA_CLASSIFICATION_EN = DATA_CLASSIFICATION_EN[DATA_CLASSIFICATION_EN.iloc[:, 2:].ne("").any(axis=1)]

    DATA_CLUSTERING = add_rank(DATA_OVERALL[["Model"] + TASK_LIST_CLUSTERING])
    DATA_CLUSTERING = DATA_CLUSTERING[DATA_CLUSTERING.iloc[:, 2:].ne("").any(axis=1)]

    DATA_PAIR_CLASSIFICATION = add_rank(DATA_OVERALL[["Model"] + TASK_LIST_PAIR_CLASSIFICATION])
    DATA_PAIR_CLASSIFICATION = DATA_PAIR_CLASSIFICATION[DATA_PAIR_CLASSIFICATION.iloc[:, 2:].ne("").any(axis=1)]

    DATA_RERANKING = add_rank(DATA_OVERALL[["Model"] + TASK_LIST_RERANKING])
    DATA_RERANKING = DATA_RERANKING[DATA_RERANKING.iloc[:, 2:].ne("").any(axis=1)]

    DATA_RETRIEVAL = add_rank(DATA_OVERALL[["Model"] + TASK_LIST_RETRIEVAL])
    DATA_RETRIEVAL = DATA_RETRIEVAL[DATA_RETRIEVAL.iloc[:, 2:].ne("").any(axis=1)]

    DATA_STS_EN = add_rank(DATA_OVERALL[["Model"] + TASK_LIST_STS])
    DATA_STS_EN = DATA_STS_EN[DATA_STS_EN.iloc[:, 2:].ne("").any(axis=1)]

    DATA_SUMMARIZATION = add_rank(DATA_OVERALL[["Model"] + TASK_LIST_SUMMARIZATION])
    DATA_SUMMARIZATION = DATA_SUMMARIZATION[DATA_SUMMARIZATION.iloc[:, 1:].ne("").any(axis=1)]

    # Fill NaN after averaging
    DATA_OVERALL.fillna("", inplace=True)

    COLUMNS = ["Rank", "Model", "Model Size (GB)", "Embedding Dimensions", "Sequence Length",
               f"Average ({len(TASK_LIST_EN)} datasets)",
               f"Classification Average ({len(TASK_LIST_CLASSIFICATION)} datasets)",
               f"Clustering Average ({len(TASK_LIST_CLUSTERING)} datasets)",
               f"Pair Classification Average ({len(TASK_LIST_PAIR_CLASSIFICATION)} datasets)",
               f"Reranking Average ({len(TASK_LIST_RERANKING)} datasets)",
               f"Retrieval Average ({len(TASK_LIST_RETRIEVAL)} datasets)",
               f"STS Average ({len(TASK_LIST_STS)} datasets)",
               f"Summarization Average ({len(TASK_LIST_SUMMARIZATION)} dataset)"]

    DATA_OVERALL = DATA_OVERALL[COLUMNS]
    DATA_OVERALL = DATA_OVERALL[DATA_OVERALL.iloc[:, 5:].ne("").any(axis=1)]

    return COLUMNS, DATA_OVERALL


COLUMNS, DATA_OVERALL = get_mteb_average()


def print_line(character='-'):
    columns, _ = shutil.get_terminal_size((80, 20))
    print(character * columns)


def to_float(val):
    try:
        return float(val)
    except:
        return float('nan')  # Convert problematic values to NaN


### Drop NAs


for col in COLUMNS[6:]:
    DATA_OVERALL[col] = DATA_OVERALL[col].apply(to_float)

DATA_OVERALL.dropna(inplace=True)

DATA_OVERALL['Average tasks'] = DATA_OVERALL[COLUMNS[6:]].mean(axis=1)

DATA_OVERALL.insert(2, 'Average tasks', DATA_OVERALL.pop('Average tasks'))

print_line()


def kendall_tau(list1, list2):
    n = len(list1)
    assert len(list1) == len(list2), "Lists must be of the same length"
    concordant = 0
    discordant = 0
    for i in range(n):
        for j in range(i + 1, n):
            list1_order = (list1[i] - list1[j])
            list2_order = (list2[i] - list2[j])
            if (list1_order * list2_order) > 0:
                concordant += 1
            elif (list1_order * list2_order) < 0:
                discordant += 1
    return (concordant - discordant) / (concordant + discordant)


def rank_models_based_on_column(data, column_name):
    return data.sort_values(by=column_name, ascending=False).index.tolist()


def get_most_similar_task_to_average(data, tasks):
    avg_ranking = rank_models_based_on_column(data, "Average tasks")
    """
    For a weighted average, use
        avg_ranking = rank_models_based_on_column(data, f"Average ({len(TASK_LIST_EN)} datasets)")
    instead.
    """

    max_tau = -2
    most_similar_task = None
    task_tau_scores = {}

    for task in tasks:
        task_ranking = rank_models_based_on_column(data, task)

        tau = kendall_tau(avg_ranking, task_ranking)
        task_tau_scores[task] = tau

        if tau > max_tau:
            max_tau = tau
            most_similar_task = task

    # 4. Display table
    print('-' * 80)  # Print a separator line
    print("{:<50} | {:<15}".format("Task", "Kendall Tau"))  # Table header
    print('-' * 80)  # Print a separator line
    for task, tau_score in sorted(task_tau_scores.items(), key=lambda x: x[1], reverse=True):  # Sorting by tau_score
        print("{:<50} | {:<15.4f}".format(task, tau_score))
    print('-' * 80)  # Print a separator line

    return most_similar_task, max_tau


most_similar_task, tau_value = get_most_similar_task_to_average(DATA_OVERALL, COLUMNS[6:])
print(
    f"\nThe task most similar to the average ranking is '{most_similar_task}' with a Kendall's Tau of {tau_value:.4f}.")
