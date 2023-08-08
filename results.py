"""MTEB Results"""

import json

import datasets


logger = datasets.logging.get_logger(__name__)


_CITATION = """@article{muennighoff2022mteb,
  doi = {10.48550/ARXIV.2210.07316},
  url = {https://arxiv.org/abs/2210.07316},
  author = {Muennighoff, Niklas and Tazi, Nouamane and Magne, Lo{\"\i}c and Reimers, Nils},
  title = {MTEB: Massive Text Embedding Benchmark},
  publisher = {arXiv},
  journal={arXiv preprint arXiv:2210.07316},  
  year = {2022}
}
"""

_DESCRIPTION = """Results on MTEB"""

URL = "https://huggingface.co/datasets/mteb/results/resolve/main/paths.json"
VERSION = datasets.Version("1.0.1")
EVAL_LANGS = ['af', 'afr-eng', 'am', 'amh-eng', 'ang-eng', 'ar', 'ar-ar', 'ara-eng', 'arq-eng', 'arz-eng', 'ast-eng', 'awa-eng', 'az', 'aze-eng', 'bel-eng', 'ben-eng', 'ber-eng', 'bn', 'bos-eng', 'bre-eng', 'bul-eng', 'cat-eng', 'cbk-eng', 'ceb-eng', 'ces-eng', 'cha-eng', 'cmn-eng', 'cor-eng', 'csb-eng', 'cy', 'cym-eng', 'da', 'dan-eng', 'de', 'de-fr', 'de-pl', 'deu-eng', 'dsb-eng', 'dtp-eng', 'el', 'ell-eng', 'en', 'en-ar', 'en-de', 'en-en', 'en-tr', 'epo-eng', 'es', 'es-en', 'es-es', 'es-it', 'est-eng', 'eus-eng', 'fa', 'fao-eng', 'fi', 'fin-eng', 'fr', 'fr-en', 'fr-pl', 'fra-eng', 'fry-eng', 'gla-eng', 'gle-eng', 'glg-eng', 'gsw-eng', 'he', 'heb-eng', 'hi', 'hin-eng', 'hrv-eng', 'hsb-eng', 'hu', 'hun-eng', 'hy', 'hye-eng', 'id', 'ido-eng', 'ile-eng', 'ina-eng', 'ind-eng', 'is', 'isl-eng', 'it', 'it-en', 'ita-eng', 'ja', 'jav-eng', 'jpn-eng', 'jv', 'ka', 'kab-eng', 'kat-eng', 'kaz-eng', 'khm-eng', 'km', 'kn', 'ko', 'ko-ko', 'kor-eng', 'kur-eng', 'kzj-eng', 'lat-eng', 'lfn-eng', 'lit-eng', 'lv', 'lvs-eng', 'mal-eng', 'mar-eng', 'max-eng', 'mhr-eng', 'mkd-eng', 'ml', 'mn', 'mon-eng', 'ms', 'my', 'nb', 'nds-eng', 'nl', 'nl-ende-en', 'nld-eng', 'nno-eng', 'nob-eng', 'nov-eng', 'oci-eng', 'orv-eng', 'pam-eng', 'pes-eng', 'pl', 'pl-en', 'pms-eng', 'pol-eng', 'por-eng', 'pt', 'ro', 'ron-eng', 'ru', 'rus-eng', 'sl', 'slk-eng', 'slv-eng', 'spa-eng', 'sq', 'sqi-eng', 'srp-eng', 'sv', 'sw', 'swe-eng', 'swg-eng', 'swh-eng', 'ta', 'tam-eng', 'tat-eng', 'te', 'tel-eng', 'tgl-eng', 'th', 'tha-eng', 'tl', 'tr', 'tuk-eng', 'tur-eng', 'tzl-eng', 'uig-eng', 'ukr-eng', 'ur', 'urd-eng', 'uzb-eng', 'vi', 'vie-eng', 'war-eng', 'wuu-eng', 'xho-eng', 'yid-eng', 'yue-eng', 'zh', 'zh-CN', 'zh-TW', 'zh-en', 'zsm-eng']

SKIP_KEYS = ["std", "evaluation_time", "main_score", "threshold"]

MODELS = [
    "all-MiniLM-L12-v2",
    "all-MiniLM-L6-v2",
    "all-mpnet-base-v2",
    "allenai-specter",
    "bert-base-swedish-cased",
    "bert-base-uncased",
    "bge-base-zh",
    "bge-large-zh",
    "bge-large-zh-noinstruct",
    "bge-small-zh",
    "contriever-base-msmarco",
    "cross-en-de-roberta-sentence-transformer",
    "dfm-encoder-large-v1",
    "dfm-sentence-encoder-large-1",
    "distiluse-base-multilingual-cased-v2",
    "DanskBERT",
    "e5-base",
    "e5-large",
    "e5-small",
    "electra-small-nordic",
    "electra-small-swedish-cased-discriminator",
    "gbert-base",
    "gbert-large",
    "gelectra-base",
    "gelectra-large",
    "gottbert-base",
    "glove.6B.300d",
    "gtr-t5-base",
    "gtr-t5-large",
    "gtr-t5-xl",
    "gtr-t5-xxl",
    "komninos",
    "luotuo-bert-medium",
    "LaBSE",
    "LASER2",
    "msmarco-bert-co-condensor",
    "m3e-base",
    "m3e-large",
    "multilingual-e5-base",
    "multilingual-e5-large",
    "multilingual-e5-small",
    "nb-bert-base",
    "nb-bert-large",
    "norbert3-base",
    "norbert3-large",
    "paraphrase-multilingual-MiniLM-L12-v2",
    "paraphrase-multilingual-mpnet-base-v2",
    "sentence-bert-swedish-cased",
    "sentence-t5-base",
    "sentence-t5-large",
    "sentence-t5-xl",
    "sentence-t5-xxl",
    "sgpt-bloom-1b7-nli",
    "sgpt-bloom-7b1-msmarco",
    "sup-simcse-bert-base-uncased",
    "text2vec-base-chinese",
    "text2vec-large-chinese"
    "text-embedding-ada-002",
    "text-similarity-ada-001",
    "text-similarity-babbage-001",
    "text-similarity-curie-001",
    "text-similarity-davinci-001",
    "text-search-ada-doc-001",
    "text-search-ada-001",
    "text-search-babbage-001",
    "text-search-curie-001",
    "text-search-davinci-001",
    "unsup-simcse-bert-base-uncased",
    "use-cmlm-multilingual",
    "xlm-roberta-base",
    "xlm-roberta-large",
]


# Needs to be run whenever new files are added
def get_paths():
    import collections, json, os
    files = collections.defaultdict(list)
    for model_dir in os.listdir("results"):
        results_model_dir = os.path.join("results", model_dir)
        if not os.path.isdir(results_model_dir):
            print(f"Skipping {results_model_dir}")
            continue
        for res_file in os.listdir(results_model_dir):
            if res_file.endswith(".json"):
                results_model_file = os.path.join(results_model_dir, res_file)
                files[model_dir].append(results_model_file)
    with open("paths.json", "w") as f:
        json.dump(files, f)
    return files


class MTEBResults(datasets.GeneratorBasedBuilder):
    """MTEBResults"""

    BUILDER_CONFIGS = [
        datasets.BuilderConfig(
            name=model,
            description=f"{model} MTEB results",
            version=VERSION,
        )
        for model in MODELS
    ]

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "mteb_dataset_name": datasets.Value("string"),
                    "eval_language": datasets.Value("string"),
                    "metric": datasets.Value("string"),
                    "score": datasets.Value("float"),
                }
            ),
            supervised_keys=None,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        path_file = dl_manager.download_and_extract(URL)
        with open(path_file) as f:
            files = json.load(f)

        downloaded_files = dl_manager.download_and_extract(files[self.config.name])
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={'filepath': downloaded_files}
            )
        ]

    def _generate_examples(self, filepath):
        """This function returns the examples in the raw (text) form."""
        logger.info(f"Generating examples from {filepath}")

        out = []

        for path in filepath:
            with open(path, encoding="utf-8") as f:
                res_dict = json.load(f)
                ds_name = res_dict["mteb_dataset_name"]
                split = "test"
                if ds_name in ["AFQMC", "CmedqaRetrieval", "Cmnli", "CovidRetrieval", "DuRetrieval", "EcomRetrieval", "IFlyTek", "JDReview", "MedicalRetrieval", "MultilingualSentiment", "MMarcoReranking", "MMarcoRetrieval", "MSMARCO", "Ocnli", "OnlineShopping", "T2Reranking", "T2Retrieval", "TNews", "VideoRetrieval",]:
                    split = "dev" if "dev" in res_dict else "validation"
                if (split not in res_dict) and (ds_name in ["JDReview"]):
                    split = "test"
                if split not in res_dict:
                    print(f"Skipping {ds_name} as split {split} not present.")
                    continue
                res_dict = res_dict.get(split)
                is_multilingual = any(x in res_dict for x in EVAL_LANGS)
                langs = res_dict.keys() if is_multilingual else ["en"]
                for lang in langs:
                    if lang in SKIP_KEYS: continue
                    test_result_lang = res_dict.get(lang) if is_multilingual else res_dict
                    for metric, score in test_result_lang.items():
                        if not isinstance(score, dict):
                            score = {metric: score}
                        for sub_metric, sub_score in score.items():
                            if any(x in sub_metric for x in SKIP_KEYS): continue
                            out.append({
                                "mteb_dataset_name": ds_name,
                                "eval_language": lang if is_multilingual else "",
                                "metric": f"{metric}_{sub_metric}" if metric != sub_metric else metric,
                                "score": sub_score * 100,
                            })
        for idx, row in enumerate(sorted(out, key=lambda x: x["mteb_dataset_name"])):
            yield idx, row
