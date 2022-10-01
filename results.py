"""MTEB Results"""

import io
import json

import datasets


logger = datasets.logging.get_logger(__name__)


_CITATION = """
TODO
"""

_DESCRIPTION = """\
Results on MTEB
"""

URL = "https://huggingface.co/datasets/mteb/results/resolve/main/paths.json"
VERSION = datasets.Version("1.0.0")
EVAL_LANGS = [
    "en", 
    "de", 
    "es", 
    "fr", 
    "hi", 
    "th",
    "af",
    "am",
    "ar",
    "az",
    "bn",
    "cy",
    "da",
    "de",
    "el",
    "en",
    "es",
    "fa",
    "fi",
    "fr",
    "he",
    "hi",
    "hu",
    "hy",
    "id",
    "is",
    "it",
    "ja",
    "jv",
    "ka",
    "km",
    "kn",
    "ko",
    "lv",
    "ml",
    "mn",
    "ms",
    "my",
    "nb",
    "nl",
    "pl",
    "pt",
    "ro",
    "ru",
    "sl",
    "sq",
    "sv",
    "sw",
    "ta",
    "te",
    "th",
    "tl",
    "tr",
    "ur",
    "vi",
    "zh-CN",
    "zh-TW",
    "ko-ko", "ar-ar", "en-ar", "en-de", "en-en", "en-tr", "es-en", "es-es", "fr-en", "it-en", "nl-en"
    "de-en",
    "es-en",
    "it",
    "pl-en",
    "zh-en",
    "es-it",
    "de-fr",
    "de-pl",
    "fr-pl",
]
SKIP_KEYS = ["std", "evaluation_time", "main_score", "threshold"]

MODELS = [
    "LASER2",
    "LaBSE",
]
"""    
README.md
SGPT-1.3B-weightedmean-msmarco-specb-bitfit
SGPT-125M-weightedmean-msmarco-specb-bitfit
SGPT-125M-weightedmean-msmarco-specb-bitfit-doc
SGPT-125M-weightedmean-msmarco-specb-bitfit-que
SGPT-125M-weightedmean-nli-bitfit
SGPT-2.7B-weightedmean-msmarco-specb-bitfit
SGPT-5.8B-weightedmean-msmarco-specb-bitfit
SGPT-5.8B-weightedmean-msmarco-specb-bitfit-que
SGPT-5.8B-weightedmean-nli-bitfit
all-MiniLM-L12-v2
all-MiniLM-L6-v2
all-mpnet-base-v2
allenai-specter
bert-base-uncased
contriever-base-msmarco
glove.6B.300d
gtr-t5-base
gtr-t5-large
gtr-t5-xl
gtr-t5-xxl
komninos
msmarco-bert-co-condensor
paraphrase-multilingual-MiniLM-L12-v2
paraphrase-multilingual-mpnet-base-v2
results.py
sentence-t5-base
sentence-t5-large
sentence-t5-xl
sentence-t5-xxl
sgpt-bloom-1b3-nli
sgpt-bloom-7b1-msmarco
sgpt-nli-bloom-1b3
sup-simcse-bert-base-uncased
text-similarity-ada-001
unsup-simcse-bert-base-uncased
"""

# Needs to be run whenever new files are added
def get_paths():
    import json, os
    files = {}
    for model_dir in os.listdir("results"):
        results_model_dir = os.path.join("results", model_dir)
        for res_file in os.listdir(results_model_dir):
            if res_file.endswith(".json"):
                results_model_file = os.path.join(results_model_dir, res_file)
                files.setdefault(model_dir, [])            
                files[model_dir].append(results_model_file)
    with open(f"paths.json", "w") as f:
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
                    "dataset": datasets.Value("string"),
                    "metric": datasets.Value("string"),
                    "score": datasets.Value("float"),
                }
            ),
            supervised_keys=None,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        path_file = dl_manager.download_and_extract(URL)
        with open(path_file, "r") as f:
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
        logger.info("Generating examples from {}".format(filepath))
        
        out = []

        for path in filepath:
            ds_name = path.split("/")[-1]
            with io.open(path, "r", encoding="utf-8") as f:
                res_dict = json.load(f)
                split = "test"
                if ds_name == "MSMARCO":
                    split = "dev" if "dev" in res_dict else "validation"
                if split not in res_dict:
                    print(f"Skipping {ds_name} as split {split} not present.")
                    continue
                res_dict = res_dict.get(split)
                is_multilingual = True if any([x in res_dict for x in EVAL_LANGS]) else False
                langs = res_dict.keys() if is_multilingual else ["en"]
                for lang in langs:
                    if lang in SKIP_KEYS: continue
                    test_result_lang = res_dict.get(lang) if is_multilingual else res_dict
                    for (metric, score) in test_result_lang.items():
                        if not isinstance(score, dict):
                            score = {metric: score}
                        for sub_metric, sub_score in score.items():
                            if any([x in sub_metric for x in SKIP_KEYS]): continue
                            print("GOT", filepath, ds_name, res_dict, sub_score)
                            out.append({
                                "dataset": ds_name,
                                "metric": f"{metric}_{sub_metric}" if metric != sub_metric else metric,
                                "score": sub_score * 100,
                            })
        for idx, row in enumerate(out):
            yield idx, row
