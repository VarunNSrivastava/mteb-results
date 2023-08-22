EXTERNAL_MODELS = [
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
    "LASER2",
    "LaBSE",
    "m3e-base",
    "m3e-large",
    "msmarco-bert-co-condensor",
    "multilingual-e5-base",
    "multilingual-e5-large",
    "multilingual-e5-small",
    "nb-bert-base",
    "nb-bert-large",
    "norbert3-base",
    "norbert3-large",
    "paraphrase-multilingual-MiniLM-L12-v2",
    "paraphrase-multilingual-mpnet-base-v2",
    # "sentence-bert-swedish-cased",
    "sentence-t5-base",
    "sentence-t5-large",
    "sentence-t5-xl",
    "sentence-t5-xxl",
    "sup-simcse-bert-base-uncased",
    "st-polish-paraphrase-from-distilroberta",
    # "st-polish-paraphrase-from-mpnet",
    # "text2vec-base-chinese",
    # "text2vec-large-chinese",
    "text-embedding-ada-002",
    "text-similarity-ada-001",
    # "text-similarity-babbage-001",
    # "text-similarity-curie-001",
    # "text-similarity-davinci-001"
    # "text-search-ada-doc-001",
    # "text-search-ada-001",
    # "text-search-babbage-001",
    # "text-search-curie-001",
    # "text-search-davinci-001",
    "unsup-simcse-bert-base-uncased",
    "use-cmlm-multilingual",
    "xlm-roberta-base",
    "xlm-roberta-large",
]

EXTERNAL_MODEL_TO_LINK = {
    "allenai-specter": "https://huggingface.co/sentence-transformers/allenai-specter",
    "allenai-specter": "https://huggingface.co/sentence-transformers/allenai-specter",
    "all-MiniLM-L12-v2": "https://huggingface.co/sentence-transformers/all-MiniLM-L12-v2",
    "all-MiniLM-L6-v2": "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2",
    "all-mpnet-base-v2": "https://huggingface.co/sentence-transformers/all-mpnet-base-v2",
    "bert-base-swedish-cased": "https://huggingface.co/KB/bert-base-swedish-cased",
    "bert-base-uncased": "https://huggingface.co/bert-base-uncased",
    # "bge-base-zh": "https://huggingface.co/BAAI/bge-base-zh",
    # "bge-large-zh": "https://huggingface.co/BAAI/bge-large-zh",
    # "bge-large-zh-noinstruct": "https://huggingface.co/BAAI/bge-large-zh-noinstruct",
    # "bge-small-zh": "https://huggingface.co/BAAI/bge-small-zh",
    "contriever-base-msmarco": "https://huggingface.co/nthakur/contriever-base-msmarco",
    "cross-en-de-roberta-sentence-transformer": "https://huggingface.co/T-Systems-onsite/cross-en-de-roberta-sentence-transformer",
    "DanskBERT": "https://huggingface.co/vesteinn/DanskBERT",
    "distiluse-base-multilingual-cased-v2": "https://huggingface.co/sentence-transformers/distiluse-base-multilingual-cased-v2",
    "dfm-encoder-large-v1": "https://huggingface.co/chcaa/dfm-encoder-large-v1",
    "dfm-sentence-encoder-large-1": "https://huggingface.co/chcaa/dfm-encoder-large-v1",
    "e5-base": "https://huggingface.co/intfloat/e5-base",
    "e5-large": "https://huggingface.co/intfloat/e5-large",
    "e5-small": "https://huggingface.co/intfloat/e5-small",
    "electra-small-nordic": "https://huggingface.co/jonfd/electra-small-nordic",
    "electra-small-swedish-cased-discriminator": "https://huggingface.co/KBLab/electra-small-swedish-cased-discriminator",
    "gbert-base": "https://huggingface.co/deepset/gbert-base",
    "gbert-large": "https://huggingface.co/deepset/gbert-large",
    "gelectra-base": "https://huggingface.co/deepset/gelectra-base",
    "gelectra-large": "https://huggingface.co/deepset/gelectra-large",
    "glove.6B.300d": "https://huggingface.co/sentence-transformers/average_word_embeddings_glove.6B.300d",
    "gottbert-base": "https://huggingface.co/uklfr/gottbert-base",
    "gtr-t5-base": "https://huggingface.co/sentence-transformers/gtr-t5-base",
    "gtr-t5-large": "https://huggingface.co/sentence-transformers/gtr-t5-large",
    "gtr-t5-xl": "https://huggingface.co/sentence-transformers/gtr-t5-xl",
    "gtr-t5-xxl": "https://huggingface.co/sentence-transformers/gtr-t5-xxl",
    "komninos": "https://huggingface.co/sentence-transformers/average_word_embeddings_komninos",
    "luotuo-bert-medium": "https://huggingface.co/silk-road/luotuo-bert-medium",
    "LASER2": "https://github.com/facebookresearch/LASER",
    "LaBSE": "https://huggingface.co/sentence-transformers/LaBSE",
    "m3e-base": "https://huggingface.co/moka-ai/m3e-base",
    "m3e-large": "https://huggingface.co/moka-ai/m3e-large",
    "msmarco-bert-co-condensor": "https://huggingface.co/sentence-transformers/msmarco-bert-co-condensor",
    "multilingual-e5-base": "https://huggingface.co/intfloat/multilingual-e5-base",
    "multilingual-e5-large": "https://huggingface.co/intfloat/multilingual-e5-large",
    "multilingual-e5-small": "https://huggingface.co/intfloat/multilingual-e5-small",
    "nb-bert-base": "https://huggingface.co/NbAiLab/nb-bert-base",
    "nb-bert-large": "https://huggingface.co/NbAiLab/nb-bert-large",
    "norbert3-base": "https://huggingface.co/ltg/norbert3-base",
    "norbert3-large": "https://huggingface.co/ltg/norbert3-large",
    "paraphrase-multilingual-mpnet-base-v2": "https://huggingface.co/sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
    "paraphrase-multilingual-MiniLM-L12-v2": "https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    "sentence-bert-swedish-cased": "https://huggingface.co/KBLab/sentence-bert-swedish-cased",
    "sentence-t5-base": "https://huggingface.co/sentence-transformers/sentence-t5-base",
    "sentence-t5-large": "https://huggingface.co/sentence-transformers/sentence-t5-large",
    "sentence-t5-xl": "https://huggingface.co/sentence-transformers/sentence-t5-xl",
    "sentence-t5-xxl": "https://huggingface.co/sentence-transformers/sentence-t5-xxl",
    "sup-simcse-bert-base-uncased": "https://huggingface.co/princeton-nlp/sup-simcse-bert-base-uncased",
    "st-polish-paraphrase-from-distilroberta": "https://huggingface.co/sdadas/st-polish-paraphrase-from-distilroberta",
    "st-polish-paraphrase-from-mpnet": "https://huggingface.co/sdadas/st-polish-paraphrase-from-mpnet",
    # "text2vec-base-chinese": "https://huggingface.co/shibing624/text2vec-base-chinese",
    # "text2vec-large-chinese": "https://huggingface.co/GanymedeNil/text2vec-large-chinese",
    "text-embedding-ada-002": "https://beta.openai.com/docs/guides/embeddings/types-of-embedding-models",
    "text-similarity-ada-001": "https://beta.openai.com/docs/guides/embeddings/types-of-embedding-models",
    "text-similarity-babbage-001": "https://beta.openai.com/docs/guides/embeddings/types-of-embedding-models",
    "text-similarity-curie-001": "https://beta.openai.com/docs/guides/embeddings/types-of-embedding-models",
    "text-similarity-davinci-001": "https://beta.openai.com/docs/guides/embeddings/types-of-embedding-models",
    "text-search-ada-doc-001": "https://beta.openai.com/docs/guides/embeddings/types-of-embedding-models",
    "text-search-ada-query-001": "https://beta.openai.com/docs/guides/embeddings/types-of-embedding-models",
    "text-search-ada-001": "https://beta.openai.com/docs/guides/embeddings/types-of-embedding-models",
    "text-search-curie-001": "https://beta.openai.com/docs/guides/embeddings/types-of-embedding-models",
    "text-search-babbage-001": "https://beta.openai.com/docs/guides/embeddings/types-of-embedding-models",
    "text-search-davinci-001": "https://beta.openai.com/docs/guides/embeddings/types-of-embedding-models",
    "unsup-simcse-bert-base-uncased": "https://huggingface.co/princeton-nlp/unsup-simcse-bert-base-uncased",
    "use-cmlm-multilingual": "https://huggingface.co/sentence-transformers/use-cmlm-multilingual",
    "xlm-roberta-base": "https://huggingface.co/xlm-roberta-base",
    "xlm-roberta-large": "https://huggingface.co/xlm-roberta-large",
}

EXTERNAL_MODEL_TO_DIM = {
    "all-MiniLM-L12-v2": 384,
    "all-MiniLM-L6-v2": 384,
    "all-mpnet-base-v2": 768,
    "allenai-specter": 768,
    "bert-base-swedish-cased": 768,
    "bert-base-uncased": 768,
    "bge-base-zh": 768,
    "bge-large-zh": 1024,
    "bge-large-zh-noinstruct": 1024,
    "bge-small-zh": 512,
    "contriever-base-msmarco": 768,
    "cross-en-de-roberta-sentence-transformer": 768,
    "DanskBERT": 768,
    "distiluse-base-multilingual-cased-v2": 512,
    "dfm-encoder-large-v1": 1024,
    "dfm-sentence-encoder-large-1": 1024,
    "e5-base": 768,
    "e5-small": 384,
    "e5-large": 1024,
    "electra-small-nordic": 256,
    "electra-small-swedish-cased-discriminator": 256,
    "luotuo-bert-medium": 768,
    "LASER2": 1024,
    "LaBSE": 768,
    "gbert-base": 768,
    "gbert-large": 1024,
    "gelectra-base": 768,
    "gelectra-large": 1024,
    "glove.6B.300d": 300,
    "gottbert-base": 768,
    "gtr-t5-base": 768,
    "gtr-t5-large": 768,
    "gtr-t5-xl": 768,
    "gtr-t5-xxl": 768,
    "komninos": 300,
    "m3e-base": 768,
    "m3e-large": 768,
    "msmarco-bert-co-condensor": 768,
    "multilingual-e5-base": 768,
    "multilingual-e5-small": 384,
    "multilingual-e5-large": 1024,
    "nb-bert-base": 768,
    "nb-bert-large": 1024,
    "norbert3-base": 768,
    "norbert3-large": 1024,
    "paraphrase-multilingual-MiniLM-L12-v2": 384,
    "paraphrase-multilingual-mpnet-base-v2": 768,
    "sentence-bert-swedish-cased": 768,
    "sentence-t5-base": 768,
    "sentence-t5-large": 768,
    "sentence-t5-xl": 768,
    "sentence-t5-xxl": 768,
    "sup-simcse-bert-base-uncased": 768,
    "st-polish-paraphrase-from-distilroberta": 768,
    "st-polish-paraphrase-from-mpnet": 768,
    "text2vec-base-chinese": 768,
    "text2vec-large-chinese": 1024,
    "text-embedding-ada-002": 1536,
    "text-similarity-ada-001": 1024,
    "text-similarity-babbage-001": 2048,
    "text-similarity-curie-001": 4096,
    "text-similarity-davinci-001": 12288,
    "text-search-ada-doc-001": 1024,
    "text-search-ada-query-001": 1024,
    "text-search-ada-001": 1024,
    "text-search-babbage-001": 2048,
    "text-search-curie-001": 4096,
    "text-search-davinci-001": 12288,
    "unsup-simcse-bert-base-uncased": 768,
    "use-cmlm-multilingual": 768,
    "xlm-roberta-base":  768,
    "xlm-roberta-large":  1024,
}

EXTERNAL_MODEL_TO_SEQLEN = {
    "all-MiniLM-L12-v2": 512,
    "all-MiniLM-L6-v2": 512,
    "all-mpnet-base-v2": 514,
    "allenai-specter": 512,
    "bert-base-swedish-cased": 512,
    "bert-base-uncased": 512,
    "bge-base-zh": 512,
    "bge-large-zh": 512,
    "bge-large-zh-noinstruct": 512,
    "bge-small-zh": 512,
    "contriever-base-msmarco": 512,
    "cross-en-de-roberta-sentence-transformer": 514,
    "DanskBERT": 514,
    "dfm-encoder-large-v1": 512,
    "dfm-sentence-encoder-large-1": 512,
    "distiluse-base-multilingual-cased-v2": 512,
    "e5-base": 512,
    "e5-large": 512,
    "e5-small": 512,
    "electra-small-nordic": 512,
    "electra-small-swedish-cased-discriminator": 512,
    "gbert-base": 512,
    "gbert-large": 512,
    "gelectra-base": 512,
    "gelectra-large": 512,
    "gottbert-base": 512,
    "glove.6B.300d": "N/A",
    "gtr-t5-base": 512,
    "gtr-t5-large": 512,
    "gtr-t5-xl": 512,
    "gtr-t5-xxl": 512,
    "komninos": "N/A",
    "luotuo-bert-medium": 512,
    "LASER2": "N/A",
    "LaBSE": 512,
    "m3e-base": 512,
    "m3e-large": 512,
    "msmarco-bert-co-condensor": 512,
    "multilingual-e5-base": 514,
    "multilingual-e5-large": 514,
    "multilingual-e5-small": 512,
    "nb-bert-base": 512,
    "nb-bert-large": 512,
    "norbert3-base": 512,
    "norbert3-large": 512,
    "paraphrase-multilingual-MiniLM-L12-v2": 512,
    "paraphrase-multilingual-mpnet-base-v2": 514,
    "sentence-bert-swedish-cased": 512,
    "sentence-t5-base": 512,
    "sentence-t5-large": 512,
    "sentence-t5-xl": 512,
    "sentence-t5-xxl": 512,
    "sup-simcse-bert-base-uncased": 512,
    "st-polish-paraphrase-from-distilroberta": 514,
    "st-polish-paraphrase-from-mpnet": 514,
    "text2vec-base-chinese": 512,
    "text2vec-large-chinese": 512,
    "text-embedding-ada-002": 8191,
    "text-similarity-ada-001": 2046,
    "text-similarity-babbage-001": 2046,
    "text-similarity-curie-001": 2046,
    "text-similarity-davinci-001": 2046,
    "text-search-ada-doc-001": 2046,
    "text-search-ada-query-001": 2046,
    "text-search-ada-001": 2046,
    "text-search-babbage-001": 2046,
    "text-search-curie-001": 2046,
    "text-search-davinci-001": 2046,
    "use-cmlm-multilingual": 512,
    "unsup-simcse-bert-base-uncased": 512,
    "xlm-roberta-base": 514,
    "xlm-roberta-large": 514,
}

EXTERNAL_MODEL_TO_SIZE = {
    "allenai-specter": 0.44,
    "all-MiniLM-L12-v2": 0.13,
    "all-MiniLM-L6-v2": 0.09,
    "all-mpnet-base-v2": 0.44,
    "bert-base-uncased": 0.44,
    "bert-base-swedish-cased": 0.50,
    "bge-base-zh": 0.41,
    "bge-large-zh": 1.30,
    "bge-large-zh-noinstruct": 1.30,
    "bge-small-zh": 0.10,
    "cross-en-de-roberta-sentence-transformer": 1.11,
    "contriever-base-msmarco": 0.44,
    "DanskBERT": 0.50,
    "distiluse-base-multilingual-cased-v2": 0.54,
    "dfm-encoder-large-v1": 1.42,
    "dfm-sentence-encoder-large-1": 1.63,
    "e5-base": 0.44,
    "e5-small": 0.13,
    "e5-large": 1.34,
    "electra-small-nordic": 0.09,
    "electra-small-swedish-cased-discriminator": 0.06,
    "gbert-base": 0.44,
    "gbert-large": 1.35,
    "gelectra-base": 0.44,
    "gelectra-large": 1.34,
    "glove.6B.300d": 0.48,
    "gottbert-base": 0.51,
    "gtr-t5-base": 0.22,
    "gtr-t5-large": 0.67,
    "gtr-t5-xl": 2.48,
    "gtr-t5-xxl": 9.73,
    "komninos": 0.27,
    "luotuo-bert-medium": 1.31,
    "LASER2": 0.17,
    "LaBSE": 1.88,
    "m3e-base": 0.41,
    "m3e-large": 0.41,
    "msmarco-bert-co-condensor": 0.44,
    "multilingual-e5-base": 1.11,
    "multilingual-e5-small": 0.47,
    "multilingual-e5-large": 2.24,
    "nb-bert-base": 0.71,
    "nb-bert-large": 1.42,
    "norbert3-base": 0.52,
    "norbert3-large": 1.47,
    "paraphrase-multilingual-mpnet-base-v2": 1.11,
    "paraphrase-multilingual-MiniLM-L12-v2": 0.47,
    "sentence-bert-swedish-cased": 0.50,
    "sentence-t5-base": 0.22,
    "sentence-t5-large": 0.67,
    "sentence-t5-xl": 2.48,
    "sentence-t5-xxl": 9.73,
    "sup-simcse-bert-base-uncased": 0.44,
    "st-polish-paraphrase-from-distilroberta": 0.50,
    "st-polish-paraphrase-from-mpnet": 0.50,
    "text2vec-base-chinese": 0.41,
    "text2vec-large-chinese": 1.30,
    "unsup-simcse-bert-base-uncased": 0.44,
    "use-cmlm-multilingual": 1.89,
    "xlm-roberta-base": 1.12,
    "xlm-roberta-large": 2.24,
}

MODELS_TO_SKIP = {
    "baseplate/instructor-large-1", # Duplicate
    "radames/e5-large", # Duplicate
    "gentlebowl/instructor-large-safetensors", # Duplicate
    "Consensus/instructor-base", # Duplicate
    "GovCompete/instructor-xl", # Duplicate
    "GovCompete/e5-large-v2", # Duplicate
    "t12e/instructor-base", # Duplicate
    "michaelfeil/ct2fast-e5-large-v2",
    "michaelfeil/ct2fast-e5-large",
    "michaelfeil/ct2fast-e5-small-v2",
    "newsrx/instructor-xl-newsrx",
    "newsrx/instructor-large-newsrx",
    "fresha/e5-large-v2-endpoint",
    "ggrn/e5-small-v2",
    "michaelfeil/ct2fast-e5-small",
    "jncraton/e5-small-v2-ct2-int8",
    "anttip/ct2fast-e5-small-v2-hfie",
    "newsrx/instructor-large",
    "newsrx/instructor-xl",
    "dmlls/all-mpnet-base-v2",
    "cgldo/semanticClone",
    "Malmuk1/e5-large-v2_Sharded",
    "jncraton/gte-small-ct2-int8",
}


