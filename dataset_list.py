
TASK_LIST_BITEXT_MINING = ['BUCC (de-en)', 'BUCC (fr-en)', 'BUCC (ru-en)', 'BUCC (zh-en)', 'Tatoeba (afr-eng)', 'Tatoeba (amh-eng)', 'Tatoeba (ang-eng)', 'Tatoeba (ara-eng)', 'Tatoeba (arq-eng)', 'Tatoeba (arz-eng)', 'Tatoeba (ast-eng)', 'Tatoeba (awa-eng)', 'Tatoeba (aze-eng)', 'Tatoeba (bel-eng)', 'Tatoeba (ben-eng)', 'Tatoeba (ber-eng)', 'Tatoeba (bos-eng)', 'Tatoeba (bre-eng)', 'Tatoeba (bul-eng)', 'Tatoeba (cat-eng)', 'Tatoeba (cbk-eng)', 'Tatoeba (ceb-eng)', 'Tatoeba (ces-eng)', 'Tatoeba (cha-eng)', 'Tatoeba (cmn-eng)', 'Tatoeba (cor-eng)', 'Tatoeba (csb-eng)', 'Tatoeba (cym-eng)', 'Tatoeba (dan-eng)', 'Tatoeba (deu-eng)', 'Tatoeba (dsb-eng)', 'Tatoeba (dtp-eng)', 'Tatoeba (ell-eng)', 'Tatoeba (epo-eng)', 'Tatoeba (est-eng)', 'Tatoeba (eus-eng)', 'Tatoeba (fao-eng)', 'Tatoeba (fin-eng)', 'Tatoeba (fra-eng)', 'Tatoeba (fry-eng)', 'Tatoeba (gla-eng)', 'Tatoeba (gle-eng)', 'Tatoeba (glg-eng)', 'Tatoeba (gsw-eng)', 'Tatoeba (heb-eng)', 'Tatoeba (hin-eng)', 'Tatoeba (hrv-eng)', 'Tatoeba (hsb-eng)', 'Tatoeba (hun-eng)', 'Tatoeba (hye-eng)', 'Tatoeba (ido-eng)', 'Tatoeba (ile-eng)', 'Tatoeba (ina-eng)', 'Tatoeba (ind-eng)', 'Tatoeba (isl-eng)', 'Tatoeba (ita-eng)', 'Tatoeba (jav-eng)', 'Tatoeba (jpn-eng)', 'Tatoeba (kab-eng)', 'Tatoeba (kat-eng)', 'Tatoeba (kaz-eng)', 'Tatoeba (khm-eng)', 'Tatoeba (kor-eng)', 'Tatoeba (kur-eng)', 'Tatoeba (kzj-eng)', 'Tatoeba (lat-eng)', 'Tatoeba (lfn-eng)', 'Tatoeba (lit-eng)', 'Tatoeba (lvs-eng)', 'Tatoeba (mal-eng)', 'Tatoeba (mar-eng)', 'Tatoeba (max-eng)', 'Tatoeba (mhr-eng)', 'Tatoeba (mkd-eng)', 'Tatoeba (mon-eng)', 'Tatoeba (nds-eng)', 'Tatoeba (nld-eng)', 'Tatoeba (nno-eng)', 'Tatoeba (nob-eng)', 'Tatoeba (nov-eng)', 'Tatoeba (oci-eng)', 'Tatoeba (orv-eng)', 'Tatoeba (pam-eng)', 'Tatoeba (pes-eng)', 'Tatoeba (pms-eng)', 'Tatoeba (pol-eng)', 'Tatoeba (por-eng)', 'Tatoeba (ron-eng)', 'Tatoeba (rus-eng)', 'Tatoeba (slk-eng)', 'Tatoeba (slv-eng)', 'Tatoeba (spa-eng)', 'Tatoeba (sqi-eng)', 'Tatoeba (srp-eng)', 'Tatoeba (swe-eng)', 'Tatoeba (swg-eng)', 'Tatoeba (swh-eng)', 'Tatoeba (tam-eng)', 'Tatoeba (tat-eng)', 'Tatoeba (tel-eng)', 'Tatoeba (tgl-eng)', 'Tatoeba (tha-eng)', 'Tatoeba (tuk-eng)', 'Tatoeba (tur-eng)', 'Tatoeba (tzl-eng)', 'Tatoeba (uig-eng)', 'Tatoeba (ukr-eng)', 'Tatoeba (urd-eng)', 'Tatoeba (uzb-eng)', 'Tatoeba (vie-eng)', 'Tatoeba (war-eng)', 'Tatoeba (wuu-eng)', 'Tatoeba (xho-eng)', 'Tatoeba (yid-eng)', 'Tatoeba (yue-eng)', 'Tatoeba (zsm-eng)']
TASK_LIST_BITEXT_MINING_OTHER = ["BornholmBitextMining"]

TASK_LIST_CLASSIFICATION = [
    "AmazonCounterfactualClassification (en)",
    "AmazonPolarityClassification",
    "AmazonReviewsClassification (en)",
    "Banking77Classification",
    "EmotionClassification",
    "ImdbClassification",
    "MassiveIntentClassification (en)",
    "MassiveScenarioClassification (en)",
    "MTOPDomainClassification (en)",
    "MTOPIntentClassification (en)",
    "ToxicConversationsClassification",
    "TweetSentimentExtractionClassification",
]

TASK_LIST_CLASSIFICATION_NORM = [x.replace(" (en)", "") for x in TASK_LIST_CLASSIFICATION]

TASK_LIST_CLASSIFICATION_DA = [
    "AngryTweetsClassification",
    "DanishPoliticalCommentsClassification",
    "DKHateClassification",
    "LccSentimentClassification",
    "MassiveIntentClassification (da)",
    "MassiveScenarioClassification (da)",
    "NordicLangClassification",
    "ScalaDaClassification",
]

TASK_LIST_CLASSIFICATION_NB = [
    "NoRecClassification",
    "NordicLangClassification",
    "NorwegianParliament",
    "MassiveIntentClassification (nb)",
    "MassiveScenarioClassification (nb)",
    "ScalaNbClassification",
]

TASK_LIST_CLASSIFICATION_PL = [
    "AbusiveClauses",
    "AllegroReviews",
    "CBD",
    "MassiveIntentClassification (pl)",
    "MassiveScenarioClassification (pl)",
    "PolEmo2.0-IN",
    "PolEmo2.0-OUT",
]

TASK_LIST_CLASSIFICATION_SV = [
    "DalajClassification",
    "MassiveIntentClassification (sv)",
    "MassiveScenarioClassification (sv)",
    "NordicLangClassification",
    "ScalaSvClassification",
    "SweRecClassification",
]

TASK_LIST_CLASSIFICATION_ZH = [
    "AmazonReviewsClassification (zh)",
    "IFlyTek",
    "JDReview",
    "MassiveIntentClassification (zh-CN)",
    "MassiveScenarioClassification (zh-CN)",
    "MultilingualSentiment",
    "OnlineShopping",
    "TNews",
    "Waimai",
]

TASK_LIST_CLASSIFICATION_OTHER = ['AmazonCounterfactualClassification (de)', 'AmazonCounterfactualClassification (ja)', 'AmazonReviewsClassification (de)', 'AmazonReviewsClassification (es)', 'AmazonReviewsClassification (fr)', 'AmazonReviewsClassification (ja)', 'AmazonReviewsClassification (zh)', 'MTOPDomainClassification (de)', 'MTOPDomainClassification (es)', 'MTOPDomainClassification (fr)', 'MTOPDomainClassification (hi)', 'MTOPDomainClassification (th)', 'MTOPIntentClassification (de)', 'MTOPIntentClassification (es)', 'MTOPIntentClassification (fr)', 'MTOPIntentClassification (hi)', 'MTOPIntentClassification (th)', 'MassiveIntentClassification (af)', 'MassiveIntentClassification (am)', 'MassiveIntentClassification (ar)', 'MassiveIntentClassification (az)', 'MassiveIntentClassification (bn)', 'MassiveIntentClassification (cy)', 'MassiveIntentClassification (de)', 'MassiveIntentClassification (el)', 'MassiveIntentClassification (es)', 'MassiveIntentClassification (fa)', 'MassiveIntentClassification (fi)', 'MassiveIntentClassification (fr)', 'MassiveIntentClassification (he)', 'MassiveIntentClassification (hi)', 'MassiveIntentClassification (hu)', 'MassiveIntentClassification (hy)', 'MassiveIntentClassification (id)', 'MassiveIntentClassification (is)', 'MassiveIntentClassification (it)', 'MassiveIntentClassification (ja)', 'MassiveIntentClassification (jv)', 'MassiveIntentClassification (ka)', 'MassiveIntentClassification (km)', 'MassiveIntentClassification (kn)', 'MassiveIntentClassification (ko)', 'MassiveIntentClassification (lv)', 'MassiveIntentClassification (ml)', 'MassiveIntentClassification (mn)', 'MassiveIntentClassification (ms)', 'MassiveIntentClassification (my)', 'MassiveIntentClassification (nl)', 'MassiveIntentClassification (pt)', 'MassiveIntentClassification (ro)', 'MassiveIntentClassification (ru)', 'MassiveIntentClassification (sl)', 'MassiveIntentClassification (sq)', 'MassiveIntentClassification (sw)', 'MassiveIntentClassification (ta)', 'MassiveIntentClassification (te)', 'MassiveIntentClassification (th)', 'MassiveIntentClassification (tl)', 'MassiveIntentClassification (tr)', 'MassiveIntentClassification (ur)', 'MassiveIntentClassification (vi)', 'MassiveIntentClassification (zh-TW)', 'MassiveScenarioClassification (af)', 'MassiveScenarioClassification (am)', 'MassiveScenarioClassification (ar)', 'MassiveScenarioClassification (az)', 'MassiveScenarioClassification (bn)', 'MassiveScenarioClassification (cy)', 'MassiveScenarioClassification (de)', 'MassiveScenarioClassification (el)', 'MassiveScenarioClassification (es)', 'MassiveScenarioClassification (fa)', 'MassiveScenarioClassification (fi)', 'MassiveScenarioClassification (fr)', 'MassiveScenarioClassification (he)', 'MassiveScenarioClassification (hi)', 'MassiveScenarioClassification (hu)', 'MassiveScenarioClassification (hy)', 'MassiveScenarioClassification (id)', 'MassiveScenarioClassification (is)', 'MassiveScenarioClassification (it)', 'MassiveScenarioClassification (ja)', 'MassiveScenarioClassification (jv)', 'MassiveScenarioClassification (ka)', 'MassiveScenarioClassification (km)', 'MassiveScenarioClassification (kn)', 'MassiveScenarioClassification (ko)', 'MassiveScenarioClassification (lv)', 'MassiveScenarioClassification (ml)', 'MassiveScenarioClassification (mn)', 'MassiveScenarioClassification (ms)', 'MassiveScenarioClassification (my)', 'MassiveScenarioClassification (nl)', 'MassiveScenarioClassification (pt)', 'MassiveScenarioClassification (ro)', 'MassiveScenarioClassification (ru)', 'MassiveScenarioClassification (sl)', 'MassiveScenarioClassification (sq)', 'MassiveScenarioClassification (sw)', 'MassiveScenarioClassification (ta)', 'MassiveScenarioClassification (te)', 'MassiveScenarioClassification (th)', 'MassiveScenarioClassification (tl)', 'MassiveScenarioClassification (tr)', 'MassiveScenarioClassification (ur)', 'MassiveScenarioClassification (vi)', 'MassiveScenarioClassification (zh-TW)']

TASK_LIST_CLUSTERING = [
    "ArxivClusteringP2P",
    "ArxivClusteringS2S",
    "BiorxivClusteringP2P",
    "BiorxivClusteringS2S",
    "MedrxivClusteringP2P",
    "MedrxivClusteringS2S",
    "RedditClustering",
    "RedditClusteringP2P",
    "StackExchangeClustering",
    "StackExchangeClusteringP2P",
    "TwentyNewsgroupsClustering",
]


TASK_LIST_CLUSTERING_DE = [
    "BlurbsClusteringP2P",
    "BlurbsClusteringS2S",
    "TenKGnadClusteringP2P",
    "TenKGnadClusteringS2S",
]

TASK_LIST_CLUSTERING_PL = [
    "8TagsClustering",
]

TASK_LIST_CLUSTERING_ZH = [
    "CLSClusteringP2P",
    "CLSClusteringS2S",
    "ThuNewsClusteringP2P",
    "ThuNewsClusteringS2S",
]

TASK_LIST_PAIR_CLASSIFICATION = [
    "SprintDuplicateQuestions",
    "TwitterSemEval2015",
    "TwitterURLCorpus",
]

TASK_LIST_PAIR_CLASSIFICATION_PL = [
    "CDSC-E",
    "PPC",
    "PSC",
    "SICK-E-PL",
]

TASK_LIST_PAIR_CLASSIFICATION_ZH = [
    "Cmnli",
    "Ocnli",
]

TASK_LIST_RERANKING = [
    "AskUbuntuDupQuestions",
    "MindSmallReranking",
    "SciDocsRR",
    "StackOverflowDupQuestions",
]

TASK_LIST_RERANKING_ZH = [
    "CMedQAv1",
    "CMedQAv2",
    "MMarcoReranking",
    "T2Reranking",
]

TASK_LIST_RETRIEVAL = [
    "ArguAna",
    "ClimateFEVER",
    "CQADupstackRetrieval",
    "DBPedia",
    "FEVER",
    "FiQA2018",
    "HotpotQA",
    "MSMARCO",
    "NFCorpus",
    "NQ",
    "QuoraRetrieval",
    "SCIDOCS",
    "SciFact",
    "Touche2020",
    "TRECCOVID",
]

TASK_LIST_RETRIEVAL_PL = [
    "ArguAna-PL",
    "DBPedia-PL",
    "FiQA-PL",
    "HotpotQA-PL",
    "MSMARCO-PL",
    "NFCorpus-PL",
    "NQ-PL",
    "Quora-PL",
    "SCIDOCS-PL",
    "SciFact-PL",
    "TRECCOVID-PL",
]

TASK_LIST_RETRIEVAL_ZH = [
    "CmedqaRetrieval",
    "CovidRetrieval",
    "DuRetrieval",
    "EcomRetrieval",
    "MedicalRetrieval",
    "MMarcoRetrieval",
    "T2Retrieval",
    "VideoRetrieval",
]

TASK_LIST_RETRIEVAL_NORM = TASK_LIST_RETRIEVAL + [
    "CQADupstackAndroidRetrieval",
    "CQADupstackEnglishRetrieval",
    "CQADupstackGamingRetrieval",
    "CQADupstackGisRetrieval",
    "CQADupstackMathematicaRetrieval",
    "CQADupstackPhysicsRetrieval",
    "CQADupstackProgrammersRetrieval",
    "CQADupstackStatsRetrieval",
    "CQADupstackTexRetrieval",
    "CQADupstackUnixRetrieval",
    "CQADupstackWebmastersRetrieval",
    "CQADupstackWordpressRetrieval"
]

TASK_LIST_STS = [
    "BIOSSES",
    "SICK-R",
    "STS12",
    "STS13",
    "STS14",
    "STS15",
    "STS16",
    "STS17 (en-en)",
    "STS22 (en)",
    "STSBenchmark",
]

TASK_LIST_STS_PL = [
    "CDSC-R",
    "SICK-R-PL",
    "STS22 (pl)",
]

TASK_LIST_STS_ZH = [
    "AFQMC",
    "ATEC",
    "BQ",
    "LCQMC",
    "PAWSX",
    "QBQTC",
    "STS22 (zh)",
    "STSB",
]

TASK_LIST_STS_OTHER = ["STS17 (ar-ar)", "STS17 (en-ar)", "STS17 (en-de)", "STS17 (en-tr)", "STS17 (es-en)", "STS17 (es-es)", "STS17 (fr-en)", "STS17 (it-en)", "STS17 (ko-ko)", "STS17 (nl-en)", "STS22 (ar)", "STS22 (de)", "STS22 (de-en)", "STS22 (de-fr)", "STS22 (de-pl)", "STS22 (es)", "STS22 (es-en)", "STS22 (es-it)", "STS22 (fr)", "STS22 (fr-pl)", "STS22 (it)", "STS22 (pl)", "STS22 (pl-en)", "STS22 (ru)", "STS22 (tr)", "STS22 (zh-en)", "STSBenchmark",]
TASK_LIST_STS_NORM = [x.replace(" (en)", "").replace(" (en-en)", "") for x in TASK_LIST_STS]

TASK_LIST_SUMMARIZATION = ["SummEval",]

TASK_LIST_EN = TASK_LIST_CLASSIFICATION + TASK_LIST_CLUSTERING + TASK_LIST_PAIR_CLASSIFICATION + TASK_LIST_RERANKING + TASK_LIST_RETRIEVAL + TASK_LIST_STS + TASK_LIST_SUMMARIZATION
TASK_LIST_PL = TASK_LIST_CLASSIFICATION_PL + TASK_LIST_CLUSTERING_PL + TASK_LIST_PAIR_CLASSIFICATION_PL + TASK_LIST_RETRIEVAL_PL + TASK_LIST_STS_PL
TASK_LIST_ZH = TASK_LIST_CLASSIFICATION_ZH + TASK_LIST_CLUSTERING_ZH + TASK_LIST_PAIR_CLASSIFICATION_ZH + TASK_LIST_RERANKING_ZH + TASK_LIST_RETRIEVAL_ZH + TASK_LIST_STS_ZH


def add_task(examples):
    # Could be added to the dataset loading script instead
    if examples["mteb_dataset_name"] in TASK_LIST_CLASSIFICATION_NORM + TASK_LIST_CLASSIFICATION_DA + TASK_LIST_CLASSIFICATION_NB + TASK_LIST_CLASSIFICATION_PL + TASK_LIST_CLASSIFICATION_SV + TASK_LIST_CLASSIFICATION_ZH:
        examples["mteb_task"] = "Classification"
    elif examples["mteb_dataset_name"] in TASK_LIST_CLUSTERING + TASK_LIST_CLUSTERING_DE + TASK_LIST_CLUSTERING_PL + TASK_LIST_CLUSTERING_ZH:
        examples["mteb_task"] = "Clustering"
    elif examples["mteb_dataset_name"] in TASK_LIST_PAIR_CLASSIFICATION + TASK_LIST_PAIR_CLASSIFICATION_PL + TASK_LIST_PAIR_CLASSIFICATION_ZH:
        examples["mteb_task"] = "PairClassification"
    elif examples["mteb_dataset_name"] in TASK_LIST_RERANKING + TASK_LIST_RERANKING_ZH:
        examples["mteb_task"] = "Reranking"
    elif examples["mteb_dataset_name"] in TASK_LIST_RETRIEVAL_NORM + TASK_LIST_RETRIEVAL_PL + TASK_LIST_RETRIEVAL_ZH:
        examples["mteb_task"] = "Retrieval"
    elif examples["mteb_dataset_name"] in TASK_LIST_STS_NORM + TASK_LIST_STS_PL + TASK_LIST_STS_ZH:
        examples["mteb_task"] = "STS"
    elif examples["mteb_dataset_name"] in TASK_LIST_SUMMARIZATION:
        examples["mteb_task"] = "Summarization"
    elif examples["mteb_dataset_name"] in [x.split(" ")[0] for x in TASK_LIST_BITEXT_MINING + TASK_LIST_BITEXT_MINING_OTHER]:
        examples["mteb_task"] = "BitextMining"
    else:
        print("WARNING: Task not found for dataset", examples["mteb_dataset_name"])
        examples["mteb_task"] = "Unknown"
    return examples