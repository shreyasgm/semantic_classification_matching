import os
import re
from pathlib import Path

import fasttext as ft
import nltk
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.spatial.distance import cdist, pdist, squareform


PROJ = Path(os.path.realpath("."))
ROOT = PROJ.parent
DATA = ROOT / "data"


def download_resources():
    """Download NLTK and FastText resources required for NLP"""
    import fasttext.util

    # Download NLTK resources if they do not exist
    try:
        nltk.data.find("tokenizers/punkt")
        nltk.data.find("corpora/wordnet")
        nltk.data.find("corpora/stopwords")
    except LookupError:
        nltk.download("punkt")
        nltk.download("wordnet")
        nltk.download("stopwords")

    # Download FastText model
    fasttext.util.download_model("en", if_exists="ignore")


def process_raw_classification(clas_df, titlecol, codecol=None):
    # Prepare unique classification df with a code
    clas_df = clas_df.drop_duplicates().copy()
    # Remove nulls
    clas_df = clas_df.dropna(subset=[titlecol])
    clas_df = clas_df[clas_df[titlecol].str.strip() != ""]
    # Create ID if there's none
    if codecol is None:
        clas_df["codecol"] = [f"a{str(x)}" for x in range(len(clas_df))]
        codecol = "codecol"
    # Error if codecol is duplicated
    if clas_df[codecol].duplicated().sum() > 0:
        raise ValueError("Code column is duplicated.")
    return clas_df, codecol


def preprocess_df(df, text_colname):
    # Setup
    en_stop = set(nltk.corpus.stopwords.words("english"))
    stemmer = nltk.stem.WordNetLemmatizer()
    word_punctuation_tokenizer = nltk.WordPunctTokenizer()

    def preprocess_text(document):
        # Remove all the special characters
        document = re.sub(r"\W", " ", str(document))

        # remove all single characters
        document = re.sub(r"\s+[a-zA-Z]\s+", " ", document)

        # Remove single characters from the start
        document = re.sub(r"\^[a-zA-Z]\s+", " ", document)

        # Substituting multiple spaces with single space
        document = re.sub(r"\s+", " ", document, flags=re.I)

        # Converting to Lowercase
        document = document.lower()

        # Lemmatization
        tokens = document.split()
        tokens = [stemmer.lemmatize(word) for word in tokens]
        tokens = [word for word in tokens if word not in en_stop]
        tokens = [word for word in tokens if len(word) > 3]

        preprocessed_text = " ".join(tokens)
        word_tokenized_corpus = word_punctuation_tokenizer.tokenize(preprocessed_text)
        return word_tokenized_corpus

    final_corpus = [preprocess_text(doc) for doc in tqdm(df[text_colname])]
    assert len(final_corpus) == len(df), "Pre-processing error with some statements"
    return final_corpus


def get_mean_word_vector(doc_text, ft_model):
    """Average word vectors to get document vector"""
    # Calculate word vector and get mean
    if len(doc_text) > 0:
        word_vectors = np.array([ft_model.get_word_vector(x) for x in doc_text])
        result = np.nanmean(word_vectors, axis=0)
        if result is not None:
            return result

    # If we get empty result or if input is empty, return nan
    result = np.array([np.nan] * ft_model.get_word_vector("").shape[0])
    return result


def vectorize_textlist(textlist, ft_model):
    """Convert textlist to embeddings"""
    doc_vectors = np.array(
        [get_mean_word_vector(x, ft_model) for x in tqdm(textlist, total=len(textlist))]
    )
    return doc_vectors


def get_embeddings(df, ft_model, textcol):
    """Preprocess text and convert to embeddings"""
    df_text = preprocess_df(df, textcol)
    vec = vectorize_textlist(df_text, ft_model)
    vec_df = pd.concat([df.reset_index(drop=True), pd.DataFrame(vec)], axis=1)
    vec_df = vec_df.set_index(keys=list(df.columns))
    vec_df.columns = [str(x) for x in vec_df.columns]
    return vec_df


def prepare_data_and_embeddings(
    clas_a_df, clas_b_df, titlecol_a, titlecol_b, codecol_a, codecol_b
):
    print("Pre-processing text")
    # Process raw classification df's
    clas_a_df, codecol_a = process_raw_classification(clas_a_df, titlecol_a, codecol_a)
    clas_b_df, codecol_b = process_raw_classification(clas_b_df, titlecol_b, codecol_b)

    # Download required resources
    download_resources()

    # Initialize model
    ft_model = ft.load_model("cc.en.300.bin")

    # Get embeddings
    print("Preparing embeddings")
    clas_a_vec = get_embeddings(clas_a_df, ft_model, titlecol_a)
    clas_b_vec = get_embeddings(clas_b_df, ft_model, titlecol_b)
    return clas_a_vec, clas_b_vec, codecol_a, codecol_b


def get_topn_matches(clas_a, clas_b, titlecol_b, codecol_b, n_best,):
    """Get top N matches in clas_b for each element in clas_a"""
    # Calculate distance matrix using cosine similarity
    similarity_df = pd.DataFrame(
        1 - cdist(clas_a.values, clas_b.values, metric="cosine"),
        index=clas_a.index,
        columns=clas_b.index,
    )
    # Get order of sorting, as well as the scores
    order = np.argsort(-similarity_df.values, axis=1)[:, :n_best]
    scores = -np.sort(-similarity_df.values, axis=1)[:, :n_best]
    # Get the corresponding codes and names, and return
    clas_b_codes = similarity_df.columns.get_level_values(codecol_b)
    clas_b_names = similarity_df.columns.get_level_values(titlecol_b)
    result_names = pd.DataFrame(
        np.array(clas_b_names)[order],
        columns=["top{}".format(i) for i in range(1, n_best + 1)],
        index=similarity_df.index,
    )
    result_codes = pd.DataFrame(
        np.array(clas_b_codes)[order],
        columns=["top{}".format(i) for i in range(1, n_best + 1)],
        index=similarity_df.index,
    )
    result_scores = pd.DataFrame(
        scores,
        columns=["top{}".format(i) for i in range(1, n_best + 1)],
        index=similarity_df.index,
    ) 
    result_dict = {"names": result_names, "codes": result_codes, "scores": result_scores}
    return result_dict


def process_data_and_match(
    clas_a_df,
    clas_b_df,
    titlecol_a,
    titlecol_b,
    codecol_a=None,
    codecol_b=None,
    n_best=5,
):
    """
    Args:
        clas_a_df, clas_b_df: Dataframes containing titles to match
        titlecol_a, titlecol_b: Column names in respective dataframes containing titles
        codecol_a, codecol_b: (optional) if the titles have codes, column name corresponding to the code
        n_best: top N matches to return
    """
    clas_a_vec, clas_b_vec, codecol_a, codecol_b = prepare_data_and_embeddings(
        clas_a_df, clas_b_df, titlecol_a, titlecol_b, codecol_a, codecol_b
    )
    result_dict = get_topn_matches(
        clas_a_vec, clas_b_vec, titlecol_b, codecol_b, n_best,
    )
    return result_dict
