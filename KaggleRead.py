from docx import Document  
from sentence_transformers import SentenceTransformer  
from sentence_transformers import util   
from chatStart import ask_ollama  
import csv   
import numpy as np  
from sklearn.feature_extraction.text import TfidfVectorizer  
from sklearn.metrics.pairwise import cosine_similarity   


 

def load_rows_csv(csv_path: str) -> list[dict]:  # read from cvs and generate function dictionary list

    # list[dict]
    #   [
    #     {"instruction": "I want to cancel my order", "intent": "cancel_order", ...},
    #     {"instruction": "Where is my package?", "intent": "track_order", ...},
    #   ]
    with open(csv_path, "r", encoding="utf-8") as f:  # open csvs and read only
        return list(csv.DictReader(f))  # this will put each row into a diction and return a list of dicts
    #if i want read first instruction
    #[0]["instruction"]  first intent [0]["intent"]  first category [0]["category"]  first response [0]["response"]


def build_dense_index(model: SentenceTransformer, rows: list[dict]):   


    # before process rows example:
    # [
    #   {"instruction": "  I want to cancel my order  ", "intent": "cancel_order", ...},
    #   {"instruction": "Where is my package?", "intent": "track_order", ...},
    # ]
    # after process corpus_texts example:
    # ["I want to cancel my order", "Where is my package?"]
    corpus_texts = [r["instruction"].strip() for r in rows]  # get  only the instruction fild and get out of the space



    # before handle
    # ["I want to cancel my order", "Where is my package?"]
    # 
    # [
    #   [0.021, -0.114, 0.087, ..., 0.009],  
    #   [-0.044, 0.063, 0.105, ..., -0.012],
    # ]
    corpus_emb = model.encode(corpus_texts, normalize_embeddings=True)  # encode a list of text string

# return (
#   corpus_texts: [
#     "I want to cancel my order",
#     "Where is my package?"
#   ],
#   corpus_emb: array([
#     [ 0.021, -0.114,  0.087, ...,  0.009],   #  
#     [-0.044,  0.063,  0.105, ..., -0.012]    #  
#   ])   
# )
#access by the index like [0] for the first text [0][0] for the first text's first dimension of the vector
    return corpus_texts, corpus_emb  #

