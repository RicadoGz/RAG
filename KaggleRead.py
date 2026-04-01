from docx import Document  
from sentence_transformers import SentenceTransformer  
from sentence_transformers import util   
from charllama import ask_kaggle_llama  
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



def dense_search(  # function goal: use the sentence transformer model to search the most similar instruction and return the response
    model: SentenceTransformer,  # trasnfer query to vector
    query: str,  # user question 
    corpus_emb,  # embedding dlist（N x D）
    rows: list[dict],  # ：original resource（this was for to get response）
    top_k: int = 5  # gow much in formation you want to return
) -> list[dict]:  # this will reutn a list of dict like this [{"score": 0.8612, "instruction": "...", "intent": "...", "category": "...", "response": "..."}, ...]





    # query = "I want to cancel my order"      (str)
    # [query] = ["I want to cancel my order"]  (list[str])
    # 
    # q_emb = [[0.021, -0.114, 0.087, ...]]    (matrix[1, D])
    q_emb = model.encode([query], normalize_embeddings=True)  

    # [
    #   {"corpus_id": 132, "score": 0.8612},
    #   {"corpus_id": 54, "score": 0.8341},
    #   ...
    # ]
    hits = util.semantic_search(q_emb, corpus_emb, top_k=top_k)[0]  # this alloww mutiple research and return a number of list of them
    #this time only research one question so this is [0] to get the first list of the result
    #semantic search will check the cosine similarity between the query embedding and each corpus embedding, and return the top_k most similar ones with their index (corpus_id) and score

    results = []  # this will take the answer for the question and return to the user
    for h in hits:  # loop each result been choice h；h struct: {"corpus_id": 123, "score": 0.84}
        # before deal
        # h = {"corpus_id": 132, "score": 0.8612}
        # after
        # idx = 132 (int)
        idx = h["corpus_id"]  # get the index of the accurance result
        #  
        # {
        #   "score": 0.8612,
        #   "instruction": rows[132]["instruction"],
        #   "intent": rows[132]["intent"],
        #   "category": rows[132]["category"],
        #   "response": rows[132]["response"]
        # }
        results.append({  #get the search result into human readable
            "score": float(h["score"]),  
            "instruction": rows[idx]["instruction"],  
            "intent": rows[idx]["intent"],  
            "category": rows[idx]["category"],  
            "response": rows[idx]["response"], 
        }) 
    #  
    # [
    #   {"score": 0.8612, "instruction": "...", "intent": "...", "category": "...", "response": "..."},
    #   {"score": 0.8341, "instruction": "...", "intent": "...", "category": "...", "response": "..."},
    # ]
    return results   



def build_tfidf_index(rows: list[dict]): 
    # input:
    # rows: list[dict]，需要 instruction 字段
    # output:
    # vectorizer: TfidfVectorizer（已 fit）

    #  
    # [
    #   {"instruction": " I need refund ", ...},
    #   {"instruction": "Track my package", ...},
    # ]
    #  
    # ["I need refund", "Track my package"]
    corpus_texts = [r["instruction"].strip() for r in rows]  #  
    vectorizer = TfidfVectorizer(  #  
        lowercase=True, #lowcase for all
        ngram_range=(1, 2),  # keep trace both one word or two word together
        min_df=2 #at least come twice will kepp-> big accurancy than the first one 
    )   
    # before deal 
    # corpus_texts: list[str], lenth=N
    # after deal
    # X: sparse matrix[N, V]
    #V is the instruction we keep
    #corpus_texts = [
   # "I want to cancel my order",
  #  "How can I cancel order quickly",
 #   "Where is my package"
#]
                    #after deal this will be like this:
                    #X = [          cancel  order  now    -> this is the V
                    #d0         >0     >0   >0
                    #d1          0     >0   >0
                    #d2         >0      0    0
                    # this d0 d 1 is X 
 
    # d0 = "cancel order now"
    # d1 = "track order now"
    # d2 = "cancel refund"
    # after min_df=2  : [cancel, order, now]
    #  
    # X[0] (d0) = [w_cancel, w_order, w_now] = [0.62, 0.49, 0.61]
    # X[1] (d1) = [w_cancel, w_order, w_now] = [0.00, 0.58, 0.57]
    # X[2] (d2) = [w_cancel, w_order, w_now] = [0.77, 0.00, 0.00]
 
    X = vectorizer.fit_transform(corpus_texts)   
 
    return vectorizer, X   



def tfidf_search(  # foudn the most closed in the term frequency0inverse document frequency space
    vectorizer: TfidfVectorizer,  # the fit vectore
    X,  # the sparse matrix of the corpus instruction in the tfidf space
    query: str,  # user search result
    rows: list[dict],  # origin resource for get the response
    top_k: int = 5  # return how many result you want
) -> list[dict]:  # give list 
    # 输入 -> 输出图:
    # query(str)
    #   |
    #   v vectorizer.transform
    # q(1 x V)
    #   + X(N x V)
    #   |
    #   v cosine_similarity
    # sims(长度N的一维数组)
    #   |
    #   v argsort 取前 top_k
    # top_idx(长度top_k的索引数组)
    #   |
    #   v 回查 rows
    # results(list[dict])
    # 141行处理前:
    # query = "I want to cancel my order" (str)
    # 141行处理后:
    # q = sparse matrix[1, V]
    q = vectorizer.transform([query])  # query transfer into the tfidf space
 
    # q: [1, V], X: [N, V]
 
     
    sims = cosine_similarity(q, X).ravel()  # this calculate the query accrance between the query and each instruction in the corpus, and return a list of similarity score with the same length as the corpus (N)
 
    # sims = [0.03, 0.61, 0.12, ...]
 
    # top_idx = [1, 28, 340, 7, 89]  
    top_idx = np.argsort(-sims)[:top_k]  # from high to low sort and give the most top one and get their index 
#.argsort will return the index of the sorted array, and -sims will sort from high to low, and [:top_k] will get the top_k index
 
    results = []  
    for idx in top_idx:  
        results.append({  
            "score": float(sims[idx]), 
            "instruction": rows[idx]["instruction"],   
            "intent": rows[idx]["intent"],   
            "category": rows[idx]["category"],   
            "response": rows[idx]["response"],  
        })   
  
    # [
    #   {"score": 0.6123, "instruction": "...", "intent": "...", "category": "...", "response": "..."},
    #   {"score": 0.5880, "instruction": "...", "intent": "...", "category": "...", "response": "..."},
    # ]
    return results   


def main():   
    csv_path = "document/csv/Bitext_Sample_Customer_Support_Training_Dataset_27K_responses-v11.csv"  # research the resource
   
    rows = load_rows_csv(csv_path)  # read into dictionary

    query = input("Please enter your question: ")  # get the user question

 
    model = SentenceTransformer("all-MiniLM-L6-v2")  # pretarin vector
    #  
    # corpus_emb: matrix[N, D]
    _, corpus_emb = build_dense_index(model, rows)   
    #  this will get the origin text and embedding 
    # dense_results: list[dict], 长度=5
    dense_results = dense_search(model, query, corpus_emb, rows, top_k=5)  #search to match question

    print("\n=== Dense (Embedding) top-5 ===")  # 打 
    for r in dense_results:  #  
        print(f'score={r["score"]:.4f} intent={r["intent"]} | {r["instruction"][:80]}')  #  
        print("resp:", r["response"], "\n---")  #  

    answer = ask_kaggle_llama(query, dense_results)  # this will send the question and the search result to the model and get the answer back
    print("\n=== OLLAMA Answer from Dense ===")
    print(answer)  # print the answer from the model

#build TF-idf
    vectorizer, X = build_tfidf_index(rows)   
    #get the dictionary result 
    tfidf_results = tfidf_search(vectorizer, X, query, rows, top_k=5)  # TF-IDF top-k 检索

    print("\n=== Sparse (TF-IDF) top-5 ===")  
    for r in tfidf_results:  
        print(f'score={r["score"]:.4f} intent={r["intent"]} | {r["instruction"][:80]}')  
        print("resp:", r["response"], "\n---")  
    answer = ask_kaggle_llama(query, tfidf_results)  # this will send the question and the search result to the model and get the answer back
    print("\n=== OLLAMA Answer from TF-IDF ===")
    print(answer)  # print the answer from the model

if __name__ == "__main__":  # Python 启动入口判断：仅当“直接运行该文件”时为 True
    main()  # 执行主函数
