import re
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from rank_bm25 import BM25Okapi

def is_binary_string(s):
    return bool(re.match('^[01]+$', s))

def read_jsonl_file(filename):
    data = []
    with open(filename, 'r', encoding='utf-8') as file:
        for line in file:
            data.append(json.loads(line))
    return data

def write_jsonl_file(filename, data):
    with open(filename, 'w', encoding='utf-8') as f:
        for entry in data:
            json_line = json.dumps(entry, ensure_ascii=False)
            f.write(json_line + '\n') 

def compute_similarity_TF_IDF(text1, text2):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([text1, text2])
    similirity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
    return similirity[0][0]

def batch_process(items, batch_size):
    for i in range(0, len(items), batch_size):
        yield items[i: i + batch_size]

def calculate_bm25_similarity(text1, text2):
    text1_tokens = word_tokenize(text1)
    text2_tokens = word_tokenize(text2)
    stop_words = set(stopwords.words('english'))
    text1_tokens = [word for word in text1_tokens if word.lower() not in stop_words]
    text2_tokens = [word for word in text2_tokens if word.lower() not in stop_words]
    corpus = [text1_tokens, text2_tokens]
    bm25 = BM25Okapi(corpus)
    score = bm25.get_scores(text1_tokens)[1]
    return score

def truncate_text(text, max_bytes=20000):
    encoded_text = text.encode('utf-8')
    if len(encoded_text) > max_bytes:
        return encoded_text[:max_bytes].decode('utf-8', 'ignore')
    return text