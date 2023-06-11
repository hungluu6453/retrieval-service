import os
import constant
import string
import re
from nltk import word_tokenize as lib_tokenizer
from code.bm25 import BM25Gensim
from code.loader import Document_Loader
from vncorenlp import VnCoreNLP

k = 2
b = 0.85

vncorenlp_dir = 'resource/vncorenlp/VnCoreNLP-1.2.jar'
annotator = VnCoreNLP(vncorenlp_dir, annotators='wseg')

def batch_annonate(self, text):
    def annotate(annotator):
        def apply(x):
            return ' '.join([' '.join(text) for text in annotator.tokenize(x)])
        return apply
    return list(map(annotate(self.annotator), text))


def strip_context(text):        
    text = text.lower()
    text = text.replace('\n', ' ') 
    text = re.sub(r'\s+', ' ', text) 
    text = text.strip() 
    return text

def post_process(x):
    x = " ".join(word_tokenize(strip_context(x))).strip()
    x = x.replace("\n"," ")
    x = "".join([i for i in x if i not in string.punctuation])
    x = ' '.join([' '.join(text) for text in annotator.tokenize(x)])
    return x

dict_map = dict({})  
def word_tokenize(text): 
    global dict_map 
    words = text.split() 
    words_norm = [] 
    for w in words: 
        if dict_map.get(w, None) is None: 
            dict_map[w] = ' '.join(lib_tokenizer(w)).replace('``', '"').replace("''", '"') 
        words_norm.append(dict_map[w]) 
    return words_norm 

DOCUMENT_LIST_TIENSI = [
    constant.QUY_CHE_DAO_TAO_TIEN_SI,
    constant.QUY_DINH_VE_CAU_TRUC_CHUONG_TRINH_DAO_TAO,
    constant.QUY_DINH_VE_DAO_TAO_HOC_VU,
    constant.QUY_DINH_GIANG_DAY
]

DOCUMENT_LIST_THACSI = [
    constant.QUY_DINH_TO_CHUC_DAO_TAO_TRINH_DO_THAC_SI,
    constant.QUY_DINH_VE_CAU_TRUC_CHUONG_TRINH_DAO_TAO,
    constant.QUY_DINH_VE_DAO_TAO_HOC_VU,
    constant.QUY_DINH_GIANG_DAY
]

DOCUMENT_LIST_SINHVIEN = [
    constant.QUY_DINH_VE_CAU_TRUC_CHUONG_TRINH_DAO_TAO,
    constant.QUY_DINH_VE_DAO_TAO_HOC_VU,
    constant.QUY_DINH_GIANG_DAY
]
DOC_LIST = [DOCUMENT_LIST_TIENSI, DOCUMENT_LIST_THACSI, DOCUMENT_LIST_SINHVIEN]
OUT_PATH = ["./document_retrieval/index/phd", "./document_retrieval/index/master", "./document_retrieval/index/undergraduate"]


passage_path = [doc['Passage_path'] for doc in DOCUMENT_LIST_TIENSI]
index_offset = [doc['Index_offset'] for doc in DOCUMENT_LIST_TIENSI]
out_path = OUT_PATH[0]

document_loader = Document_Loader(passage_path, index_offset)
data = document_loader.get_context()
data = list(map(post_process, data))

retriever = BM25Gensim(data)
retriever.create_model(out_path, k=k, b=b)


passage_path = [doc['Passage_path'] for doc in DOCUMENT_LIST_THACSI]
index_offset = [doc['Index_offset'] for doc in DOCUMENT_LIST_THACSI]
out_path = OUT_PATH[1]

document_loader = Document_Loader(passage_path, index_offset)
data = document_loader.get_context()
data = list(map(post_process, data))

retriever = BM25Gensim(data)
retriever.create_model(out_path, k=k, b=b)


passage_path = [doc['Passage_path'] for doc in DOCUMENT_LIST_SINHVIEN]
index_offset = [doc['Index_offset'] for doc in DOCUMENT_LIST_SINHVIEN]
out_path = OUT_PATH[2]

document_loader = Document_Loader(passage_path, index_offset)
data = document_loader.get_context()
data = list(map(post_process, data))

retriever = BM25Gensim(data)
retriever.create_model(out_path, k=k, b=b)

