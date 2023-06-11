import os
import pickle
import string
import tensorflow as tf
import numpy as np
import pandas as pd

from pprint import pprint
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras.layers import GlobalAveragePooling1D
import tensorflow_addons as tfa
# from sklearn import svm
from typing import Any
from transformers import TFAutoModelForQuestionAnswering, AutoTokenizer

import constant

from code.bm25 import BM25Gensim
from code.loader import Document_Loader


class RetrieverModule:
    def __init__(
            self,
            useRerank=False,
            K=10,
            top_return=3
        ):
        
        self.document_loader_list = list()
        self.retriever_list = list()
        self.K = K
        self.useRerank = useRerank
        self.top_return = top_return

        self.load_retriever()

        if useRerank:
            self.load_reranker()

        self.load_question_retriver()

    def load_retriever(self):
        for index, docs in enumerate(constant.DOC_LIST):
            passage_path = [doc['Passage_path'] for doc in docs]
            index_offset = [doc['Index_offset'] for doc in docs]
            out_path = constant.OUT_PATH[index]
        
            self.document_loader_list.append(Document_Loader(passage_path, index_offset))
            self.retriever_list.append(BM25Gensim())
            self.retriever_list[index].load_model(out_path)

    def load_reranker(self):
        self.tokenizer = AutoTokenizer.from_pretrained(constant.LM_PATH)
        self.language_model = TFAutoModelForQuestionAnswering.from_pretrained(constant.LM_PATH, from_pt=True).roberta
        self.pooling = GlobalAveragePooling1D()
        # self.classifier = pickle.load(open(constant.CLASSIFIER_PATH, 'rb'))
        os.chdir("..")
        cur_dir = os.getcwd()
        checkpoint_path = os.path.join(cur_dir, constant.CLASSIFIER_PATH)
        self.classifier = keras.models.load_model(checkpoint_path)
        os.chdir("6_retriever_service")

    
    def load_question_retriver(self):
        self.question_retriever = BM25Gensim()
        self.question_retriever.load_model(constant.QUESTION_INDEX)
        self.question_df = pd.read_csv(constant.QUESTION_DATA_PATH)

    def retrieve_context(self, role, question):
        retriever = self.retriever_list[constant.ROLE_MAP[role]]
        document_loader = self.document_loader_list[constant.ROLE_MAP[role]]

        indexes, _ = retriever.get_top_result(question, self.K)

        contexts = document_loader.context[indexes]
        paragraph_ids = document_loader.id[indexes]
        
        if not self.useRerank:
            return contexts[:self.top_return].tolist(), paragraph_ids[:self.top_return].tolist()
        else:
            top_index, score =  self.rerank(question, contexts)
            print(top_index)
            print(score)
            return contexts[top_index].tolist(), paragraph_ids[top_index].tolist()

    def rerank(self, question, contexts):
        questions  = [question for context in contexts]
        contexts = contexts.tolist()

        encoded_data = self.tokenizer(
            questions,
            contexts,
            padding='max_length',
            return_tensors="tf",
            truncation=True,
        )

        emebdded_data = self.language_model(encoded_data)
        pooling_data = self.pooling(emebdded_data[0])
        # pred = self.classifier.decision_function(pooling_data)
        # return np.argsort(pred, axis=-1)[::-1][:self.top_return]

        pred = self.classifier(pooling_data)
        return np.argsort(np.max(pred, axis=-1), axis=-1)[::-1][:self.top_return], pred
    
    def retrieve_question(self, question):
        indexes, _ = self.question_retriever.get_top_result(question, topk=3)

        ret_df = self.question_df.iloc[indexes]

        if ('faq' not in set(ret_df.tag)) or ret_df.tag.tolist()[0] != 'faq':
            return [], False
        
        question = ret_df[ret_df.tag == 'faq']['question'].tolist()
        ret = ret_df[ret_df.tag == 'faq']['return'].tolist()
        
        answers = list()
        for i in range(len(question)):
            answer = """
            Câu hỏi tương tự: 
                {}?
            \n
            Câu trả lời: 
                {}
            """.format(question[i].capitalize(), ret[i])
            answers.append(answer)
        return answers , True

    def __call__(self, role, question):
        answer, isFAQ = self.retrieve_question(question)

        if isFAQ:
            return answer, [], isFAQ

        context, ids = self.retrieve_context(role, question)
        return context, ids, isFAQ



