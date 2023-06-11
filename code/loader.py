import os
import json
import pandas as pd
import numpy as np

class Document_Loader:
    def __init__(self, passage_paths, index_offset) -> None:
        self.data = list()
        self.passage_paths = passage_paths
        self.index_offset = index_offset
        self.__load_passage()

    def __load_passage(self):
        for passage_index, passage_path in enumerate(self.passage_paths):
            index = self.index_offset[passage_index]
            for filename in sorted(os.listdir(passage_path)):
                with open(passage_path+'/'+filename, 'r') as f:
                    self.data.append(
                        {
                            'name': filename.split('.')[0],
                            'text': f.read(),
                            'id': index
                        }
                    )
                    index += 1
        
        self.context = np.array([text['text'] for text in self.data])
        self.name = np.array([text['name'] for text in self.data])
        self.id = np.array([text['id'] for text in self.data])