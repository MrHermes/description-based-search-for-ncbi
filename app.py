import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import gradio as gr


class SpermatophyteSearch():
    index = faiss.read_index('spermatophyte2.index')
    data = pd.read_csv('data/ready/data_spermatophyte.csv')
    model = SentenceTransformer('model2/')

    query = ""
    top_k = 0
    results = list()

    def fetch_plant(self, dataframe_idx):
        info = self.data.iloc[dataframe_idx]
        meta = dict()
        meta['organism_name'] = info['organism_name']
        return meta

    def search(self):
        query_vector = self.model.encode([self.query])
        self.top_k = self.index.search(query_vector, self.top_k)
        result_id = self.top_k[1].tolist()[0]
        result_id = list(np.unique(result_id))
        results = [self.fetch_plant(idx) for idx in result_id]
        return results

    def recommend(self, query, top_k=5):
        self.top_k = top_k
        self.query = query

        self.results = self.search()

        return self.results


def giveRecommend(Description, Limit):
    results = Spermatophyte.recommend(Description, int(Limit))

    out = ""
    for result in results:
        out += str(result['organism_name']) + ",\n"

    return out


Spermatophyte = SpermatophyteSearch()

# Main program
app = gr.Interface(fn=giveRecommend,
                   inputs=["textbox", "number"],
                   outputs="textbox",
                   title="Description-based Search for Spermatophyte",
                   description="Enter your keyword."
                   )
app.launch()