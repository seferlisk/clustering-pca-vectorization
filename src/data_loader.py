import pandas as pd
from sklearn.datasets import fetch_20newsgroups
from src import TextPreprocessor


class DatasetManager:
    """Manages loading and basic integrity of the text datasets."""

    def __init__(self, bbc_path):
        self.bbc_path = bbc_path
        self.preprocessor = TextPreprocessor()
        self.datasets = {}

    def prepare_data(self):
        # 1. Load BBC News
        print("Processing BBC News Dataset...")
        bbc_df = pd.read_csv(self.bbc_path)
        # Assuming columns are 'text' and 'category'
        bbc_df['cleaned_text'] = bbc_df['text'].apply(self.preprocessor.clean)
        self.datasets['bbc'] = bbc_df

        # 2. Load 20NewsGroups
        print("Processing 20NewsGroups Dataset...")
        ng_data = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'))
        ng_df = pd.DataFrame({'text': ng_data.data, 'label': ng_data.target})
        # Filter out empty entries that may occur after removing headers/footers
        ng_df = ng_df[ng_df['text'].str.strip() != ""].copy()
        ng_df['cleaned_text'] = ng_df['text'].apply(self.preprocessor.clean)
        self.datasets['20news'] = ng_df

        print("Data Loading Complete.")
        return self.datasets