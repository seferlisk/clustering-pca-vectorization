import pandas as pd
from sklearn.datasets import fetch_20newsgroups
from src import TextPreprocessor


class DatasetManager:
    """Manages loading and checks basic integrity of the text datasets."""

    def __init__(self, bbc_path):
        self.bbc_path = bbc_path
        self.preprocessor = TextPreprocessor()
        self.datasets = {}

    def prepare_data(self):
        print("Processing BBC News Dataset...")
        # 1. Load BBC News with automatic delimiter detection
        try:
            bbc_df = pd.read_csv(self.bbc_path)

            # Print columns to help debug
            print(f"Detected columns in BBC file: {list(bbc_df.columns)}")

            # Robust column selection: find the column that likely contains the text
            # We look for 'text', 'content', or 'article' (case-insensitive)
            text_col = next((c for c in bbc_df.columns if c.lower() in ['text', 'content', 'article']), None)
            label_col = next((c for c in bbc_df.columns if c.lower() in ['category', 'label', 'target']), None)

            if text_col is None:
                raise ValueError(f"Could not find a text column. Available columns: {list(bbc_df.columns)}")

            # Rename columns to our standard format for the rest of the pipeline
            bbc_df = bbc_df.rename(columns={text_col: 'text', label_col: 'category'})

            bbc_df['cleaned_text'] = bbc_df['text'].apply(self.preprocessor.clean)
            self.datasets['bbc'] = bbc_df

        except Exception as e:
            print(f"Error loading BBC dataset: {e}")
            raise

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