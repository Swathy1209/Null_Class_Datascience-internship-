import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class MedicalQuestionRetriever:
    """Retrieves answers for medical questions using TF-IDF + Cosine Similarity"""

    def __init__(self, dataset_path):
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"Dataset not found at {dataset_path}")
            
        self.dataset = pd.read_csv(dataset_path, encoding='utf-8', on_bad_lines="skip")

        # Convert column names to lowercase to prevent mismatches
        self.dataset.columns = self.dataset.columns.str.lower()

        # Ensure required columns exist
        required_columns = {"question", "answer"}
        missing_columns = required_columns - set(self.dataset.columns)
        if missing_columns:
            raise KeyError(f"Missing required columns: {missing_columns}. Available columns: {self.dataset.columns}")

        self.dataset['processed_question'] = self.dataset['question'].str.lower()
        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.tfidf_matrix = self.vectorizer.fit_transform(self.dataset['processed_question'])

    def retrieve_answer(self, query, top_k=3):
        """Finds the most relevant answers"""
        processed_query = query.lower()
        query_vector = self.vectorizer.transform([processed_query])
        similarities = cosine_similarity(query_vector, self.tfidf_matrix)[0]
        top_indices = similarities.argsort()[-top_k:][::-1]

        return [
            {'question': self.dataset.iloc[idx]['question'],
             'answer': self.dataset.iloc[idx]['answer'],
             'confidence': round(similarities[idx] * 100, 2)}
            for idx in top_indices
        ]
