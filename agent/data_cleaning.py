"""
DataCleaner — Text preprocessing for Apple Support tweets.

Cleans text columns (mentions, URLs, HTML, emojis, stopwords) and stores in SQLite.
"""

import re
import sqlite3
import pandas as pd
from pathlib import Path


class DataCleaner:
    """Loads raw CSV, cleans all text columns, saves to SQLite."""

    STOPWORDS = {
        "i", "me", "my", "myself", "we", "our", "ours", "ourselves",
        "you", "your", "yours", "yourself", "yourselves",
        "he", "him", "his", "himself", "she", "her", "hers", "herself",
        "it", "its", "itself", "they", "them", "their", "theirs", "themselves",
        "what", "which", "who", "whom", "this", "that", "these", "those",
        "am", "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "having", "do", "does", "did", "doing",
        "a", "an", "the", "and", "but", "if", "or", "because", "as",
        "until", "while", "of", "at", "by", "for", "with", "about",
        "against", "between", "through", "during", "before", "after",
        "above", "below", "to", "from", "up", "down", "in", "out",
        "on", "off", "over", "under", "again", "further", "then", "once",
        "here", "there", "when", "where", "why", "how", "all", "both",
        "each", "few", "more", "most", "other", "some", "such",
        "nor", "only", "own", "same", "so", "than", "too", "very",
        "can", "will", "just", "don", "should", "now",
        "ll", "re", "ve", "ain", "aren", "couldn", "didn", "doesn",
        "hadn", "hasn", "haven", "isn", "mightn", "mustn", "needn",
        "shan", "shouldn", "wasn", "weren", "won", "wouldn",
        "im", "ive", "id", "youre", "youve", "youll", "youd",
        "hes", "shes", "theyre", "theyve", "theyll", "theyd",
        "wont", "dont", "doesnt", "didnt", "cant", "couldnt",
        "shouldnt", "wouldnt", "isnt", "arent", "wasnt", "werent",
        "hasnt", "havent", "hadnt", "mustnt",
        "get", "got", "getting", "go", "going", "gone", "went",
        "come", "came", "coming", "take", "took", "taken", "taking",
        "make", "made", "making", "know", "knew", "known",
        "think", "thought", "say", "said", "see", "saw", "seen",
        "want", "need", "use", "used", "using", "try", "tried",
        "also", "back", "even", "still", "way", "well", "like",
        "would", "could", "one", "two", "new", "right", "much",
        "let", "may", "might", "shall", "put", "keep",
    }

    TEXT_COLUMNS = ["complaint_text", "complaint_context", "apple_reply"]

    def __init__(self, csv_path, db_path):
        self.csv_path = Path(csv_path)
        self.db_path = Path(db_path)
        self.df = None

    def load(self):
        self.df = pd.read_csv(self.csv_path)
        return self

    def _clean_text(self, text):
        """Clean a single text string: mentions, HTML, emojis, special chars, stopwords."""
        if pd.isna(text) or not str(text).strip():
            return ""
        text = str(text)
        text = re.sub(r"@\w+", "", text)           # Remove @mentions
        text = re.sub(r"&\w+;", " ", text)          # HTML entities
        text = text.encode("ascii", "ignore").decode("ascii")  # Strip non-ASCII
        text = re.sub(r"[^a-zA-Z\s]", " ", text)    # Keep only letters + spaces
        text = text.lower()
        words = [w for w in text.split() if w not in self.STOPWORDS and len(w) > 1]
        return " ".join(words)

    def _clean_context(self, context):
        """Clean complaint_context (conversation turns separated by ' || ')."""
        if pd.isna(context) or not str(context).strip():
            return ""
        cleaned_turns = []
        for turn in str(context).split(" || "):
            if ": " in turn:
                speaker, message = turn.split(": ", 1)
                cleaned_msg = self._clean_text(message)
                if cleaned_msg:
                    cleaned_turns.append(f"{speaker.strip().lower()}: {cleaned_msg}")
            else:
                cleaned = self._clean_text(turn)
                if cleaned:
                    cleaned_turns.append(cleaned)
        return " || ".join(cleaned_turns)

    def _parse_timestamp(self, ts):
        try:
            return pd.to_datetime(ts, format="%a %b %d %H:%M:%S %z %Y").strftime("%Y-%m-%d %H:%M:%S")
        except Exception:
            return ts

    def clean_all(self):
        """Apply cleaning to every text column. Adds _clean suffix columns."""
        for col in self.TEXT_COLUMNS:
            if col not in self.df.columns:
                continue
            cleaner = self._clean_context if col == "complaint_context" else self._clean_text
            self.df[f"{col}_clean"] = self.df[col].apply(cleaner)

        if "created_at" in self.df.columns:
            self.df["created_at_parsed"] = self.df["created_at"].apply(self._parse_timestamp)
        return self

    def save_to_sqlite(self, table_name="apple_support"):
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.db_path)
        self.df.to_sql(table_name, conn, if_exists="replace", index=False)
        conn.close()
        return self

    def run(self):
        """Full pipeline: load → clean → save."""
        return self.load().clean_all().save_to_sqlite()


if __name__ == "__main__":
    DataCleaner("processed/apple_rag.csv", "processed/apple_support.db").run()
