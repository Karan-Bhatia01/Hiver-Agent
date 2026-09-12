"""Project configuration — paths, API settings, and hyperparameters."""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(override=True)

# Project Paths
BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "dataset"
PROCESSED_DIR = BASE_DIR / "processed"
PROMPTS_DIR = BASE_DIR / "prompts"

DB_PATH = PROCESSED_DIR / "apple_support.db"
FAISS_DIR = PROCESSED_DIR / "faiss_index"
INTENTS_DIR = PROCESSED_DIR / "intents"
INTENT_MAP_PATH = INTENTS_DIR / "intent_map.json"
CENTROIDS_PATH = INTENTS_DIR / "centroids.npy"

# Models & API
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
GROQ_MODEL_NAME = os.getenv("GROQ_MODEL_NAME", "openai/gpt-oss-120b")
COHERE_RERANK_MODEL = "rerank-v3.5"

# Hyperparameters
DEFAULT_TOP_K = 3
DEFAULT_CANDIDATE_K = 20
DEFAULT_RRF_ALPHA = 0.6
MAX_GENERATION_TOKENS = 800
GUARDRAIL_MAX_TOKENS = 250
