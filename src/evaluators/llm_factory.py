"""
Copyright (c) 2026 Badan Riset dan Inovasi Nasional (BRIN)
THIS SOFTWARE SOURCE CODE AND ANY EXECUTABLE DERIVED THEREOF ARE PROPRIETARY
TO BRIN, AS APPLICABLE, AND SHALL NOT BE USED IN ANY WAY
OTHER THAN BEFOREHAND AGREED ON BY BRIN, NOR BE REPRODUCED
OR DISCLOSED TO THIRD PARTIES WITHOUT PRIOR WRITTEN AUTHORIZATION BY
BRIN, AS APPLICABLE.

Author      : Chandra
Version     : 0.1.0
Created     : 13-10-2026
Description : Factory wrapper untuk menginisialisasi LLM dan Embedding model
              dengan dukungan multi-backend (Ollama, OpenAI API, dan vLLM).

Changelog:
- 0.1.0 (13-10-2026): Inisialisasi factory untuk Ollama (Gemma, Qwen), OpenAI (GPT-4o), dan embeddings HuggingFace.
"""

import os
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper


def build_evaluator_llm(backend: str, model_name: str, num_ctx: int = 131072, **kwargs):
    """
    Factory untuk membangun wrapper LLM evaluator RAGAS berdasarkan backend yang dipilih.
    
    Args:
        backend (str): 'ollama', 'openai', atau 'vllm'.
        model_name (str): Nama model target (contoh: 'gpt-4o-mini', 'gemma3:12b', 'qwen2.5:14b').
        num_ctx (int): Kapasitas context window untuk Ollama (default 131072 tokens).
    
    Returns:
        LangchainLLMWrapper: Wrapper LLM yang kompatibel dengan RAGAS.
    """
    backend = backend.lower()
    if backend == "ollama":
        from langchain_ollama import ChatOllama

        ollama_host = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")
        llm = ChatOllama(
            base_url=ollama_host,
            model=model_name,
            temperature=0.0,
            num_ctx=num_ctx,
        )
        return LangchainLLMWrapper(llm)

    elif backend == "openai":
        from langchain_openai import ChatOpenAI

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY tidak ditemukan di environment variables / file .env.")

        llm = ChatOpenAI(
            model=model_name,
            temperature=0.0,
            api_key=api_key,
        )
        return LangchainLLMWrapper(llm)

    elif backend == "vllm":
        from langchain_openai import ChatOpenAI

        api_base = os.getenv("VLLM_API_BASE", "http://127.0.0.1:8000/v1")
        api_key = os.getenv("VLLM_API_KEY", "empty")

        llm = ChatOpenAI(
            model=model_name,
            openai_api_base=api_base,
            openai_api_key=api_key,
            temperature=0.0,
        )
        return LangchainLLMWrapper(llm)

    else:
        raise ValueError(f"Backend '{backend}' tidak didukung. Pilihan yang valid: 'ollama', 'openai', 'vllm'.")


def build_evaluator_embeddings(backend: str = "hf", model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
    """
    Factory untuk membangun embeddings wrapper untuk metrik kesamaan semantik (Answer Correctness).
    
    Args:
        backend (str): 'hf' (Hugging Face) atau 'openai'.
        model_name (str): Nama model embedding.
    
    Returns:
        LangchainEmbeddingsWrapper: Wrapper embedding RAGAS.
    """
    if backend == "hf":
        from langchain_community.embeddings import HuggingFaceEmbeddings

        embeddings = HuggingFaceEmbeddings(model_name=model_name)
        return LangchainEmbeddingsWrapper(embeddings)
    elif backend == "openai":
        from langchain_openai import OpenAIEmbeddings

        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        return LangchainEmbeddingsWrapper(embeddings)
    else:
        raise ValueError(f"Embedding backend '{backend}' tidak didukung. Pilihan: 'hf', 'openai'.")
