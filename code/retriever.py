"""
Retriever — builds a ChromaDB vector store from corpus chunks using
local embeddings, plus BM25 for hybrid search, and a CrossEncoder for re-ranking.
"""

from __future__ import annotations

import hashlib
from typing import List, Optional

import chromadb
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

from config import VECTOR_STORE_DIR, TOP_K
from corpus_loader import Chunk, load_corpus

_COLLECTION_NAME = "support_corpus"

def _build_id(chunk: Chunk, idx: int) -> str:
    raw = f"{idx}:{chunk.company}:{chunk.source_file}:{chunk.text[:200]}"
    return hashlib.md5(raw.encode()).hexdigest()

def tokenize(text: str) -> List[str]:
    return text.lower().split()

class Retriever:
    def __init__(self, rebuild: bool = False):
        VECTOR_STORE_DIR.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(
            path=str(VECTOR_STORE_DIR),
        )
        marker = VECTOR_STORE_DIR / ".built"
        if rebuild or not marker.exists():
            self._build_index()
            marker.touch()
            
        self._col = self._client.get_collection(name=_COLLECTION_NAME)
        print(f"[retriever] Collection ready — {self._col.count()} chunks indexed")
        
        # Build in-memory BM25 index
        print("[retriever] Building BM25 index in memory...")
        all_data = self._col.get(include=["documents", "metadatas"])
        self._bm25_docs = all_data["documents"]
        self._bm25_metas = all_data["metadatas"]
        self._bm25_ids = all_data["ids"]
        tokenized_corpus = [tokenize(doc) for doc in self._bm25_docs]
        self._bm25 = BM25Okapi(tokenized_corpus)
        
        # Load CrossEncoder
        print("[retriever] Loading CrossEncoder for re-ranking...")
        self._cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

    def _build_index(self) -> None:
        try:
            self._client.delete_collection(_COLLECTION_NAME)
        except Exception:
            pass

        col = self._client.create_collection(
            name=_COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        chunks = load_corpus()

        BATCH = 500
        for i in range(0, len(chunks), BATCH):
            batch = chunks[i : i + BATCH]
            col.add(
                ids=[_build_id(c, i + j) for j, c in enumerate(batch)],
                documents=[c.text for c in batch],
                metadatas=[
                    {
                        "company": c.company,
                        "category": c.category,
                        "source_file": c.source_file,
                        "title": c.title,
                    }
                    for c in batch
                ],
            )
            print(f"  [index] Embedded batch {i // BATCH + 1}"
                  f" ({min(i + BATCH, len(chunks))}/{len(chunks)})")
        print(f"[retriever] Indexed {len(chunks)} chunks (local embeddings)")

    def _rrf(self, list1: List[str], list2: List[str], k: int = 60) -> dict[str, float]:
        """Reciprocal Rank Fusion"""
        scores = {}
        for rank, item in enumerate(list1):
            scores[item] = scores.get(item, 0) + 1 / (k + rank)
        for rank, item in enumerate(list2):
            scores[item] = scores.get(item, 0) + 1 / (k + rank)
        return scores

    def retrieve(
        self,
        query: str,
        company: Optional[str] = None,
        top_k: int = TOP_K,
    ) -> List[dict]:
        # 1. Vector Search (Top 50)
        where = {"company": company} if company else None
        vec_results = self._col.query(
            query_texts=[query],
            n_results=50,
            where=where,
        )
        vec_ids = vec_results["ids"][0] if vec_results["ids"] else []
        
        # 2. BM25 Search (Top 50)
        bm25_scores = self._bm25.get_scores(tokenize(query))
        
        # Filter by company manually for BM25
        filtered_bm25 = []
        for idx, score in enumerate(bm25_scores):
            if company and self._bm25_metas[idx]["company"] != company:
                continue
            filtered_bm25.append((score, self._bm25_ids[idx]))
            
        filtered_bm25.sort(key=lambda x: x[0], reverse=True)
        bm25_ids = [item[1] for item in filtered_bm25[:50]]
        
        # 3. Reciprocal Rank Fusion
        rrf_scores = self._rrf(vec_ids, bm25_ids)
        sorted_rrf_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)[:20]
        
        if not sorted_rrf_ids:
            return []
            
        # 4. Cross-Encoder Re-ranking
        # Fetch the actual documents for the top 20 candidates
        candidate_data = self._col.get(ids=sorted_rrf_ids, include=["documents", "metadatas"])
        
        # The get() method might not return in the same order as requested
        # Map id to doc/meta
        id_to_doc = {id_: doc for id_, doc in zip(candidate_data["ids"], candidate_data["documents"])}
        id_to_meta = {id_: meta for id_, meta in zip(candidate_data["ids"], candidate_data["metadatas"])}
        
        # Prepare pairs for cross-encoder
        pairs = [[query, id_to_doc[id_]] for id_ in sorted_rrf_ids]
        ce_scores = self._cross_encoder.predict(pairs)
        
        # Combine score with id
        scored_candidates = list(zip(ce_scores, sorted_rrf_ids))
        scored_candidates.sort(key=lambda x: x[0], reverse=True)
        
        # Take Top K
        final_ids = [id_ for score, id_ in scored_candidates[:top_k]]
        
        hits = []
        for id_ in final_ids:
            hits.append({
                "text": id_to_doc[id_],
                "company": id_to_meta[id_]["company"],
                "category": id_to_meta[id_]["category"],
                "source_file": id_to_meta[id_]["source_file"],
                "title": id_to_meta[id_]["title"],
            })
            
        return hits
