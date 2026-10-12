"""Tiny vector store of resolved cases (TF-IDF stand-in for Pinecone/FAISS).
The feedback loop: reviewer-confirmed BENIGN patterns are embedded here, so the same
pattern auto-completes next time. Swap for Bedrock embeddings + Pinecone in production."""

class CaseStore:
    def __init__(self):
        self.sigs, self.labels, self._vec, self._mat = [], [], None, None

    def add(self, signature, label="benign"):
        from sklearn.feature_extraction.text import TfidfVectorizer      # loaded on first use to keep start-up memory low
        self.sigs.append(signature); self.labels.append(label)
        self._vec = TfidfVectorizer(token_pattern=r"\S+").fit(self.sigs)
        self._mat = self._vec.transform(self.sigs)

    def best_match(self, signature):
        if not self.sigs:
            return 0.0, None
        from sklearn.metrics.pairwise import cosine_similarity
        sims = cosine_similarity(self._vec.transform([signature]), self._mat)[0]
        i = int(sims.argmax())
        return float(sims[i]), self.labels[i]
