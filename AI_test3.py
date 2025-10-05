from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer("all-mpnet-base-v2")

target_word = "queues"
context_sentence = "But when I visited there were no queues in the waiting lounge."
candidates = ["complaints", "chairs", "workers", "lines"]

# Encode target word in context + candidates
embeddings = model.encode([context_sentence] + candidates, convert_to_tensor=True)
target_emb = embeddings[0]
candidate_embs = embeddings[1:]

# Compute cosine similarity
similarities = util.cos_sim(target_emb, candidate_embs)[0]

ranked = sorted(zip(candidates, similarities), key=lambda x: x[1], reverse=True)
for word, score in ranked:
    print(word, score.item())