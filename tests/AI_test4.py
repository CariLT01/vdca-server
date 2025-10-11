from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer("all-mpnet-base-v2")
context = "The strangeness of entering as ___ the house where she had so long commanded."
choices = ["an investor", "a suppliant", "an authority", "a fanatic"]

context_emb = model.encode(context)
scores = [util.cos_sim(model.encode(context.replace("___", c)), context_emb) for c in choices]
print(scores)