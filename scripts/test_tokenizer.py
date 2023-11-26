import json
from transformers import CLIPTokenizer

tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")

text = "αβ, Γ γ, Δ δ, Ε ε, Ζ ζ, Η η, Θ θ, Ι ι, Κ κ, Λ λ, Μ μ, Ν ν, Ξ ξ, Ο ο, Π π, Ρ ρ, Σ σ/ς, Τ τ, Υ υ, Φ φ, Χ χ, Ψ ψ, Ω ω."

token_dict = json.load(open("SD15/tokenizer/vocab.json", "r"))
# reverse key and value in token_dict because the token ids are value in vocab.json
token_dict = {v: k for k, v in token_dict.items()}


tokens = tokenizer(text, truncation=False, padding="max_length", return_tensors="pt").input_ids

tokens = tokens.tolist()[0]
tokens = [t for t in tokens if t not in [49406, 49407]] #remove start/end/pad tokens
tokens2 = [token_dict[t] for t in tokens]
print(f"text: {text}")
print(f"token ids:{tokens}")
print(f"tokens: {tokens2}")
print(f"length (special tokens removed, max 75): {len(tokens)}, over limit: {len(tokens) > 75}")

