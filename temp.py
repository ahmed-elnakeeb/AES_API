import transformers

model=transformers.AutoModel.from_pretrained("albert-base-v2")
model.save_pretrained("albert-base-v2")