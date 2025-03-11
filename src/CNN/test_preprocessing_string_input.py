import torch
from dataset import charger_corpus, tokenizer_occitan, vectorizer_phrase, padding_liste_phrases, tensorizer_phrase, charger_fasttext

phrase_test = "L'ostal es blanc."

model = charger_fasttext()

phrase_tokenisee = tokenizer_occitan(phrase_test)
print("phrase_tokenisee:", phrase_tokenisee)
phrase_vectorisee = vectorizer_phrase(phrase_tokenisee, model, 56)
print("phrase_vectorisee:", phrase_vectorisee)
phrase_vectorisee_padded = padding_liste_phrases(phrase_vectorisee, 56)
print("phrase_vectorisee_padded:", phrase_vectorisee_padded)
phrase_tensor = tensorizer_phrase(phrase_vectorisee_padded, torch.float32)
print("phrase_tensor:", phrase_tensor)

