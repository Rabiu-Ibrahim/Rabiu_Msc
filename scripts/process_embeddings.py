from gensim.models import Word2Vec
import gensim.downloader as gd
import os
import patoolib

EMBED_PATH = 'embeddings/hausa/hauWE-Models'

patoolib.extract_archive(os.path.join(EMBED_PATH, "hauWE CBOW.rar"), outdir=EMBED_PATH)
patoolib.extract_archive(os.path.join(EMBED_PATH, "hauWE SG.rar"), outdir=EMBED_PATH)

ha_model = Word2Vec.load(os.path.join(EMBED_PATH, 'hauWE CBOW/model'))
ha_model.wv.save_word2vec_format('embeddings/ha_model.txt', binary=False)

en_model = gd.load('word2vec-google-news-300')
en_model.wv.save_word2vec_format('embeddings/en_model_google_news_300.txt', binary=False)

# extract only top 100,000 words
with open('embeddings/en_model_google_news_300.txt', 'r') as f:
  vec = f.readlines()

# insert dimensions at index 0
vec.insert(0, '100000 300')

with open('embeddings/en_model.txt', 'w') as f:
  f.writelines(vec[:100002])

print('Saved first 100,000 word vectors.')