# Copyright (C) 2016-2018  Mikel Artetxe <artetxem@gmail.com>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

import embeddings
from cupy_utils import *
from utils import *

import argparse
import collections
import numpy as np
import pandas as pd
import sys
import os
import nltk
from collections import defaultdict

from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

tokenizer = AutoTokenizer.from_pretrained("masakhane/afroxlmr-large-ner-masakhaner-1.0_2.0")
model = AutoModelForTokenClassification.from_pretrained("masakhane/afroxlmr-large-ner-masakhaner-1.0_2.0")
nlp = pipeline("ner", model=model, tokenizer=tokenizer)


def get_translation(src, x, z, cuda, dot, inv_temperature, inv_sample, neighborhood, retrieval, dtype, seed=0, BATCH_SIZE=500):
  # Find translations
  translation = collections.defaultdict(int)

  # NumPy/CuPy management
  if cuda:
    if not supports_cupy():
      print('ERROR: Install CuPy for CUDA support', file=sys.stderr)
      sys.exit(-1)
    xp = get_cupy()
    x = xp.asarray(x)
    z = xp.asarray(z)
  else:
    print('cuda not provided, using cpu.')
    xp = np
  xp.random.seed(seed)

  # Length normalize embeddings so their dot product effectively computes the cosine similarity
  if not dot:
    embeddings.length_normalize(x)
    embeddings.length_normalize(z)

  # Standard nearest neighbor
  if retrieval == 'nn':
    for i in range(0, len(src), BATCH_SIZE):
      j = min(i + BATCH_SIZE, len(src))
      similarities = x[src[i:j]].dot(z.T)
      nn = similarities.argmax(axis=1).tolist()
      for k in range(j-i):
        translation[src[i+k]] = nn[k]
  
  # Inverted nearest neighbor
  elif retrieval == 'invnn':
    best_rank = np.full(len(src), x.shape[0], dtype=int)
    best_sim = np.full(len(src), -100, dtype=dtype)
    for i in range(0, z.shape[0], BATCH_SIZE):
      j = min(i + BATCH_SIZE, z.shape[0])
      similarities = z[i:j].dot(x.T)
      ind = (-similarities).argsort(axis=1)
      ranks = asnumpy(ind.argsort(axis=1)[:, src])
      sims = asnumpy(similarities[:, src])
      for k in range(i, j):
        for l in range(len(src)):
          rank = ranks[k-i, l]
          sim = sims[k-i, l]
          if rank < best_rank[l] or (rank == best_rank[l] and sim > best_sim[l]):
            best_rank[l] = rank
            best_sim[l] = sim
            translation[src[l]] = k
  
  # Inverted softmax
  elif retrieval == 'invsoftmax':
    sample = xp.arange(x.shape[0]) if inv_sample is None else xp.random.randint(0, x.shape[0], inv_sample)
    partition = xp.zeros(z.shape[0])
    for i in range(0, len(sample), BATCH_SIZE):
      j = min(i + BATCH_SIZE, len(sample))
      partition += xp.exp(inv_temperature*z.dot(x[sample[i:j]].T)).sum(axis=1)
    for i in range(0, len(src), BATCH_SIZE):
      j = min(i + BATCH_SIZE, len(src))
      p = xp.exp(inv_temperature*x[src[i:j]].dot(z.T)) / partition
      nn = p.argmax(axis=1).tolist()
      for k in range(j-i):
        translation[src[i+k]] = nn[k]
  
  # Cross-domain similarity local scaling
  elif retrieval == 'csls':
    knn_sim_bwd = xp.zeros(z.shape[0])
    for i in range(0, z.shape[0], BATCH_SIZE):
      j = min(i + BATCH_SIZE, z.shape[0])
      knn_sim_bwd[i:j] = topk_mean(z[i:j].dot(x.T), k=neighborhood, inplace=True)
    for i in range(0, len(src), BATCH_SIZE):
      j = min(i + BATCH_SIZE, len(src))
      similarities = 2*x[src[i:j]].dot(z.T) - knn_sim_bwd  # Equivalent to the real CSLS scores for NN
      nn = similarities.argmax(axis=1).tolist()
      for k in range(j-i):
        translation[src[i+k]] = nn[k]

  return translation

def get_word_types(sent, src, ne_with_date=False):
  words = word_tokenize(sent)

  words_index = defaultdict(list)
  for key, wrd in enumerate(words):
    words_index[wrd].append(key)

  words_index = dict(words_index)
  print('Words indices', words_index)

  ne = get_ner(sent, src, ne_with_date)
  non_ne = [w.lower() for w in words if w not in ne]

  return words_index, non_ne, ne

# def translate_sentence(sent, src, tgt, transliterations, retrieval, ne_with_date=False):

#   words_index, non_ne, ne = get_word_types(sent, src, ne_with_date)

#   print('Named-Entites:', ne)
#   print('Non Named-Entites:', non_ne)

#   src_embed = '../embeddings/en-ha-clwe/supervised/' + src + '_model'
#   tgt_embed = '../embeddings/en-ha-clwe/supervised/' + tgt + '_model'

#   translation = translate_words(src_embed, tgt_embed, non_ne, src_topn=0, trg_topn=0, retrieval=retrieval, inv_temperature=1, inv_sample=None, neighborhood=10, dot=True, encoding='utf-8', seed=0, precision='fp32', cuda=False)

def format_translation(ne, non_ne, translation, words_index, transliterations):

  for i, wrd in enumerate(non_ne):
    try:
      non_ne[i] = translation[str(wrd)]
    except:
      continue

  for i, wrd in enumerate(ne):
    try:
      ne[i] = transliterations[str(wrd)]
    except:
      continue

  for wrd_ne in ne:
    try:
      indices = words_index[str(wrd_ne)]
      for ind in indices:
        non_ne.insert(ind, wrd_ne)
    except:
      continue

  trans = ' '.join([str(c) for c in non_ne])

  return trans

def translate_sentences(files, src_embeddings, trg_embeddings, transliterations_path, src, tgt, output_path, ne_with_date=False, src_topn=0, trg_topn=0, retrieval='nn', inv_temperature=1, inv_sample=None, neighborhood=10, dot=True, encoding='utf-8', seed=0, precision='fp32', BATCH_SIZE=500, cuda=False):
  
  # Choose the right dtype for the desired precision
  if precision == 'fp16':
    dtype = 'float16'
  elif precision == 'fp32':
    dtype = 'float32'
  elif precision == 'fp64':
    dtype = 'float64'

  if not os.path.isdir(output_path):
    try:
      os.makedirs(output_path)
      print('creating output directory: done')
    except:
      print('failed to create output directory: %s' % output_path)
      sys.exit(-1)


  # Read input embeddings
  srcfile = open(src_embeddings, encoding=encoding, errors='surrogateescape')
  trgfile = open(trg_embeddings, encoding=encoding, errors='surrogateescape')
  src_words, x = embeddings.read(srcfile, threshold=int(src_topn), dtype=dtype)
  trg_words, z = embeddings.read(trgfile, threshold=int(trg_topn), dtype=dtype)

  # Build word to index map
  src_word2ind = {word: i for i, word in enumerate(src_words)}
  #trg_word2ind = {word: i for i, word in enumerate(trg_words)}

  transliterations, len_transl = read_transliterations(transliterations_path, src)
  print(len_transl, 'transliterations loaded successfully.')

  # Read the words and compute coverage
  #with open(words_file, encoding=encoding, errors='surrogateescape') as f:
  #  lines = f.readlines()
  #  lines = [line.strip() for line in lines]
  
  for file, sents in files.items():
    trans = []
    for sent in sents:
      words_index, non_ne, ne = get_word_types(sent, src, ne_with_date)

      print('Named-Entites:', ne)
      print('Non Named-Entites:', non_ne)

      non_ne = [n.strip() for n in non_ne]

      oov = set()
      vocab = set()
      src_ind = []
      for word in non_ne:
        try:
          src_ind.append(src_word2ind[word])
          vocab.add(word)
        except KeyError:
          oov.add(word)
      
      oov -= vocab  # If one of the translation options is in the vocabulary, then the entry is not an oov
      print('words in embedding', vocab)
      print('out of vocabulary words', oov)

      translation = get_translation(src_ind, x, z, cuda, dot, inv_temperature, inv_sample, neighborhood, retrieval, dtype, BATCH_SIZE)
      translation = {src: trs for src, trs in zip([src_words[s] for s in translation.keys()], [trg_words[t] for t in translation.values()])}

      trans.append(format_translation(ne, non_ne, translation, words_index, transliterations))

      print('\n\n')

    # save the generated synthetic translations
    df = pd.DataFrame({src: sents, tgt: trans})
    df.to_csv(os.path.join(output_path, file + '.csv'), index=False)

    print('done:', file, '\n\n')

def main():
  # Parse command line arguments
  parser = argparse.ArgumentParser(description='Generating translations using Cross-Lingual Word Embeddings and Transliterations of Nmaed Entities.')
  parser.add_argument('src_embeddings', help='the source language embeddings')
  parser.add_argument('trg_embeddings', help='the target language embeddings')
  parser.add_argument('src_lang', required=True, help='the source language')
  parser.add_argument('trg_lang', required=True, help='the target language')
  parser.add_argument('-f', '--files_dir', required=True, help='the directory containing the files to be translated.')
  parser.add_argument('-t', '--transliterations', required=True, help='the excel, csv or tsv file containing transliterations.')
  parser.add_argument('-o', '--output', default='', help='path to save the translations.')
  parser.add_argument('--max_files', default=0, help='the maximum number of files to be translated.')
  parser.add_argument('--src_topn', default=0, help='number of words to use in source embedding')
  parser.add_argument('--trg_topn', default=0, help='number of words to use in target embedding')
  parser.add_argument('--retrieval', default='nn', choices=['nn', 'invnn', 'invsoftmax', 'csls'], help='the retrieval method (nn: standard nearest neighbor; invnn: inverted nearest neighbor; invsoftmax: inverted softmax; csls: cross-domain similarity local scaling)')
  parser.add_argument('--inv_temperature', default=1, type=float, help='the inverse temperature (only compatible with inverted softmax)')
  parser.add_argument('--inv_sample', default=None, type=int, help='use a random subset of the source vocabulary for the inverse computations (only compatible with inverted softmax)')
  parser.add_argument('-k', '--neighborhood', default=10, type=int, help='the neighborhood size (only compatible with csls)')
  parser.add_argument('--dot', action='store_true', help='use the dot product in the similarity computations instead of the cosine')
  parser.add_argument('--encoding', default='utf-8', help='the character encoding for input/output (defaults to utf-8)')
  parser.add_argument('--seed', type=int, default=0, help='the random seed')
  parser.add_argument('--precision', choices=['fp16', 'fp32', 'fp64'], default='fp32', help='the floating-point precision (defaults to fp32)')
  parser.add_argument('--cuda', action='store_true', help='use cuda (requires cupy)')
  args = parser.parse_args()

  if not os.path.isdir(args.output):
    try:
      os.makedirs(args.output)
    except OSError:
      print('Error creating output directory')
      sys.exit(1)

  files_names = os.listdir(args.files_dir)
  done_files = os.listdir(args.output)
  done_files = [f.replace('.csv', '') for f in done_files]

  files_names = [f for f in files_names if f not in done_files]
  max_files = args.max_files if args.max_files > 0 else len(files_names)

  for n, filename in enumerate(files_names):
    
    files = {}
    trans = []

    # read monolingual text file
    with open(os.path.join(args.files_dir, filename), 'r') as f:
      files[filename] = [remove_punct(l) for l in f.read().splitlines() if l != '']

    if n + 1 == max_files:
      break

  translate_sentences(files, args.src_embeddings, args.trg_embeddings, args.transliterations, args.src_lang, args.trg_lang, args.output, args.max_files, args.src_topn, args.trg_topn, args.retrieval, args.inv_temperature, args.inv_sample, args.neighborhood, args.dot, args.encoding, args.seed, args.precision, args.cuda)

if __name__ == '__main__':
  main()