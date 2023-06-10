import os
import pandas as pd
from cupy_utils import get_array_module

from nltk.tokenize import RegexpTokenizer, word_tokenize
from nltk.tag import pos_tag

def topk_mean(m, k, inplace=False):  # TODO Assuming that axis is 1
  xp = get_array_module(m)
  n = m.shape[0]
  ans = xp.zeros(n, dtype=m.dtype)
  if k <= 0:
    return ans
  if not inplace:
    m = xp.array(m)
  ind0 = xp.arange(n)
  ind1 = xp.empty(n, dtype=int)
  minimum = m.min()
  for i in range(k):
    m.argmax(axis=1, out=ind1)
    ans += m[ind0, ind1]
    m[ind0, ind1] = minimum
  return ans / k

def deduplicate_mono_sentences(indir, outdir, max_files):
  
  if not os.path.isdir(outdir):
    os.makedirs(outdir)
  
  files = os.listdir(indir)
  count = 0
  all_lines = set()
  
  for f in files:
    with open(os.path.join(indir, f), 'r') as fp:
      lines = fp.read().splitlines()
    
    lines = [l.strip() for l in lines if l not in all_lines and len(l) > 0 and not l.isspace()]
    all_lines.update(set(lines))
    
    if lines:
      with open(os.path.join(outdir, f), 'w') as fp:
        fp.writelines([l + '\n' for l in lines])
      
      count += 1
      print('saved', f, ': file', count, 'of', max_files)
      
      if count == max_files:
        print(all_lines)
        return

def clean_sentence(sentence):
  sentence = sentence.lower()
  sentence = ''.join(c for c in sentence if ord(c) < 128)
  return sentence

def read_transliterations(file_path, src):
  
  df = pd.read_excel(file_path, header=None, usecols=[1,2], names=['en','ha'])
  if src == 'en':
    df = {en: ha for en, ha in zip(df.en, df.ha)}
  else:
    df = {ha: en for ha, en in zip(df.ha, df.en)}

  return df, len(df)

def get_ner(nlp, sent, src, with_date=False):

  words = []

  if src == 'ha':
    w_ent = ''
    ner_results = nlp(sent)
    if not with_date:
      for ent in ner_results:
        if not 'DATE' in ent['entity']:
          word = ent['word']
          if ord(word[0]) == 9601:
            if w_ent[1:] != '' and w_ent[1:] not in words:
              words.append(w_ent[1:])
            w_ent = word
          else:
            w_ent += word

      if w_ent[1:] not in words:
        words.append(w_ent[1:])

    else:
      for ent in ner_results:
        word = ent['word']
        if ord(word[0]) == 9601:
          if w_ent[1:] != '' and w_ent[1:] not in words:
            words.append(w_ent[1:])
          w_ent = word
        else:
          w_ent += word

      if w_ent[1:] not in words:
        words.append(w_ent[1:])

  else:
    sent = word_tokenize(sent)
    sent = pos_tag(sent)
    if not with_date:
      words = list(set([n for n, e in sent if e in ['NNP']]))
    else:
      words = list(set([n for n, e in sent if e in ['NNP', 'CD']]))

  return words

def remove_punct(sent):

  tokenizer = RegexpTokenizer(r'\w+')
  tokens = []
  
  for token in sent.split(' '):
    token = tokenizer.tokenize(token)
    if token:
      tokens.append(''.join(token))

  return ' '.join([str(t) for t in tokens])