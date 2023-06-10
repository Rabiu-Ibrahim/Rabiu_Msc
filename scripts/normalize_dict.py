# normalize the lexicons

import unicodedata as ud
import argparse
import os
import sys

def strip_accents(s):
  return ''.join(
    c for c in ud.normalize('NFD', s) if ud.category(c) != 'Mn'
  )

def strip_input(s):
  try:
    return s.strip()
  except:
    return str(s).strip()
  
parser = argparse.ArgumentParser(description='Normalize source and target words.')
parser.add_argument('-i', '--input', required=True, type=str, help='file containing the bilingual lexicons.')
parser.add_argument('-o', '--output_path', required=True, type=str, help="path to store the normalized bilingual lexicons.")
args = parser.parse_args()

if not os.path.isdir(args.output_path):
  try:
    os.makedirs(args.output_path)
  except:
    print('Could not create directory ' + args.output_path)
    sys.exit(-1)

with open(args.input, 'r') as f:
  lines = f.read().splitlines()

print(len(lines))

src = []
tgt = []
for line in lines:
  s, t = line.split('\t')
  src.append(s)
  tgt.append(t)

norm_hau = [strip_accents(h) for h in list(src)]
norm_eng = [strip_accents(e) for e in list(tgt)]

with open('normalized_bilingual_lexicons.txt', 'w') as f:
  for s, t in zip(norm_hau, norm_eng):
    f.write('%s\t%s\n'%(strip_input(s), strip_input(t)))

print(len(lines), len(norm_hau))
print('Normalized bilingual lexicons saved to ' + args.output_path + '/normalized_bilingual_lexicons.txt')