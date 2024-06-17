'''
Parallel data filtering hard rules, including:
1. duplicated sentences remove
2. same source and target sentences remove
3. sentences with '/', '|', '-' > 5
4. setnences with digtial numbers/characters > 0.5
5. sentences contains word composed by more than 40 charcters
6. sentences with average characs for word > 20 or <4
7. sentences with punctuations > 15
8. sentences with punctuations/characters > 0.5
9. src punctuations/tgt punctuations > 5 or 1/5
10. sentences with html address and html tags
11. optional: non english characters > 0.25
12. optional: src characters / tgt characters > 3 or 1/3
'''
import sys
import re
import argparse
from string import punctuation
from zhon.hanzi import punctuation as punc_zh
from collections import Counter

from nltk.corpus import stopwords
import string
stop = set(stopwords.words('english') + list(string.punctuation))
numeric = set(list(string.digits))

parser = argparse.ArgumentParser()
parser.add_argument('src', help='source file')
parser.add_argument('tgt', help='target file')
parser.add_argument('--soft_html', action='store_true', default=True, help='whether to use soft version only to remove html tag, not the sentence')
args = parser.parse_args()
f1 = args.src
f2 = args.tgt

fw_dirty = open(f"{f1}.dirty", 'w', encoding='utf-8')

min_tok = 10
max_top = 150
avg_word_len_lb = 2
avg_word_len_ub = 20

def write_dirty(error_type, src, tgt, src_error=None, tgt_error=None):
  if src_error is not None and tgt_error is not None:
    fw_dirty.write("[{}]: ||| {} ||| {} ||| {} ||| {}\n".format(
      error_type, src.strip(), tgt.strip(), str(src_error), str(tgt_error))
    )
  else:
    fw_dirty.write("[{}]: ||| {} ||| {}\n".format(
      error_type, src.strip(), tgt.strip())
    )
    
# Duplicated sentences remove
def dup_remove(x_in, y_in):
  tok = 'lijun_wu'
  all_lines = [x.strip() for x in x_in]
  for idx, sent in enumerate(y_in):
    all_lines[idx] += (tok + sent.strip())  # [src+tok+tgt]
  all_lines = list(set(all_lines))  # make as set

  x_out = []
  y_out = []
  for sent in all_lines:
    segs = sent.split(tok)
    x_out.append(segs[0])
    y_out.append(segs[1])
  assert len(x_out) == len(y_out)
  print('After removing duplicated sentences, remain %i pairs' % len(x_out))
  return x_out, y_out

# Same source and target sentence remove
def src_tgt_same_remove(x_in, y_in):
  x_out = []
  y_out = []
  for (x,y) in zip(x_in, y_in):
    if x.strip() == y.strip():
      write_dirty("src_tgt_same", x, y)
      continue
    x_out.append(x.strip())
    y_out.append(y.strip())

  assert len(x_out) == len(y_out)
  print('After removing same source and target sentence, remain %i pairs' % len(x_out))
  return x_out, y_out

# Sentence words number remove
def sentence_word_num_remove(x_in, y_in):

  def check_word_num(sent):
    segs = sent.strip().split()
    if len(segs) < min_tok or len(segs) > max_top:
      return False
    return True

  x_out = []
  y_out = []

  for (x, y) in zip(x_in, y_in):
    if check_word_num(x) and check_word_num(y):
      x_out.append(x.strip())
      y_out.append(y.strip())
    else:
      write_dirty("word_num", x, y)

  assert len(x_out) == len(y_out)
  print('After removing sentences with too less or too many words, reamin %i pairs' % len(x_out))
  return x_out, y_out


# Sentence pair words ratio exceeded remove
def sentence_words_ratio_remove(x_in, y_in):
  x_out = []
  y_out = []

  for (x, y) in zip(x_in, y_in):
    m_x = len(x.strip().split())
    m_y = len(y.strip().split())

    if m_x / m_y > 3.0 or m_y / m_x > 3.0:
      write_dirty("word_raiot", x, y, m_x, m_y)
      continue
    x_out.append(x.strip())
    y_out.append(y.strip())

  assert len(x_out) == len(y_out)
  print('After removing sentence pair exceeds length ratio, reamin %i pairs' % len(x_out))
  return x_out, y_out

# Specific punctuation number exceeded sentence remove
def specfic_punc_remove(x_in, y_in):

  def hot_fix_filter(sent):
    sent = sent.strip()
    '''
    if sent.count("/")  > 5:
      return False
    if sent.count("|") > 5:
      return False 
    if sent.count("-") > 5:
      return False
    '''
    if len(re.findall("[\d\-\|/]", sent)) / len(sent) > 0.5:
      return False
    return True

  x_out = []
  y_out = []

  for (x, y) in zip(x_in, y_in):
    if hot_fix_filter(x) and hot_fix_filter(y):
      x_out.append(x.strip())
      y_out.append(y.strip())
    else:
      write_dirty("specific_punc", x, y)

  assert len(x_out) == len(y_out)
  print('After removing sentences with too many specific punctuations, reamin %i pairs' % len(x_out))
  return x_out, y_out


# Characters condition remove
def characs_remove(x_in, y_in):

  def filter_by_len(sent):
    segs = sent.strip().split()
    for x in segs:
      if len(x) > 40:
        return False
    
    # added
    '''
    m_char = sum([len(x) for x in segs])
    m_word = len(segs)
    ratio = m_char * 1. / (m_word + 1e-9)
    if ratio > avg_word_len_ub or ratio < avg_word_len_lb:
      return False
    '''

    return True

  x_out = []
  y_out = []

  for (x, y) in zip(x_in, y_in):
    if filter_by_len(x) and filter_by_len(y):
      x_out.append(x.strip())
      y_out.append(y.strip())
    else:
      write_dirty("characs_remove", x, y)

  assert len(x_out) == len(y_out)
  print('After removing sentence with characters condition, remain %i pairs' % len(x_out))
  return x_out, y_out


# Punctuation condition remove
def punctuation_remove(x_in, y_in):
  x_out = []
  y_out = []

  count_func = lambda l1,l2: sum([1 for x in l1.split() if x in l2])

  punctuation_set = set(punctuation + punc_zh)
  #punctuation_set.remove('@') # bpe
  punctuation_set.add('&quot;')
  punctuation_set.add('&apos;')
  for (x, y) in zip(x_in, y_in):
    m_punc_x = count_func(x.strip(), set(punctuation_set))
    m_punc_y = count_func(y.strip(), set(punctuation_set))
    if (#m_punc_x / (len(x.strip()) + 1e-9) > 0.5 
        #or m_punc_y / (len(y.strip()) + 1e-9) > 0.5 
        m_punc_x > 10 
        or m_punc_y > 10
        #or m_punc_x / (m_punc_y + 1e-9) > 5 
        #or m_punc_y/ (m_punc_x + 1e-9) > 5
    ):
      #fw_dirty.write("[punc_remove]: |||"+x+" ||| "+y+" ||| "+str(m_punc_x)+" ||| "+str(m_punc_y)+"\n")
      write_dirty("punc_remove", x, y, m_punc_x, m_punc_y)
      continue
    x_out.append(x.strip()) 
    y_out.append(y.strip())

  assert len(x_out) == len(y_out)
  print('After removing sentences with too much punctuations, remain %i pairs' % len(x_out))
  return x_out, y_out


# Html address or tags contained sentence remove
def html_remove(x_in, y_in):
  x_out = []
  y_out = []

  def filter_by_html(sentence):
    sen = sentence.strip()
    detector = re.compile('<.*?>')
    html_tag = re.findall(detector, sen)
    if html_tag or 'https://' in sen or 'http://' in sen:
      return False
    return True

  def soft_filter_by_html(sent):
    sent = sent.strip()
    detector = re.compile('<.*?>')
    sent = re.sub(detector, '', sent)
    sent = re.sub('https?:\/\/.*[ \r\n]', '', sent, flags=re.MULTILINE)
    return sent

  for (x, y) in zip(x_in, y_in):
    if args.soft_html:
      x_out.append(soft_filter_by_html(x))
      y_out.append(soft_filter_by_html(y))
    else:
      if filter_by_html(x) or filter_by_html(y):
        x_out.append(x.strip())
        y_out.append(y.strip())
      else:
        write_dirty("html", x, y)

  assert len(x_out) == len(y_out)
  print('After removing sentences with html address or tags, remain %i pairs' % len(x_out))
  return x_out, y_out


# From Teacher Xia, special chars (hard to print)
def special_char_remove(x_in, y_in):
  x_out = []
  y_out = []

  for (x, y) in zip(x_in, y_in):
    if r"\x" in x or r"\x" in y:
      write_dirty("special_char", x, y)
      continue
    x_out.append(x.strip())
    y_out.append(y.strip())

  assert len(x_out) == len(y_out)
  print('After removing sentences with special characters, remain %i pairs' % len(x_out))
  return x_out, y_out


# Optional: Src/tgt chars ratio exceeded remove
def characs_sum_remove(x_in, y_in):
  x_out = []
  y_out = []

  for (x, y) in zip(x_in, y_in):
    segs_x = x.strip().split()
    m_char_x = sum([len(x) for x in segs_x])

    segs_y = y.strip().split()
    m_char_y = sum([len(y) for y in segs_y])
    
    if m_char_x ==0:
      m_char_x = 1
    if m_char_y == 0:
      m_char_y = 1

    if m_char_x / m_char_y > 10 or m_char_y / m_char_x > 10:
      write_dirty("char_ratio", x, y, m_char_x, m_char_y)
      continue
    x_out.append(x.strip())
    y_out.append(y.strip())

  assert len(x_out) == len(y_out)
  print('After removing setnence with characters ratio condition, remain %i pairs' % len(x_out))
  return x_out, y_out


# Optional: Lattin letter contained sentence remove
def lattin_remove_in_target(x_in, y_in):
  
  def count_lattin(sent):
    if len(re.findall("[a-zA-Z]", sent)) / len(sent) > 0.25:
      return False
    return True

  x_out = []
  y_out = []
  for (x, y) in zip(x_in, y_in):
    if count_lattin(y.strip()):
      x_out.append(x.strip())
      y_out.append(y.strip())
    else:
      write_dirty("lattin_remove", x, y)

  assert len(x_out) == len(y_out)
  print('After removing sentences with too much lattin characs, remian %i pairs' % len(x_out))
  return x_out, y_out

# added
def emptyline_remove(x_in, y_in):
  x_out = []
  y_out = []
  for (x, y) in zip(x_in, y_in):
    if x.strip() != '' and y.strip() != '':
      x_out.append(x.strip())
      y_out.append(y.strip())
    else:
      write_dirty("emptyline_remove", x, y)
  assert len(x_out) == len(y_out)
  print('After removing empty source or target sentences, remian %i pairs' % len(x_out))
  return x_out, y_out


def token_repeating_filter(x_in, y_in, thres=0.3):
    x_out = []
    y_out = []
    for (x, y) in zip(x_in, y_in):
      x_tok = x.split()
      y_tok = y.split()

      xcounts = Counter(x_tok)
      xratio = (max(xcounts.values()) / len(x_tok))

      #ycounts = Counter(y_tok)
      #yratio = (max(ycounts.values()) / len(y_tok))

      if max(xcounts.values()) > 1 and xratio > thres:
        write_dirty(f"token repeat > {thres}", x, y)
      else:
        x_out.append(x.strip())
        y_out.append(y.strip())

    assert len(x_out) == len(y_out)
    print('After removing token repeating source or target sentences, remian %i pairs' % len(x_out))
    return x_out, y_out


def mostly_uninformative_filter(x_in, y_in, a=0.3, b=0.8):
  x_out = []
  y_out = []
  for (x, y) in zip(x_in, y_in):
    x_tok = x.split()
    xratio = (len([x for x in x_tok if x not in stop]) / len(x_tok))

    if a <= xratio <= b:
      x_out.append(x.strip())
      y_out.append(y.strip())
    else:
      write_dirty(f"mostly_uninformative > {a} < {b}", x, y)

  assert len(x_out) == len(y_out)
  print('After removing mostly_uninformative source or target sentences, remian %i pairs' % len(x_out))
  return x_out, y_out


def numeric_filter(x_in, y_in, thres=0.2):
  x_out = []
  y_out = []
  age_re = re.compile('[\d\s]')
  for (x, y) in zip(x_in, y_in):
    # xratio = (len([x for x in x_tok if x not in numeric]) / len(x_tok))
    xratio = len(age_re.sub("", x)) / len(x)

    if xratio > thres:
      x_out.append(x.strip())
      y_out.append(y.strip())
    else:
      write_dirty(f"numeric_filter < {thres}", x, y)

  assert len(x_out) == len(y_out)
  print('After removing more numeric source or target sentences, remian %i pairs' % len(x_out))
  return x_out, y_out


filter_1 = []
filter_2 = []

fr_1 = open(f1, "r", encoding="utf-8") 
fr_2 = open(f2, "r", encoding="utf-8") 

#f1_all_lines = fr_1.readlines()
#f2_all_lines = fr_2.readlines()
filter_1 = fr_1.readlines()
filter_2 = fr_2.readlines()
print('read all pairs into memory', len(filter_1))

filter_1, filter_2 = dup_remove(filter_1, filter_2)
filter_1, filter_2 = src_tgt_same_remove(filter_1, filter_2)

filter_1, filter_2 = sentence_word_num_remove(filter_1, filter_2)
filter_1, filter_2 = sentence_words_ratio_remove(filter_1, filter_2)
filter_1, filter_2 = specfic_punc_remove(filter_1, filter_2)

filter_1, filter_2 = characs_remove(filter_1, filter_2)

filter_1, filter_2 = special_char_remove(filter_1, filter_2)
filter_1, filter_2 = punctuation_remove(filter_1, filter_2)
filter_1, filter_2 = html_remove(filter_1, filter_2)
filter_1, filter_2 = characs_sum_remove(filter_1, filter_2)

# put it at last
filter_1, filter_2 = token_repeating_filter(filter_1, filter_2)
filter_1, filter_2 = mostly_uninformative_filter(filter_1, filter_2)
filter_1, filter_2 = numeric_filter(filter_1, filter_2)

# only for en-zh
#filter_1, filter_2 = lattin_remove_in_target(filter_1, filter_2)

filter_1, filter_2 = emptyline_remove(filter_1, filter_2)

fr_1.close()
fr_2.close()


fw_1 = open(f1 + ".clean_filt_ms", "w", encoding="utf8")
fw_2 = open(f2 + ".clean_filt_ms", "w", encoding="utf8")

assert len(filter_1) == len(filter_2)
print('After all filtering rules, remain %i pairs' % len(filter_1))

for x in filter_1:
  print(x, file=fw_1)

for y in filter_2:
  print(y, file=fw_2)

fw_1.close()
fw_2.close()

