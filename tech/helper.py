import pandas as pd
import re
import string
import pickle

df_special_char = pd.read_csv("./data/spesial_characters_HTML.csv", encoding='ISO-8859-1')
df_alay_map = pd.read_csv("./data/new_kamusalay.csv", encoding='ISO-8859-1')
df_id_stopword = pd.read_csv("./data/stopwordbahasa.csv", encoding='ISO-8859-1')

def remoji(sentence):
    temp = []
    for i in sentence.split(" "):
        if "\\x" not in i:
            temp.append(i)
    result = " ".join(temp)
    return result

def char_replacer(text):
    text = re.sub('\n',' ',text)
    text = re.sub('RT',' ',text)
    text = re.sub('USER',' ',text)
    text = re.sub('URL',' ',text)
    text = re.sub('Retweeted',' ',text)
    text = re.sub('((www\.[^\s]+)|(https?://[^\s]+)|(http?://[^\s]+))',' ',text)
    return text

def remspchar(sentence):
    text = ' '.join(['' if word in df_special_char else word for word in sentence.split(' ')])
    st = re.sub('  +', ' ', text)
    result = st.strip()
    return result

def number_replacer(text):
    return str(text).replace(r'\d+','')

def space_replacer(text):
    return str(text).replace('\n','')

def alay_removal(text):
    return ' '.join([df_alay_map[word] if word in df_alay_map else word for word in text.split(' ')])

def lowercase(sentence : str):
    return sentence.lower()

def remove_punctuation(text):
    punctuationfree= [i for i in text if i not in string.punctuation]
    word_punct = ''.join(punctuationfree)
    return word_punct

def stopper(text):
    text = ' '.join(['' if word in df_id_stopword.stopword.values else word for word in text.split(' ')])
    text = re.sub('  +', ' ', text)
    text = text.strip()
    return text

def helper(sentence : str):
    functioner = [
        remoji,
        char_replacer,
        remspchar,
        number_replacer,
        space_replacer,
        lowercase,
        remove_punctuation,
        alay_removal,
        stopper
    ]
    st = sentence
    for f in functioner:
        result = f(st)
    return st

def token(path : str):
    with open(path, "rb") as handle:
        tokener = pickle.load(handle)
    return tokener