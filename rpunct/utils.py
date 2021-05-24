# -*- coding: utf-8 -*-
# 💾⚙️🔮

__author__ = "Daulet N."
__email__ = "daulet.nurmanbetov@gmail.com"

def prepare_unpunct_text(text):
    """
    Given a text, normalizes it to subsequently restore punctuation
    """
    formatted_txt = text.replace('\n', '').strip()
    formatted_txt = formatted_txt.lower()
    formatted_txt_lst = formatted_txt.split(" ")
    punct_strp_txt = [strip_punct(i) for i in formatted_txt_lst]
    normalized_txt = " ".join([i for i in punct_strp_txt if i])
    return normalized_txt

def strip_punct(wrd):
    """
    Given a word, strips non aphanumeric characters that precede and follow it
    """
    if not wrd:
        return wrd
    
    while not wrd[-1:].isalnum():
        if not wrd:
            break
        wrd = wrd[:-1]
    
    while not wrd[:1].isalnum():
        if not wrd:
            break
        wrd = wrd[1:]
    return wrd
