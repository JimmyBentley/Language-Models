import pandas as pd
import numpy as np
import os
import re
import requests
import time


def get_book(url):
    """
    takes url of 'Plain Text UTF-8' book and returns string
    containing book's contents.

    http://www.gutenberg.org/files/57988/57988-0.txt'
    """
    req = requests.get(url)
    time.sleep(5)
    text = req.text
    text = text.replace('\r\n', '\n')
    start_idx = text.index('START OF THIS PROJECT')
    end_idx = text.index('END OF THIS PROJECT')
    text = text[start_idx:end_idx]
    start_idx = text.index('***')+3
    text = text[start_idx:-4]
    return text

def custom_split (s):
    if re.search(r'[^a-zA-Z0-9]',s) ==None:
        return s
    out = ''
    for c in s:
        if re.match(r'[^a-zA-Z0-9]',c) != None:
            out = out + ' '+ c + ' '
        else:
            out += c
    return out.split()

def tokenize(book_string):
    """
    tokenizes book_string and returns as a list of strings
    """
    pre_split = re.sub(r'\n{2,}',' \x03 \x02 ',book_string)
    ser = pd.Series(pre_split.split())

    if book_string[:2] == '\n\n' and book_string[-2:] == '\n\n' :
        return list(ser.transform(custom_split).explode())[1:-1]
    elif book_string[:2] == '\n\n' :
        return ['\x02'] + list(ser.transform(custom_split).explode())[1:]
    elif book_string[-2:] == '\n\n':
        return list(ser.transform(custom_split).explode())[:-1] + ['\x03']
    else:
        return ['\x02'] + list(ser.transform(custom_split).explode()) + ['\x03']


class UniformLM(object):
    """
    Uniform Language Model class.
    """

    def __init__(self, tokens):
        """
        Initializes a Uniform languange model using a
        list of tokens. It trains the language model
        using 'train' and saves it to an attribute
        self.mdl.
        """
        self.mdl = self.train(tokens)

    def train(self, tokens):
        """
        Trains a uniform language model given a list of tokens.
        The output is a series indexed on distinct tokens, and
        values giving the (uniform) probability of a token occuring
        in the language.
        """
        tokens = pd.Series(tokens)
        return pd.Series(data = 1 / len(tokens.unique()), index = tokens.unique())


    def probability(self, words):
        """
        Gives the probability a sequence of words
        appears under the language model.
        :param: words: a tuple of tokens
        :returns: the probability 'words' appears under the language
        model.
        """
        for w in words:
            if w not in self.mdl.index:
                return 0
        return self.mdl.iloc[0]**len(words)


    def sample(self, M):
        """
        sample selects tokens from the language model of length M, returning
        a string of tokens.
        """
        return ' '.join(np.random.choice(self.mdl.index, size = M))


class UnigramLM(object):

    def __init__(self, tokens):
        """
        Initializes a Unigram language model using a
        list of tokens. Trains the language model
        using 'train' and saves it to an attribute
        self.mdl.
        """
        self.mdl = self.train(tokens)

    def train(self, tokens):
        """
        Trains a unigram language model given a list of tokens.
        The output is a series indexed on distinct tokens, and
        values giving the probability of a token occuring
        in the language.
        """
        out = {}
        for t in tokens:
            if t in out:
                out[t] += 1
            else:
                out[t] = 1
        return pd.Series(out) / len(tokens)

    def probability(self, words):
        """
        probability gives the probabiliy a sequence of words
        appears under the language model.
        :param: words: a tuple of tokens
        :returns: the probability 'words' appears under the language
        model.
        """
        out = 1
        for w in words:
            if w not in self.mdl.index:
                return 0
            else:
                out *= self.mdl.loc[w]
        return out

    def sample(self, M):
        """
        sample selects tokens from the language model of length M, returning
        a string of tokens.
        """
        return ' '.join(np.random.choice(self.mdl.index, size = M, p = self.mdl.values))


class NGramLM(object):

    def __init__(self, N, tokens):
        """
        Initializes a N-gram languange model using a
        list of tokens. It trains the language model
        using 'train' and saves it to an attribute
        self.mdl.
        """

        self.N = N
        self.tokens = tokens
        ngrams = self.create_ngrams(tokens)

        self.ngrams = ngrams
        self.mdl = self.train(ngrams)

        if N < 2:
            raise Exception('N must be greater than 1')
        elif N == 2:
            self.prev_mdl = UnigramLM(tokens)
        else:
            self.prev_mdl = NGramLM(N-1, tokens)

    def create_ngrams(self, tokens):
        """
        takes in a list of tokens and returns a list of N-grams.
        """
        output = []
        for i in np.arange(0, len(tokens)-self.N + 1):
            tpl = tuple(tokens[i:i+self.N])
            output.append(tpl)
        return output

    def train(self, ngrams):
        """
        Trains a n-gram language model given a list of tokens.
        The output is a dataframe with three columns (ngram, n1gram, prob).
        """
        ngram = pd.Series(ngrams)
        n1gram = ngram.transform(lambda x: x[0:self.N-1])

        df = pd.DataFrame()
        df['ngram'] = ngram
        df['n1gram'] = n1gram
        d_ngram = dict(ngram.value_counts())
        d_n1gram = dict(n1gram.value_counts())
        ngram_count = df['ngram'].apply(lambda x: d_ngram[x])
        n1gram_count = df['n1gram'].apply(lambda x: d_n1gram[x])
        df['prob'] = ngram_count/n1gram_count
        return df.drop_duplicates('ngram').reset_index(drop = True)

    def probability(self, words):
        """
        gives the probability a sequence of words appear
        under the language model.
        :param: words: a tuple of tokens
        :returns: the probability `words` appears under the language
        model.
        """
        #new words list
        words = list(pd.Series(words).replace({'\x02': np.NaN, '\x03': np.NaN}).dropna())

        # orignial tokens
        tokenz = self.tokens
        N_current = self.N
        og_obs = []
        while(N_current > 1):
            og_obs.append(NGramLM(N_current, tokenz))
            N_current -= 1
        og_obs.append(UnigramLM(tokenz))
        og_obs = og_obs[::-1] # og_obs is a list of ngram objects from unigram ... ngram

        # updating N-current
        if len(words)<=N_current:
            N_current = len(words)
        else:
            N_current = self.N

        #creating a list of ngram objects for the passed in word, also from unigram to ngram
        new_obs = []
        while(N_current > 1):
            new_obs.append(NGramLM(N_current, words))
            N_current -= 1
        new_obs.append(UnigramLM(words))
        new_obs = new_obs[::-1]

        #initialize output variable
        out = 1

        #unigram - if the sentence has len = 1, return the result from unigram
        out*=og_obs[0].mdl[new_obs[0].mdl.index[0]]
        if len(words) == 1:
            return out

        # bigram to n-1gram
        for i in range(1,len(new_obs)-1):
            n = new_obs[i].mdl.iloc[0]['ngram']
            n1 = new_obs[i].mdl.iloc[0]['n1gram']
            og = og_obs[i].mdl
            og = og[og['ngram']==n]
            if(len(og[og['n1gram']==n1])):
                current_prob = og[og['n1gram']==n1].iloc[0]['prob']
                out*=current_prob
            else:
                out*=0

        #n-1gram for the rest of the sentence
        og = og_obs[-1].mdl
        ns = new_obs[-1].mdl['ngram']
        n1s = new_obs[-1].mdl['n1gram']
        for i in range(len(ns)):
            n = ns.iloc[i]
            n1 = n1s.iloc[i]
            og = og_obs[-1].mdl
            og = og[og['ngram']==n]
            if(len(og[og['n1gram']==n1])):
                current_prob = og[og['n1gram']==n1].iloc[0]['prob']
                out*=current_prob
            else:
                out*=0
        return out

    def sample(self, M):
        """
        selects tokens from the language model of length M, returning
        a string of tokens.
        """
        tokenz = self.tokens
        N_current = self.N
        og_obs = []
        while(N_current > 1):
            og_obs.append(NGramLM(N_current, tokenz))
            N_current -= 1
        og_obs.append(UnigramLM(tokenz))
        og_obs = og_obs[::-1]


        num = 1
        sentence_len = M
        sentence = ['\x02']

        while len(sentence)<sentence_len:
            #different for each iteration until the number reaches self.N-1
            word_sample = og_obs[num].mdl[og_obs[num].mdl['n1gram']==tuple(sentence[-num:])]

            #same for all iterations
            s = list(word_sample['ngram'].transform(lambda tu: tu[-1]))
            p = list(word_sample['prob'])
            if s:
                word = np.random.choice(s, p = p)
                sentence.append(word)
            else:
                sentence.append('\x03')

            # updates counter
            if num<self.N-1:
                num+=1

        return ' '.join(sentence) + ' \x03'
