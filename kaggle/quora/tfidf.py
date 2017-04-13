import pandas as pd
import numpy as np
from collections import Counter

class tfidf(object):
    def __init__(self, corpus_file_path):
	self.corpus_dict = corpus_file_path
	self.corpus_dict = open(corpus_file_path).read().lower().split('')

    def vec(self, sent):
