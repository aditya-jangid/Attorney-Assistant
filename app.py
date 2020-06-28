import requests
import nltk
import os
import tempfile
import numpy as np
import pandas as pd
import re
import sumy
import networkx as nx
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer
from num2words import num2words
from io import StringIO
from nltk.tokenize import sent_tokenize
from sklearn.metrics.pairwise import cosine_similarity
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from pdfminer.pdfdocument import PDFDocument
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.pdfpage import PDFPage
from pdfminer.pdfparser import PDFParser
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lex_rank import LexRankSummarizer
from sumy.summarizers.lsa import LsaSummarizer

# tokenizer=RegexpTokenizer(r'\w+')
stop_words = set(stopwords.words("english"))
# stemmer=PorterStemmer()
# lemmatizer=WordNetLemmatizer()
# stop_words=set(stopwords.words("english"))
tempdirectory = tempfile.gettempdir()

app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def index():
    errors = []
    results = {}
    methodSelected = request.form.get('methodSelected')

    if request.method == "POST":
        try:
            output_string = StringIO()
            file = request.files['file']
            filename = secure_filename(file.filename)
            file.save(os.path.join(tempdirectory, filename))
            with open(os.path.join(tempdirectory, filename), 'rb') as in_file:
                parser = PDFParser(in_file)
                doc = PDFDocument(parser)
                rsrcmgr = PDFResourceManager()
                codec = 'utf-8'
                device = TextConverter(
                    rsrcmgr, output_string, codec=codec, laparams=LAParams())
                interpreter = PDFPageInterpreter(rsrcmgr, device)
                for page in PDFPage.create_pages(doc):
                    interpreter.process_page(page)

            textImport = ''
            textImport = output_string.getvalue()
            # os.remove(filename)
            # file=open(os.path.join(tempdirectory, filename), encoding="utf-8")
            # textImport = file.read()
        except:
            errors.append(
                "Only PDF supported"
            )
            return render_template('index.html', errors=errors)
        if textImport:
            if methodSelected == 'textrank':
                nltk.data.path.append('./nltk_data/')  # set the path
                # tok = tokenizer.tokenize(textImport)    #word tokenization
                # result = [i for i in tok if not i in stop_words]    #stop word removal
                # final=[""]
                # for word in result:
                #     final.append(stemmer.stem(word))   #stemming

                # final2=[""]
                # for word in final:
                #     final2.append(lemmatizer.lemmatize(word))   #lemmatization
                # print(final2)
                # results = final2

                sentences = []
                sentences.append(sent_tokenize(textImport))
                sentences = [y for x in sentences for y in x]
                clean_sentences = pd.Series(
                    sentences).str.replace("[^a-zA-Z]", " ")
                clean_sentences = [s.lower() for s in clean_sentences]

                def remove_stopwords(sen):
                    sen_new = " ".join([i for i in sen if i not in stop_words])
                    return sen_new

                # remove stopwords from the sentences
                clean_sentences = [remove_stopwords(
                    r.split()) for r in clean_sentences]

                # Extract word vectors
                word_embeddings = {}
                f = open('glove.6B.100d.txt', encoding='utf-8')
                for line in f:
                    values = line.split()
                    word = values[0]
                    coefs = np.asarray(values[1:], dtype='float32')
                    word_embeddings[word] = coefs
                f.close()

                sentence_vectors = []
                for i in clean_sentences:
                    if len(i) != 0:
                        v = sum([word_embeddings.get(w, np.zeros((100,)))
                                 for w in i.split()])/(len(i.split())+0.001)
                    else:
                        v = np.zeros((100,))
                    sentence_vectors.append(v)

                len(sentence_vectors)

                # similarity matrix
                sim_mat = np.zeros([len(sentences), len(sentences)])

                for i in range(len(sentences)):
                    for j in range(len(sentences)):
                        if i != j:
                            sim_mat[i][j] = cosine_similarity(sentence_vectors[i].reshape(
                                1, 100), sentence_vectors[j].reshape(1, 100))[0, 0]

                nx_graph = nx.from_numpy_array(sim_mat)
                scores = nx.pagerank(nx_graph)

                ranked_sentences = sorted(
                    ((scores[i], s) for i, s in enumerate(sentences)), reverse=True)

                # Specify number of sentences to form the summary
                sn = 10
                temp = []
                # # Generate summary
                for i in range(sn):
                    temp.append(ranked_sentences[i][1])

                results = temp

            if methodSelected == 'lexrank':
                print('lexrank selected')
                ParsedOutputLexrank = PlaintextParser.from_string(
                    textImport, Tokenizer("english"))
                summarizer = LexRankSummarizer()
                summaryOutputLexrank = summarizer(
                    ParsedOutputLexrank.document, 10)

                for sentence in summaryOutputLexrank:
                    print(sentence)

                results = ''.join(map(str, summaryOutputLexrank))

            if methodSelected == 'lsa':
                print('lsa selected')
                ParsedOutputLexrank = PlaintextParser.from_string(
                    textImport, Tokenizer("english"))
                summarizer_lsa = LsaSummarizer()
                summaryOutputLSA = summarizer_lsa(
                    ParsedOutputLexrank.document, 10)

                # for sentence in summaryOutputLSA:
                #     print(sentence)

                results = ''.join(map(str, summaryOutputLSA))

        output_string.close()

    return render_template('index.html', errors=errors, results=results)
