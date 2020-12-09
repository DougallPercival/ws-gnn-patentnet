import os

import nltk

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.corpus import stopwords as nltkstopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer
from nltk.tokenize import sent_tokenize
import gensim

import warnings

warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
warnings.filterwarnings(action='ignore', category=FutureWarning, module='gensim')


class SimilarityCompare():
    def __init__(self, main_doc_text, main_doc_pid, sim_docs_text, sim_docs_type, sim_docs_pid, max_compare=10,
                 score_threshold=0.75, add_stopwords=None):
        """
        Takes in the text from the main document and a list of text from all the similar documents.
        Takes in the PIDs of that main doc and a list for the sim docs.
        Process these test sets, run similarity comparison.
        Take the most similar passages between the main doc and the similar docs.

        Inputs:
        main_doc_text, main_doc_pid - The Text of the main document to be compares against, and the ID
        sim_docs_text, sim_docs_pis - List of text for comparable similaity documents, and their IDs
        sim_docs_type - The type of comparison document being submitted (i.e. DETD or DRWD etc.) Not needed for the main_doc
        max_compare - Default 10 passages. The maximum amount of passages to compare for a given document.
        add_stopwords - allows user to add a set of stopwords to be joined to the main english stopwords
        """
        self.main_doc_pid = main_doc_pid
        self.main_doc_text = main_doc_text
        self.sim_docs_pid = sim_docs_pid
        self.sim_docs_type = sim_docs_type
        self.sim_docs_text = sim_docs_text

        self.d2b_sim_scores = None  # populated by the compare method

        self.max_compare = max_compare
        self.score_threshold = score_threshold

        # set up implementations for tokenization
        self.lemmatizer = WordNetLemmatizer()
        self.regex_tokenizer = RegexpTokenizer(r"\w+")
        self.stopwords = list(set(nltkstopwords.words('english')))

        self.extra_stop_words = ['FIG', 'FIGS', 'invention', 'refer', 'referring', 'now', 'detail', 'description',
                                 'described', 'understood', 'illustrate', 'illustrated', 'depicted', 'embodiment']
        for word in self.extra_stop_words:
            self.stopwords.append(word)

        if add_stopwords is not None:
            for word in add_stopwords:
                self.stopwords.append(word.lower())

    def setup(self):
        """
        Setup the similairty compare
        This needs to output similar passages if they meet a certain threshold
        """
        self.__setup_d2b()
        return

    def compare(self):
        """
        Run the comparison on a document and the similar documents
        M - number of main doc sentences; N - number of target doc sentences
        GENSIM D2B produces a similarty matrix of MxN. GENSIM D2B returns these in index order, with just the similarity scores
        BERT produces a similarity matrix of NxM. BERT returns these in score-order with Tuples containing (index, score)


        """
        for i in range(len(self.sim_docs_text)):
            d2b_sim = self.__compare_doc(self.main_doc_text, self.sim_docs_text[i])
            self.__save_scores(d2b_scores=d2b_sim)

        print("Compared {} documents. Retrieve using getMostSimilar().".format(len(self.sim_docs_text)))
        print("Main doc has {} sentences.".format(len(self.__tokenize_sentences(self.main_doc_text))))
        return

    def getMostSimilar(self, max_compare=None, threshold=None, get="max"):
        """
        Return the most similar passages (sentences) found between the main document and the comparison documents
        User can update the score threshold and the maximum retunring sentences here, if desired
        get can be "max" for retreive only the top n similarities (Default) or "all" to return all similar documents non-sorted.

        Returning Tuple Values at Index:
        0 - Main Doc ID
        1 - Main Doc Passage
        2 - Similar Document ID
        3 - Similar Document Text Type
        4 - Similar Document Passage
        5 - Similarity Score of passages
        6 - Tokenized words for the similar passage

        """
        main_doc_passages = self.__tokenize_sentences(self.main_doc_text)
        most_similar = []
        score_list = self.d2b_sim_scores

        if threshold is None:
            if self.score_threshold is None:
                # default to 0.5
                threshold = 0.5
            else:
                threshold = self.score_threshold

        if max_compare is None:
            if self.max_compare is None:
                # default to 10
                max_compare = 10
            else:
                max_compare = self.max_compare

        # from here, use score_list to capture the most relevant scores
        # the highest level of the list lines up to the number of similar documents
        for i in range(len(self.sim_docs_pid)):
            # the second level of the list is the number of similar doc passages
            sim_doc_passages = self.__tokenize_sentences(self.sim_docs_text[i])
            for j in range(len(sim_doc_passages)):
                # the final level of the list is the number of main doc passages
                for k in range(len(main_doc_passages)):
                    _score = score_list[i][j][k]
                    if _score > threshold:
                        # main doc passage is found in main_doc_passages at value k
                        # similar doc pid, type, text is from i
                        # similar doc passage is from j
                        main_doc_passage = main_doc_passages[k]
                        sim_pid = self.sim_docs_pid[i]
                        sim_type = self.sim_docs_type[i]
                        sim_passage = sim_doc_passages[j]
                        most_similar.append((
                                            self.main_doc_pid, main_doc_passage, sim_pid, sim_type, sim_passage, _score,
                                            self.__tokenize_words(sim_passage)))
                        # you figure it out
        if get == "all":
            return most_similar
        # we sort it for you
        elif get == "max":
            results = sorted(most_similar, key=lambda x: x[5], reverse=True)  # 5 is index of score value
            return results[0:max_compare]

    def __setup_d2b(self):
        """
        If user has specified they want to use_d2b method, this will be called by setup
        """
        self.dictionary = self.__dictionary(self.__tokenizer(self.main_doc_text))
        self.corpus = self.__corpus(self.__tokenizer(self.main_doc_text))
        self.tfidf = self.__tfidf(self.corpus)
        self.similarity_measure = gensim.similarities.Similarity(os.getcwd(), self.tfidf[self.corpus],
                                                                 num_features=len(self.dictionary))
        return

    def __tokenize_sentences(self, doc):
        """
        Given the free text of a document, tokenize into sentences
        """
        return sent_tokenize(doc)

    def __tokenize_words(self, sentence):
        """
        Given the free text of a tokenize sentence, break that down into word tokens
        Drop stopwords, numbers, punctuation. Set to lowercase. Lemmatize.
        """
        tokens = self.regex_tokenizer.tokenize(sentence)
        tokens = [self.lemmatizer.lemmatize(w) for w in tokens]
        tokens = [word for word in tokens if word.isalpha()]
        tokens = [w.lower() for w in tokens]
        tokens = [w for w in tokens if not w in self.stopwords]
        return tokens

    def __tokenizer(self, doc):
        """
        Take the free text of a doc, and break down to a list of word tokens of sentences
        """
        doc_tokens = []
        for sent in self.__tokenize_sentences(doc):
            doc_tokens.append(self.__tokenize_words(sent))

        return doc_tokens

    def __dictionary(self, doc_tokens):
        """
        Create or the gensim dictionary
        """
        return gensim.corpora.Dictionary(doc_tokens)

    def __updated_bow(self, sim_doc):
        """
        Create an updated bag of words based on the main document's dictionary/corpus/tfidf and a similar document
        """
        return self.dictionary.doc2bow(sim_doc)

    def __corpus(self, doc_tokens):
        """
        Create the corpus by extracting doc2bow with the dictionary
        Or update the existing corpus
        """
        return [self.dictionary.doc2bow(gen_doc) for gen_doc in doc_tokens]

    def __tfidf(self, corpus):
        """
        Produce a TFIDF matrix from a corpus
        """
        return gensim.models.TfidfModel(self.corpus)

    def __compare_doc(self, main_doc, sim_doc):
        """
        Run comparison on two specific documents
        """
        d2b_sim_measures = None

        main = self.__tokenizer(main_doc)
        sim_sents = self.__tokenizer(sim_doc)

        d2b_sim_measures = []
        for sent in sim_sents:
            # get the updated BoW for the sim_doc
            sim_bow = self.__updated_bow(sent)
            sim_tfidf = self.tfidf[sim_bow]
            d2b_sim_measures.append(self.similarity_measure[sim_tfidf])

        return d2b_sim_measures

    def __save_scores(self, d2b_scores=None):
        """
        Based on what the user wants to extract (d2b, bert, or both), this will save that data for later viewing
        This function also reformats it so that D2B and GENSIM are in the same format (array with M num_main_sentences x N num_target_sentences)
        Reminder that D2B is MxN and BERT is NxM
        """
        d2b_sim_scores = None

        if d2b_scores is not None:
            # convert d2b scores to a basic list array
            list_d2b = []
            for elem in d2b_scores:
                list_d2b.append(elem.tolist())
            d2b_sim_scores = list_d2b

            if self.d2b_sim_scores is None:
                self.d2b_sim_scores = []
                self.d2b_sim_scores.append(d2b_sim_scores)
            else:
                self.d2b_sim_scores.append(d2b_sim_scores)

        return