import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords
from pyemd import emd




class Sentence_Topic_Similarity():
    """
    Class object to perform sentence-topic similarity calculation.
    Can output clusters ofsentences related to a topic, or the classification of each sentence's topic.
    """
    
    
    def __init__(self, w2v, topics, original_text):
        self.w2v = w2v
        self.topics = topics
        self.original_text = original_text  
        
        
        
        
    def tokenize(self):
        """
        Get word tokens for each sentence and topic
        """
        
        self.tagged_topics = input_pos(self.topics) #Tagged topics after removing unuseful words
        self.sentences = get_content(self.original_text) #Variable for word-tokenized sentences
        return "Topic and original text tokenized"
        
        
        
        def remove_stopwords(tokenized_words):
            """Remove the stop words, such as is, are, then, and, etc."""

            stop_words = set(stopwords.words('english')) 
            removed = [w for w in tokenized_words if not w in stop_words] 
            return removed


        def keep_useful(tokenized_words):
            """Keep nouns, adjectives, verbs and adverbs only"""


            is_useful = lambda pos: pos[:2] in ["JJ","NN","VB","RB"]
            useful = [word for (word, pos) in nltk.pos_tag(tokenized_words) if is_useful(pos)] 
            return useful
        
        
        def get_content(original_text):
            """Remove stop words and return the word tokens for each sentence"""

            sentences = nltk.sent_tokenize(original_text, language = 'english')
            tagged_sentences = []
            for sentence in sentences:
                tokenized_words = nltk.word_tokenize(sentence)
                clear_tokens = remove_stopwords(tokenized_words)
                clear_tokens = keep_useful(clear_tokens)
                clear_tokens = remove_oov(clear_tokens, w2v)
                tagged_sentences.append(clear_tokens)
            return tagged_sentences
        
        
        def input_pos(topics):
            """Tag the input topics"""

            tagged_topics = []
            for topic in topics:
                topic_words = nltk.word_tokenize(topic)
                topic_words = remove_stopwords(topic_words)
                topic_words = keep_useful(topic_words)
                topic_words = remove_oov(topic_words, w2v)
                tagged_topics.append(topic_words)
            return tagged_topics
        
        
        def remove_oov(word_tokens, w2v):
            """Remove OOVs"""

            for token in word_tokens:
                try:
                    w2v[token]
                except KeyError:
                    word_tokens.remove(token)
                else:
                    continue

            return word_tokens
        
        
        
        
    def similarity(self, method = "cos_sim"):
        """
        Compute the similarity scores and construct dataframe of each sentence with regard to each topic.
        Similarity metrics include:
        "cos_sim": cosine similarity
        "euclidean": euclidean distance
        "manhattan": manhattan distance
        "wmd": word mover's distance
        
        
        Parameters
        -----------
        method: The similarity metric for sentence classification
        
        Output
        -------
        The similarity score table
        """
        
        assert method in {"cos_sim", "euclidean", "manhattan", "wmd"}
        
        sent_list = nltk.sent_tokenize(self.original_text, language = 'english')
        
        #Construct a dataframe for similarity scores for each topic with each sentence.
        similarity_score = np.zeros((len(self.sentences), len(self.topics)), dtype = float)
        similarity_score = pd.DataFrame(similarity_score, columns = self.topics, index = sent_list)
        
        #Fill in the dataframe with the corresponding scores and method
        for topic_num in range(len(self.topics)):
            score = []
            for sentence in self.sentences:
                if method in {"cos_sim", "euclidean", "manhattan"}: 
                    score.append(avg_w2w_sim(self.tagged_topics[topic_num], sentence, w2v = self.w2v, measure = method))
                elif method == "wmd":
                    score.append(wmdistance(self.tagged_topics[topic_num], sentence, w2v = self.w2v))
            similarity_score[self.topics[topic_num]] = score
        
        #Return the normalized result
        return (similarity_score - similarity_score.min())/(similarity_score.max() - similarity_score.min())

    
        
        def avg_w2w_sim(topic, sentence, w2v, measure = "cos_sim"):
            """
            General function that returns the similarity score of a topic and a sentence.
            The similarity score is the average similarity between each word in the sentence and each word in the topic.
            This should be done after cleaning all the unuseful words.
            """

            assert measure in {"cos_sim", "euclidean", "manhattan"}

            score = [] #A list that store the sentence-level similarity
            for word in topic:
                sim_list = [] #A list that store the word-level similarity
                u = w2v[word]
                for aword in sentence:
                    v = w2v[aword]
                    if measure == "cos_sim":
                        sim = cos_sim(u,v)
                    elif measure == "euclidean":
                        sim = eucli_dist(u,v)
                    elif measure == "manhattan":
                        sim = manh_dist(u,v)
                    sim_list.append(sim) #Append the similarity of a word in sentence and a word in topic
                score.append(np.mean(sim_list)) #The similarity between a sentence and one word of a topic

            return np.mean(score)
        
    
        def cos_sim(u,v):
            """
            Cosine similarity calculation
            """

            numerator_ = u.dot(v)
            denominator_= np.sqrt(np.sum(np.square(u))) * np.sqrt(np.sum(np.square(v)))
            return 0.5-(0.5*numerator_/denominator_)
    
    
        def eucli_dist(u,v):
            """
            L2 norm of v-u
            """

            return np.linalg.norm(v-u)

        
        def manh_dist(u,v):
            """
            L1 norm of v-u
            dist = 	sum(abs(v-u))
            """

            return np.linalg.norm(v-u, ord = 1)
        
        
        def wmdistance(doc1, doc2 ,w2v):
            """
            Calculation of Word Mover's Distance.
            Rely on pyemd package to compute the Earth Mover's Distance.
            Inputs are two tokenized documents (sentences).
            """

            set1 = set(doc1)
            set2 = set(doc2)

            #search wmd in the page
            vocab_set = set1 | set2
            vocab_len = len(vocab_set)

            if vocab_len == 1:
                return 0.0 #The docs contain the same word token

            distance_matrix = np.zeros((vocab_len, vocab_len))
            for i, t1 in enumerate(vocab_set):
                if t1 not in set1:
                    continue

                for j, t2 in enumerate(vocab_set):
                    if t2 not in set2 or distance_matrix[i, j] != 0.0:
                        continue

                    # Compute Manhattan distance between word vectors.
                    distance_matrix[i, j] = distance_matrix[j, i] = np.linalg.norm(w2v[t2] - w2v[t1], ord = 1) 
                    #distance_matrix[i, j] = distance_matrix[j, i] = np.sqrt(np.sum((w2v[t1] - w2v[t2])**2))

            if np.sum(distance_matrix) == 0.0:
                # `emd` gets stuck if the distance matrix contains only zeros.
                logger.warning("Distance matrix is all zeros.")
                return float("inf")

            def nbow(doc):
                nbow = np.zeros(vocab_len)
                dictionary =  dict(zip([word for word in vocab_set],[i for i in range(vocab_len)]))
                for word in doc:
                    nbow[dictionary[word]] += 1
                nbow = np.array([x/len(doc) for x in nbow])
                return nbow

            # Compute nBOW representation of documents.
            d1 = nbow(doc1)
            d2 = nbow(doc2)

            # Compute WMD.
            return emd(d1, d2, distance_matrix)


        
        
    def classify_sentence(self, method = "cos_sim", threshold = None, scores = False):
        """
        Classify each sentence to a input topic.
        Will only consider the highest similarity score of a sentence that pass the threshold.
        Not all sentence will be classified depending on the value of threshold.
        
        
        Parameters
        -----------
        method: The similarity metric for sentence classification
        threshold: The threshold for classification, from 0 to 1
        scores: True to return the entire similarity score table with the classification result
        
        Output
        -------
        The classification result (if score=True, return the entire similarity score table). 
        """
        
        similarity_score = self.similarity(method)
        
        #Obtain the max similarity
        similarity_score["Classification"] = similarity_score.idxmin(axis = 1)
        
        #Replace all the classifications that do not pass the threshold with NaN
        if threshold:
            assert threshold <= 1 and threshold >= 0
            similarity_score["Classification"].loc[similarity_score.min(axis = 1) > threshold] = None
        
        #Return the classification results (and the scores)
        if scores:
            return similarity_score
        else:
            return similarity_score["Classification"]
        

        
    def find_topic_sentence(self, method = "cos_sim",threshold = 0.5):
        """
        Find the sentences that related to topic.
        Return all the sentences that pass the similarity threshold of the topics.
        Not all topics will be assigned sentences depending on the value of threshold.
        
        
        Parameters
        -----------
        method: The similarity metric for sentence classification
        threshold: The threshold for eligible topic sentence
        
        
        Output
        -------
        pass_threshold: A dictionary of topics and their corresponding sentences
        no_category: Uncategorized sentences (sentences that do not belong to any topics)
        """
        
        similarity_score = self.similarity(method)
        assert threshold <= 1 and threshold >= 0
        
        #Dictionary for topic sentences and list for uncategorized sentences
        topic_sentences = {}
        no_category = []
        
        #Add passed/failed sentences to their corresponding iterables
        for i in range(len(similarity_score.columns)):
            topic_sentences[similarity_score.columns[i]] = similarity_score.loc[similarity_score.iloc[:,i] <= threshold].index.tolist()
            no_category.append(similarity_score.loc[similarity_score.iloc[:,i] > threshold].index.tolist())
        
        #Find the intersection of uncategorzed sentences of all the topics, i.e, the sentences that do not belong to any topic
        no_category = set(no_category[0]).intersection(* no_category[1:])
        
        return topic_sentences, no_category
