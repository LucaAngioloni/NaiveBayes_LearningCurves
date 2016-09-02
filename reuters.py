"""
Copyright 2016 Luca Angioloni

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
"""

import html
import re
from html.parser import HTMLParser
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn import cross_validation
from sklearn.learning_curve import learning_curve

from timeit import default_timer as timer
from collections import Counter

# use cross validation
use_cv = True

# Learning curve steps (train sizes)
size_steps = 10

# number of iteration for each train size (cross validation)
num_iterations = 100

# size of the test used
test_size = 0.2

# bernulli / multinomial active
bernoulli = True
multinomial = True


# Path of the reuters dataset
reuters_path = "./reuters21578/"


def print_configuration():
    # Print script parameters
    print("\n")
    print("Reuters Learning Curves\n\n")
    print("Bernoulli Naive Bayes: " + str(bernoulli))
    print("Multinomial Naive Bayes: " + str(multinomial) + "\n")
    print("Cross Validation active: " + str(use_cv) + "\n")
    if use_cv:
        print("Number of iterations: " + str(num_iterations))
        print("Test Size: " + str(test_size) + "\n")
    print("Learning curves steps: " + str(size_steps) + "\n")


class ReutersParser(HTMLParser):
    """
    ReutersParser subclasses HTMLParser and is used to open the SGML
    files associated with the Reuters-21578 categorised test collection.

    The parser is a generator and will yield a single document at a time.
    Since the data will be chunked on parsing, it is necessary to keep
    some internal state of when tags have been "entered" and "exited".
    Hence the in_body, in_topics and in_topic_d boolean members.
    """
    def __init__(self, encoding='latin-1'):
        """
        Initialise the superclass (HTMLParser) and reset the parser.
        Sets the encoding of the SGML files by default to latin-1.
        """
        html.parser.HTMLParser.__init__(self)
        self._reset()
        self.encoding = encoding

    def _reset(self):
        """
        This is called only on initialisation of the parser class
        and when a new topic-body tuple has been generated. It
        resets all off the state so that a new tuple can be subsequently
        generated.
        """
        self.in_body = False
        self.in_topics = False
        self.in_topic_d = False
        self.body = ""
        self.topics = []
        self.topic_d = ""

    def parse(self, fd):
        """
        parse accepts a file descriptor and loads the data in chunks
        in order to minimise memory usage. It then yields new documents
        as they are parsed.
        """
        self.docs = []
        for chunk in fd:
            self.feed(chunk.decode(self.encoding))
            for doc in self.docs:
                yield doc
            self.docs = []
        self.close()

    def handle_starttag(self, tag, attrs):
        """
        This method is used to determine what to do when the parser
        comes across a particular tag of type "tag". In this instance
        we simply set the internal state booleans to True if that particular
        tag has been found.
        """
        if tag == "reuters":
            pass
        elif tag == "body":
            self.in_body = True
        elif tag == "topics":
            self.in_topics = True
        elif tag == "d":
            self.in_topic_d = True

    def handle_endtag(self, tag):
        """
        This method is used to determine what to do when the parser
        finishes with a particular tag of type "tag".

        If the tag is a  tag, then we remove all
        white-space with a regular expression and then append the
        topic-body tuple.

        If the tag is a  or  tag then we simply set
        the internal state to False for these booleans, respectively.

        If the tag is a  tag (found within a  tag), then we
        append the particular topic to the "topics" list and
        finally reset it.
        """
        if tag == "reuters":
            self.body = re.sub(r'\s+', r' ', self.body)
            self.docs.append( (self.topics, self.body) )
            self._reset()
        elif tag == "body":
            self.in_body = False
        elif tag == "topics":
            self.in_topics = False
        elif tag == "d":
            self.in_topic_d = False
            self.topics.append(self.topic_d)
            self.topic_d = ""

    def handle_data(self, data):
        """
        The data is simply appended to the appropriate member state
        for that particular tag, up until the end closing tag appears.
        """
        if self.in_body:
            self.body += data
        elif self.in_topic_d:
            self.topic_d += data


def obtain_topic_tags():
    # Open the topic list file and import all of the topic names
    topics = open(
        reuters_path + "all-topics-strings.lc.txt", "r"
    ).readlines()

    # remove \n
    topics = [t.strip() for t in topics]
    return topics


def get_frequent_topic_list(topics, docs):
    # Returns a list containing the 10 most common topics
    topics_occurrences = []
    for d in docs:
        if d[0] == [] or d[0] == "":
            continue
        for t in d[0]:
            if t in topics:
                topics_occurrences.append(t)
                break
    return list(dict(Counter(topics_occurrences).most_common(10)).keys())


def filter_doc_list_through_topics(frequent_topics, docs):
    """
    Reads all of the documents and creates a new list of two-tuples
    that contain a single feature entry and the body text, instead of
    a list of topics. It removes all geographic features and only
    retains those documents which have at least one non-geographic
    topic.
    """
    ref_docs = []
    for d in docs:
        if d[0] == [] or d[0] == "":
            continue
        for t in d[0]:
            if t in frequent_topics:
                d_tup = (t, d[1])
                ref_docs.append(d_tup)
                break
    return ref_docs


def create_training_data(docs):
    # Create the training data class labels
    y = [d[0] for d in docs]

    # Create the document corpus list
    corpus = [d[1] for d in docs]

    # Create the vectorizer and transform the corpus
    # vectorizer = TfidfVectorizer(min_df=1)
    vectorizer = CountVectorizer(analyzer="word", tokenizer=None, preprocessor=None, stop_words='english')
    X = vectorizer.fit_transform(corpus)
    print('Vectorizer shape: ' + str(X.shape))
    return vectorizer, X, y


def plot_learning_curve(estimator, title, X, y, cv=None, n_jobs=1, train_sizes=np.linspace(.1, 1.0, size_steps)):
    # calculate learning curve values and insert data into the plot
    plt.figure()
    plt.title(title)
    plt.xlabel("Training examples")
    plt.ylabel("Score")

    # Using sklearn learning_curve function
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    print(title + ": Final score result = " + str(test_scores_mean[len(test_scores_mean) - 1]))

    plt.grid()
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score (score on test)")

    plt.legend(loc="best")
    return plt


print_configuration()

start = timer()

readstart = timer()
# Create the list of Reuters data and create the parser
uri = reuters_path + "reut2-%03d.sgm"
files = [uri % r for r in range(0, 22)]
parser = ReutersParser()

# Parse the document and force all generated docs into
# a list so that it can be printed out to the console
docs = []
for fn in files:
    for d in parser.parse(open(fn, 'rb')):
        docs.append(d)

# Obtain the topic tags and filter docs through it
topics = obtain_topic_tags()

ten_most_common_topics = get_frequent_topic_list(topics, docs)
ref_docs = filter_doc_list_through_topics(ten_most_common_topics, docs)

readstop = timer()
print("Reading time: " + str(readstop-readstart) + " seconds")

# Vectorise and TF-IDF transform the corpus
vectorizerstart = timer()
vectorizer, X, y = create_training_data(ref_docs)
vectorizerstop = timer()
print("Vectorizer time: " + str(vectorizerstop-vectorizerstart) + " seconds")

if bernoulli:
    bernoulli_start = timer()
    print("Bernoulli learning curve...")
    bernoulli_nb = BernoulliNB(alpha=.01)
    title = "Learning Curves (Bernoulli) - Reuters"

    cv = None
    if use_cv:
        # Cross validation with 100 iterations to get smoother mean test and train
        # score curves, each time with 20% data randomly selected as a validation set.
        cv = cross_validation.ShuffleSplit(X.shape[0], n_iter=num_iterations,
                                           test_size=test_size, random_state=None)

    estimator = bernoulli_nb
    plot_learning_curve(estimator, title, X, y, cv=cv, n_jobs=4)
    bernoulli_end = timer()
    print("Bernoulli finished")
    print("Bernoulli time elapsed: " + str(bernoulli_end - bernoulli_start) + " seconds")


if multinomial:
    multinomial_start = timer()
    print("Multinomial learning curve...")
    multinomial_nb = MultinomialNB(alpha=.01)
    title = "Learning Curves (Multinomial) - Reuters"

    cv = None
    if use_cv:
        # Cross validation with 100 iterations to get smoother mean test and train
        # score curves, each time with 20% data randomly selected as a validation set.
        cv = cross_validation.ShuffleSplit(X.shape[0], n_iter=num_iterations,
                                           test_size=test_size, random_state=None)

    estimator = multinomial_nb
    plot_learning_curve(estimator, title, X, y, cv=cv, n_jobs=4)
    multinomial_end = timer()
    print("Multinomial finished")
    print("Multinomial time elapsed: " + str(multinomial_end - multinomial_start) + " seconds")

end = timer()
print("Total time elapsed: " + str(end - start) + " seconds")

print("Plotting")
plt.show()

print("Plotted.")
print("Finished")
