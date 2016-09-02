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

import re
import os
import codecs
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import MultinomialNB
from sklearn import cross_validation
from sklearn.learning_curve import learning_curve

from timeit import default_timer as timer


# remove quotes from articles
remove_quotes = True
regular_expression_quotes = re.compile(r'(writes in|writes:|wrote:|says:|said:'r'|^In article|^Quoted from|^\||^>)')

# remove non letters
non_letters = False

# bernulli / multinomial active
bernoulli = True
multinomial = True

# use cross validation
use_cv = True

# Learning curve steps (train sizes)
size_steps = 10

# number of iteration for each train size (cross validation)
num_iterations = 100

# size of the test used
test_size = 0.2

# Path of the dataset
news_groups_folder = './20news-18828'


def print_configuration():
    # Print script parameters
    print("\n")
    print("20 News Groups Learning Curves\n\n")
    print("Bernoulli Naive Bayes: " + str(bernoulli))
    print("Multinomial Naive Bayes: " + str(multinomial) + "\n")
    print("Remove Quotes: " + str(remove_quotes))
    print("Remove non letters: " + str(non_letters))
    print("Cross Validation active: " + str(use_cv) + "\n")
    if use_cv:
        print("Number of iterations: " + str(num_iterations))
        print("Test Size: " + str(test_size) + "\n")
    print("Learning curves steps: " + str(size_steps) + "\n")


def multiple_block_header(text):
    # some times headers are more than just one block of text
    if ":" in text.splitlines()[0]:
        return len(text.splitlines()[0].split(":")[0].split()) == 1
    else:
        return False


def strip_newsgroup_header(text):
    # strip article header
    if multiple_block_header(text):
        _before, _blankline, after = text.partition('\n\n')
        if len(after) > 0 and multiple_block_header(after):
            after = strip_newsgroup_header(after)
        return after
    else:
        return text


def strip_newsgroup_quoting(text):
    # strip article quotes
    good_lines = [line for line in text.split('\n')
                  if not regular_expression_quotes.search(line)]
    return '\n'.join(good_lines)


def strip_newsgroup_footer(text):
    # strip article footer
    lines = text.strip().split('\n')
    for line_num in range(len(lines) - 1, -1, -1):
        line = lines[line_num]
        if line.strip().strip('-') == '':
            break

    if line_num > 0:
        return '\n'.join(lines[:line_num])
    else:
        return text


def clean_text(raw_text):
    # Remove header, footer and quotes
    stripped_text = strip_newsgroup_footer(strip_newsgroup_header(raw_text))
    if remove_quotes:
        stripped_text = strip_newsgroup_quoting(stripped_text)

    # Remove non-letters
    if non_letters:
        stripped_text = re.sub("[^a-zA-Z]", " ", stripped_text)

    # Convert to lower case
    lower = stripped_text.lower()

    return lower


def scan_dir(folder, articles, id, folderName=""):
    # scan folders and files to extract text from documents
    for name in os.listdir(folder):
        path = os.path.join(folder, name)
        if os.path.isfile(path):
            category = folderName
            with codecs.open(path, 'r', encoding='utf-8', errors='replace') as content_file:
                raw_text = content_file.read()
            distilled = clean_text(raw_text)
            id += 1
            articles.append((id, category, distilled))
        else:
            scan_dir(path, articles, id, name)


def get_column(matrix, i):
    # returns matrix column as an array
    return [row[i] for row in matrix]


def create_training_data(train):
    # Create the training data class labels
    y = get_column(train, 1)

    # Create the document corpus list
    corpus = get_column(train, 2)

    # Create the vectorizer and transform the corpus
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

# matrix with data extracted from articles
articles_data = []
id = 0

# parse files and get text
readstart = timer()
scan_dir(news_groups_folder, articles_data, id)
readstop = timer()
print("Reading time: " + str(readstop-readstart) + " seconds")

# get vectorizer, data and labels
vectorizerstart = timer()
vectorizer, X, y = create_training_data(articles_data)
vectorizerstop = timer()
print("Vectorizer time: " + str(vectorizerstop-vectorizerstart) + " seconds")


# Cross validation
cv = None
if use_cv:
    cv = cross_validation.ShuffleSplit(X.shape[0], n_iter=num_iterations,
                                       test_size=test_size, random_state=None)

if bernoulli:
    bernoulli_start = timer()
    print("Bernoulli learning curve...")
    bernoulli_nb = BernoulliNB(alpha=.01)
    title = "Learning Curves (Bernoulli) - 20 News Groups"

    estimator = bernoulli_nb
    plot_learning_curve(estimator, title, X, y, cv=cv, n_jobs=4)
    bernoulli_end = timer()
    print("Bernoulli finished")
    print("Bernoulli time elapsed: " + str(bernoulli_end - bernoulli_start) + " seconds")


if multinomial:
    multinomial_start = timer()
    print("Multinomial learning curve...")
    multinomial_nb = MultinomialNB(alpha=.01)
    title = "Learning Curves (Multinomial) - 20 News Groups"

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
