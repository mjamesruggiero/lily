import os
import sys
sys.path.insert(0, os.path.abspath('..'))
from lily import core
import feedparser
import random
import logging
from numpy import array


def stopwords_file():
    return "data/stopwords.txt"


def filter_stopwords(vocabulary, stopwords_file):
    """
    remove stopwords from vocabulary
    """
    stopwords = core.get_stopwords(stopwords_file)
    vocabulary = [token for token in vocabulary
                  if not core.is_stopword(token, stopwords)]
    return vocabulary


def local_words(feed_1, feed_0):
    """
    Parse two RSS feeds;
    remove the most frequently ocurring words.
    """
    doc_list = []
    class_list = []
    full_text = []
    min_len = min(len(feed_1['entries']), len(feed_0['entries']))
    for i in range(min_len):
        word_list = core.text_parse(feed_1['entries'][i]['summary'])
        doc_list.append(word_list)
        full_text.extend(word_list)
        class_list.append(1)

        word_list = core.text_parse(feed_0['entries'][i]['summary'])
        doc_list.append(word_list)
        full_text.extend(word_list)
        class_list.append(0)

    vocabulary = core.create_vocabulary(doc_list)
    vocabulary = filter_stopwords(vocabulary, stopwords_file())

    # filter out stopwords
    stopwords = core.get_stopwords(stopwords_file())
    vocabulary = [token for token in vocabulary
                  if not core.is_stopword(token, stopwords)]

    top_thirty_words = core.calculate_most_frequent(vocabulary, full_text)
    for pair_w in top_thirty_words:
        if pair_w[0] in vocabulary:
            vocabulary.remove(pair_w[0])
    training_set = range(2*min_len)

    test_set = []
    for i in range(20):
        random_i = int(random.uniform(0, len(training_set)))
        test_set.append(training_set[random_i])
        del(training_set[random_i])
    training_matrix = []
    train_classes = []
    for doc_index in training_set:
        word_vector = core.bag_of_words_to_vector(vocabulary,
                                                  doc_list[doc_index])
        training_matrix.append(word_vector)
        train_classes.append(class_list[doc_index])

    p_0_v, p_1_v, p_spam = core.train_naive_bayes0(array(training_matrix),
                                                   array(train_classes))
    error_count = 0
    for doc_index in test_set:
        word_vector = core.bag_of_words_to_vector(vocabulary,
                                                  doc_list[doc_index])
        classification = core.classify_naive_bayes(array(word_vector),
                                                   p_0_v,
                                                   p_1_v,
                                                   p_spam)
        if classification != class_list[doc_index]:
            error_count += 1
    error_rate = float(error_count)/len(test_set)
    logging.info("errors: {0}\terror rate: {1}".format(error_count,
                                                       error_rate))
    return vocabulary, p_0_v, p_1_v


def get_top_words(feed_0, feed_1, file_0, file_1):
    vocabulary, p_0_v, p_1_v = local_words(feed_0, feed_1)
    top_0 = []
    top_1 = []
    for i in range(len(p_0_v)):
        if p_0_v[i] > -6.0:
            top_0.append((vocabulary[i], p_0_v[i]))
        if p_1_v[i] > -6.0:
            top_1.append((vocabulary[i], p_1_v[i]))

    sorted_0 = sorted(top_0, key=lambda pair: pair[1], reverse=True)
    sorted_1 = sorted(top_1, key=lambda pair: pair[1], reverse=True)

    core.save_to_csv(sorted_0, file_0)
    core.save_to_csv(sorted_1, file_1)


def main():
    sf = feedparser.parse('http://sfbay.craigslist.org/eng/index.rss')
    ny = feedparser.parse('http://newyork.craigslist.org/eng/index.rss')
    sf_file = '/tmp/sf_top_words.csv'
    ny_file = '/tmp/ny_top_words.csv'

    #vocabList, pSF, pNY = local_words(ny, sf)
    get_top_words(sf, ny, sf_file, ny_file)

if __name__ == '__main__':
    FORMAT = "%(lineno)d %(message)s"
    logging.basicConfig(level=logging.DEBUG, format=FORMAT)

    main()
