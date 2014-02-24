import os
import sys
import logging
sys.path.insert(0, os.path.abspath('..'))
import lily
from lily import fp_growth
from lily import utils
import twitter
import re

logging.basicConfig(level=logging.INFO, format="%(funcName)s\t%(message)s")


def load_simple_data():
    return [
        ['r', 'z', 'h', 'j', 'p'],
        ['z', 'y', 'x', 'w', 'v', 'u', 't', 's'],
        ['z'],
        ['r', 'x', 'n', 'o', 's'],
        ['y', 'r', 'x', 'z', 'q', 't', 'p'],
        ['y', 'z', 'x', 'e', 'q', 's', 't', 'm'],
    ]


def create_initial_set(dataset):
    return_dict = {}
    for transaction in dataset:
        return_dict[frozenset(transaction)] = 1
    return return_dict


def get_many_tweets(search_string):
    CONSUMER_KEY = utils.get_env_config('CONSUMER_KEY')
    CONSUMER_SECRET = utils.get_env_config('CONSUMER_SECRET')
    ACCESS_TOKEN_KEY = utils.get_env_config('ACCESS_TOKEN_KEY')
    ACCESS_TOKEN_SECRET = utils.get_env_config('ACCESS_TOKEN_SECRET')
    api = twitter.Api(consumer_key=CONSUMER_KEY,
                      consumer_secret=CONSUMER_SECRET,
                      access_token_key=ACCESS_TOKEN_KEY,
                      access_token_secret=ACCESS_TOKEN_SECRET)

    count = 1500
    return api.GetSearch(search_string, count=count)


def parse_tweet(tweet_body):
    url_pattern = '(http[s]?:[/][/]|www.)([a-z]|[A-Z]|[0-9]|[/.]|[~])*'
    urls_removed = re.sub(url_pattern, '', tweet_body)
    tokens = re.split(r'\W*', urls_removed)
    return [tok.lower() for tok in tokens if len(tok) > 2]


def mine_tweets(tweets, minimum_support=5):
    parsed = [parse_tweet(tweet.text) for tweet in tweets]
    initial_set = create_initial_set(parsed)
    fp_tree, header = fp_growth.create_tree(initial_set, minimum_support)

    frequency_list = []
    fp_growth.mine_tree(fp_tree,
                        header,
                        minimum_support,
                        set([]),
                        frequency_list)
    return frequency_list


def twitter_test():
    several_tweets = get_many_tweets('icahn')
    for t in several_tweets:
        print t.text
    terms = mine_tweets(several_tweets, 20)
    logging.info("how many sets in 20 or more tweets? {tl}".
                 format(tl=len(terms)))
    for t in terms:
        print t


def simple_test():
    initial_set = create_initial_set(load_simple_data())
    logging.info("initial_set = {i}".format(i=initial_set))
    min_sup = 3
    fp_tree, header_table = fp_growth.create_tree(initial_set, min_sup)
    fp_tree.display()

    frequent_items = []
    mined = fp_growth.mine_tree(fp_tree,
                                header_table,
                                3,
                                set([]),
                                frequent_items)
    logging.info("mined = {m}".format(m=mined))


def main():
    twitter_test()


if __name__ == '__main__':
    main()
