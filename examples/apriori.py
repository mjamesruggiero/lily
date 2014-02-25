import os
import sys
from collections import defaultdict
from time import sleep
from votesmart import votesmart
import logging
sys.path.insert(0, os.path.abspath('..'))
import lily
from lily import apriori
from lily import utils

logging.basicConfig(level=logging.INFO, format="%(funcName)s\t%(message)s")


def action_is_final(action):
    return action.level == 'House' and \
        (action.stage == 'Passage' or
         action.stage == 'Amendment Vote')


def get_action_ids():
    votesmart.apikey = utils.get_env_config()
    action_id_list = []
    bill_title_list = []
    fr = open('data/recent_20_bills.txt')
    for line in fr.readlines():
        bill_number = int(line.split('\t')[0])
        try:
            bill_detail = votesmart.votes.getBill(bill_number)
            for action in bill_detail.actions:
                if action_is_final(action):
                    action_id = int(action.actionId)
                    message = "bill: {b} has action_id {a}"
                    logging.info(message.format(b=bill_number, a=action_id))
                    action_id_list.append(action_id)
                    bill_title_list.append(line.strip().split('\t')[1])

        except Exception, e:
            logging.warning("problem getting bill {b}; message: {m}".
                            format(b=bill_number, m=str(e)))

        sleep(1)
    return action_id_list, bill_title_list


def get_vote_value(vote, current_vote_count):
    if vote.action == 'Nay':
        return current_vote_count
    elif vote.action == 'Yea':
        return current_vote_count + 1
    return None


def get_item_meanings(bill_titles):
    item_meaning = ['Republican', 'Democratic']
    for bill_title in bill_titles:
        item_meaning.append("{b} -- Nay".format(b=bill_title))
        item_meaning.append("{b} -- Yay".format(b=bill_title))
    return item_meaning


def get_translation_list(action_ids, bill_titles):
    translations = defaultdict(list)
    vote_count = 2
    item_meaning = get_item_meanings(bill_titles)

    for action_id in action_ids:
        sleep(3)
        logging.info("getting info for action_id {0}".format(action_id))
        try:
            vote_list = votesmart.votes.getBillActionVotes(action_id)
            for vote in vote_list:
                if vote.officeParties == 'Democratic':
                    translations[vote.candidateName].append(1)
                elif vote.officeParties == 'Republican':
                    translations[vote.candidateName].append(0)
                vote_value = get_vote_value(vote, vote_count)
                if vote_value:
                    translations[vote.candidateName].append(vote_value)
        except:
            logging.warning("problem getting action_id {a}".
                            format(a=action_id))
        vote_count += 2
    return translations, item_meaning


def test_voting_with_apriori():
    """
    the final driver
    """
    action_ids, bill_titles = get_action_ids()
    translations, item_meanings = get_translation_list(action_ids, bill_titles)
    dataset = [translations[key] for key in translations.keys()]

    L, support_data = apriori.apriori(dataset, minimum_support=0.3)
    logging.info("L is {}".format(L))

    confidence_levels = (0.7, 0.95, 0.99)
    for confidence_level in confidence_levels:
        logging.info("confidence level:\t{sl}".format(sl=confidence_level))
        rules = apriori.generate_rules(L, support_data, confidence_level)
        #logging.info("rules:\n{r}".format(r=rules))


def main():
    test_voting_with_apriori()

if __name__ == '__main__':
    main()
