import logging
logging.basicConfig(level=logging.INFO, format="%(funcName)s\t%(message)s")


def createC1(dataset):
    """docstring for createC1"""
    C1 = []
    for transaction in dataset:
        for item in transaction:
            if not [item] in C1:
                C1.append([item])
    C1.sort()
    return map(frozenset, C1)


def scan_d(dataset, candidate_sets, minimum_support):
    """docstring for scan_d"""
    ss_count = {}
    for tid in dataset:
        for candidate in candidate_sets:
            if candidate.issubset(tid):
                if candidate not in ss_count:
                    ss_count[candidate] = 1
                else:
                    ss_count[candidate] += 1
    num_items = float(len(dataset))
    return_list = []
    support_data = {}
    for key in ss_count:
        support = ss_count[key] / num_items
        if support >= minimum_support:
            return_list.insert(0, key)
        support_data[key] = support
    return return_list, support_data


def apriori_generate(Lk, k):
    """
    Takes a list of frequent itemsets, Lk
    and the size of the sets, to produce
    candidate itemsets.
    """
    return_list = []
    len_Lk = len(Lk)
    for i in range(len_Lk):
        for j in range(i + 1, len_Lk):
            L1 = list(Lk[i])[:k - 2]
            L2 = list(Lk[j])[:k - 2]
            L1.sort()
            L2.sort()
            if L1 == L2:
                return_list.append(Lk[i] | Lk[j])  # set union
    return return_list


def apriori(dataset, minimum_support=0.5):
    C1 = createC1(dataset)
    D = map(set, dataset)
    L1, support_data = scan_d(D, C1, minimum_support)
    L = [L1]
    k = 2
    while(len(L[k - 2]) > 0):
        Ck = apriori_generate(L[k - 2], k)
        Lk, support_k = scan_d(D, Ck, minimum_support)
        support_data.update(support_k)
        L.append(Lk)
        k += 1
    return L, support_data


def generate_rules(L, support_data, minimum_confidence=0.7):
    big_rule_list = []
    for i in range(1, len(L)):
        for freq_set in L[i]:
            h1 = [frozenset([item]) for item in freq_set]
            if i > 1:
                rules_from_consequent(freq_set,
                                      h1,
                                      support_data,
                                      big_rule_list,
                                      minimum_confidence)
            else:
                calculate_confidence(freq_set,
                                     h1,
                                     support_data,
                                     big_rule_list,
                                     minimum_confidence)
    return big_rule_list


def calculate_confidence(freq_set,
                         h,
                         support_data,
                         big_rule_list,
                         minimum_confidence):
    pruned_h = []
    for conseq in h:
        conf = support_data[freq_set] / support_data[freq_set - conseq]
        if conf >= minimum_confidence:
            logging.info("{0} --> {1}, conf: {2}".
                         format(freq_set - conseq, conseq, conf))
            big_rule_list.append((freq_set - conseq, conseq, conf))
            pruned_h.append(conseq)
    return pruned_h


def rules_from_consequent(freq_set,
                          h,
                          support_data,
                          big_rule_list,
                          minimum_confidence=0.7):
    """
    TODO: instead of moving large param list around,
    use an object
    """
    m = len(h[0])
    if len(freq_set) > (m + 1):  # merge it more
        new_candidates = apriori_generate(h, m + 1)
        new_candidates = calculate_confidence(freq_set,
                                              new_candidates,
                                              support_data,
                                              big_rule_list,
                                              minimum_confidence)
        if len(new_candidates) > 1:  # need at least 2 sets to merge
            rules_from_consequent(freq_set,
                                  new_candidates,
                                  support_data,
                                  big_rule_list,
                                  minimum_confidence)
