import logging
logging.basicConfig(level=logging.INFO, format="%(funcName)s\t%(message)s")


class TreeNode(object):
    """
    NB: node_link is used to link similar items.
    We will also need the parent_node to in order to ascend the
    tree.
    """
    def __init__(self, name_value, number_ocurrences, parent_node):
        self.name = name_value
        self.count = number_ocurrences
        self.node_link = None
        self.parent = parent_node
        self.children = {}

    def inc(self, number_ocurrences):
        self.count += number_ocurrences

    def display(self, ind=1):
        print '  ' * ind, self.name, ' ', self.count
        for child in self.children.values():
            child.display(ind + 1)


def create_tree(dataset, minimum_support=1):
    """
    Makes two passes. First pass counts the frequency
    of each term. These are stored in the header table.
    Then prunes all elements whose count is less than the minimum
    support. Then the header table is expanded to hld a count
    and a pointer to the first item of each type.
    Then it creates the base node, which contains the null nset.
    Then you iterate over the dataset again, using only the frequents,
    you sort them, and then update_tree is called.
    """
    header_table = initialize_header(dataset)
    prune_infrequents(header_table, minimum_support)
    frequent_item_set = set(header_table.keys())

    if len(frequent_item_set) == 0:
        return None, None

    for k in header_table:
        header_table[k] = [header_table[k], None]

    return_tree = TreeNode('Null Set', 1, None)

    for transaction_set, count in dataset.items():
        local_d = {}
        for item in transaction_set:
            if item in frequent_item_set:
                local_d[item] = header_table[item][0]
        if len(local_d) > 0:
            update_tree(order_items(local_d), return_tree, header_table, count)
    return return_tree, header_table


def initialize_header(dataset):
    header_table = {}
    for transaction in dataset:
        for item in transaction:
            header_table[item] = header_table.get(item, 0) \
                + dataset[transaction]
    return header_table


def prune_infrequents(header_table, minimum_support):
    for k in header_table.keys():
        if header_table[k] < minimum_support:
            del(header_table[k])


def order_items(local_d):
    key_lambda = lambda p: p[1]
    sorted_items = sorted(local_d.items(),
                          key=key_lambda,
                          reverse=True)
    return [v[0] for v in sorted_items]


def update_tree(items, input_tree, header_table, count):
    """
    First tests if th first item exists as a child node.
    If so, it updates the count of that item. If not,
    it creates a new TreeNode and adds it as a child.
    Header table s also updated to point to a new node.
    Then it calls itself recursively with the cdr of the list.
    """
    if items[0] in input_tree.children:
        input_tree.children[items[0]].inc(count)
    else:
        input_tree.children[items[0]] = TreeNode(items[0], count, input_tree)
        if header_table[items[0]][1] is None:
            header_table[items[0]][1] = input_tree.children[items[0]]
        else:
            update_header(header_table[items[0]][1],
                          input_tree.children[items[0]])

    # recursively call update_tree on remaining items
    if len(items) > 1:
        #logging.info("recursively callling update_tree on {i}".
                     #format(i=items))
        update_tree(items[1::],
                    input_tree.children[items[0]],
                    header_table,
                    count)


def update_header(node_to_test, target_node):
    while(node_to_test.node_link is not None):
        node_to_test = node_to_test.node_link
    node_to_test.node_link = target_node


def ascend_tree(leaf_node, prefix_path):
    if leaf_node.parent is not None:
        prefix_path.append(leaf_node.name)
        ascend_tree(leaf_node.parent, prefix_path)


def find_prefix_path(tree_node):
    """
    Generate a conditional pattern base given a single item.
    Visit every node in the tree that contains that item
    """
    conditional_patterns = {}
    while tree_node is not None:
        prefix_path = []
        ascend_tree(tree_node, prefix_path)
        if len(prefix_path) > 1:
            conditional_patterns[frozenset(prefix_path[1:])] = tree_node.count
        tree_node = tree_node.node_link
    return conditional_patterns


def mine_tree(input_tree,
              header_table,
              minimum_support,
              prefix,
              frequent_items):
    base_patterns = [v[0] for v in sorted(header_table.items(),
                                          key=lambda p: p[1])]
    for base_pattern in base_patterns:
        new_frequent_set = prefix.copy()
        new_frequent_set.add(base_pattern)
        frequent_items.append(new_frequent_set)
        conditional_pattern_bases =\
            find_prefix_path(header_table[base_pattern][1])
        condition_tree, head = create_tree(conditional_pattern_bases,
                                           minimum_support)
        if head is not None:
            mine_tree(condition_tree,
                      head,
                      minimum_support,
                      new_frequent_set,
                      frequent_items)
