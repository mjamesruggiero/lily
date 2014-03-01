import unittest
from context import lily
from lily import fp_growth

import logging
logging.basicConfig(level=logging.INFO, format="%(lineno)d\t%(message)s")


class TestTestFpGrowth(unittest.TestCase):
    def setUp(self):
        self.dataset = [
            ['r', 'z', 'h', 'j', 'p'],
            ['z', 'y', 'x', 'w', 'v', 'u', 't', 's'],
            ['z'],
            ['r', 'x', 'n', 'o', 's'],
            ['y', 'r', 'x', 'z', 'q', 't', 'p'],
            ['y', 'z', 'x', 'e', 'q', 's', 't', 'm'],
        ]
        self.init_set = self.create_initial_set(self.dataset)
        self.tree, self.header = fp_growth.create_tree(self.init_set, 3)

    def create_initial_set(self, dataset):
        return_dict = {}
        for transaction in dataset:
            return_dict[frozenset(transaction)] = 1
        return return_dict

    def test_find_prefix_path(self):
        """
        fp_growth - find_prefix_path builds conditional pattern base
        """
        conditional_pattern_base =\
            fp_growth.find_prefix_path(self.header['r'][1])
        expected = {
            frozenset(['x', 's']): 1,
            frozenset(['z']): 1,
            frozenset(['y', 'x', 'z']): 1
        }
        self.assertEqual(conditional_pattern_base, expected)

    def test_mine_tree(self):
        """
        fp_growth - mine tree returns frequent word sets
        """
        frequent_items = []
        fp_growth.mine_tree(self.tree,
                            self.header,
                            3,
                            set([]),
                            frequent_items)
        logging.info("frequent_items= {f}".format(f=frequent_items))
        expected = [
            set(['y']),
            set(['y', 'x']),
            set(['y', 'z']),
            set(['y', 'x', 'z']),
            set(['s']),
            set(['x', 's']),
            set(['t']),
            set(['y', 't']),
            set(['x', 't']),
            set(['y', 'x', 't']),
            set(['z', 't']),
            set(['y', 'z', 't']),
            set(['x', 'z', 't']),
            set(['y', 'x', 'z', 't']),
            set(['r']),
            set(['x']),
            set(['x', 'z']),
            set(['z'])]
        self.assertEqual(frequent_items, expected)


if __name__ == '__main__':
    unittest.main()
