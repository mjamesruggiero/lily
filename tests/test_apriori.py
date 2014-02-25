import unittest
from context import lily
from lily import apriori

import logging
logging.basicConfig(level=logging.INFO, format="%(lineno)d\t%(message)s")


class TestApriori(unittest.TestCase):
    def setUp(self):
        self.dataset = [
            [0, 0, 0, 7, 0, 9, 0, 10, 0, 12, 0, 14, 0, 0, 21, 0, 23, 0, 28],
            [0, 3, 0, 4, 0, 7, 0, 9, 0, 23, 0, 25, 0, 26],
            [0, 3, 0, 4, 0, 7, 0, 9, 0, 10, 0, 13, 0, 14, 0, 16, 0, 25, 0, 26],
            [1, 2, 1, 5, 1, 7, 1, 8, 1, 11, 1, 13, 1, 15, 1, 17, 1, 24, 1, 27]
        ]

    def test_create_c1_builds_map_of_frozensets(self):
        """
        apriori - create C1 creates frozenset structure
        """
        expected = [frozenset([0]), frozenset([1]),
                    frozenset([2]), frozenset([3]),
                    frozenset([4]), frozenset([5]),
                    frozenset([7]), frozenset([8]),
                    frozenset([9]), frozenset([10]),
                    frozenset([11]), frozenset([12]),
                    frozenset([13]), frozenset([14]),
                    frozenset([15]), frozenset([16]),
                    frozenset([17]), frozenset([21]),
                    frozenset([23]), frozenset([24]),
                    frozenset([25]), frozenset([26]),
                    frozenset([27]), frozenset([28])]
        result = apriori.createC1(self.dataset)
        self.assertEqual(result, expected)

    def test_count_candidates(self):
        """
        apriori - count candidates counds subsets
        """
        candidate_sets = apriori.createC1(self.dataset)
        counts = apriori.count_candiates(candidate_sets, self.dataset)
        expected = {
            frozenset([25]): 2,
            frozenset([21]): 1,
            frozenset([26]): 2,
            frozenset([16]): 1,
            frozenset([9]): 3,
            frozenset([7]): 4,
            frozenset([5]): 1,
            frozenset([10]): 2,
            frozenset([3]): 2,
            frozenset([0]): 3,
            frozenset([28]): 1,
            frozenset([2]): 1,
            frozenset([15]): 1,
            frozenset([13]): 2,
            frozenset([11]): 1,
            frozenset([8]): 1,
            frozenset([14]): 2,
            frozenset([23]): 2,
            frozenset([4]): 2,
            frozenset([17]): 1,
            frozenset([1]): 1,
            frozenset([12]): 1,
            frozenset([27]): 1,
            frozenset([24]): 1
        }
        self.assertEqual(dict(counts), expected)


if __name__ == '__main__':
    unittest.main()
