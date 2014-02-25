import unittest
from context import lily
from lily import apriori

import logging
logging.basicConfig(level=logging.INFO, format="%(lineno)d\t%(message)s")


class TestApriori(unittest.TestCase):
    def test_create_c1_builds_map_of_frozensets(self):
        """
        apriori - create C1 creates frozenset structure
        """
        dataset = [
            [0, 0, 0, 7, 0, 9, 0, 10, 0, 12, 0, 14, 0, 0, 21, 0, 23, 0, 28],
            [0, 3, 0, 4, 0, 7, 0, 9, 0, 23, 0, 25, 0, 26],
            [0, 3, 0, 4, 0, 7, 0, 9, 0, 10, 0, 13, 0, 14, 0, 16, 0, 25, 0, 26],
            [1, 2, 1, 5, 1, 7, 1, 8, 1, 11, 1, 13, 1, 15, 1, 17, 1, 24, 1, 27]
        ]
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
        result = apriori.createC1(dataset)
        self.assertEqual(result, expected)

if __name__ == '__main__':
    unittest.main()
