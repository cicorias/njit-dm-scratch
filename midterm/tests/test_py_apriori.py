import py_apriori  # The code to test
import unittest   # The test framework


class Test_PyApriori_Basics(unittest.TestCase):

    def test_version(self):
        self.assertEqual(py_apriori.__version__, '0.1.0')

    def test_fail_with_none(self):
        with self.assertRaises(ValueError):
            _ = py_apriori.Apriori(None)

    def test_empty_set(self):
        test_data = []
        with self.assertRaises(ValueError):
            _ = py_apriori.Apriori(test_data)

    def test_not_tuple(self):
        test_data = (1, 2)  # list not a tuple
        with self.assertRaises(ValueError):
            _ = py_apriori.Apriori(test_data)

    def test_is_list_of_tuples(self):
        test_data = [(1, 2, 3), (2, 3, 4)]
        obj = py_apriori.Apriori(test_data)
        self.assertIsNotNone(obj.transactions)

    def test_is_not_list_of_tuples(self):
        test_data = [[1, 2, 3]]
        with self.assertRaises(ValueError):
            _ = py_apriori.Apriori(test_data)


if __name__ == '__main__':
    unittest.main()
