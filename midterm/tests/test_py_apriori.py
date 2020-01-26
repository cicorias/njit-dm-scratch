import py_apriori # The code to test
import unittest   # The test framework

from py_apriori import __version__

class Test_TestVersion(unittest.TestCase):

    def test_version(self):
        self.assertEqual(__version__, '0.1.0')

if __name__ == '__main__':
    unittest.main()