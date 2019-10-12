# To run tests:
# python -m unittest discover tests

import unittest
import mypkg

class TestHello(unittest.TestCase):

    def test_hello(self):
        self.assertEqual(mypkg.hello(), 'Hello World!')


if __name__ == '__main__':
    unittest.main()