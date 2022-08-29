#! /bin/env python3

import numpy as np
import unittest

class TestDemo(unittest.TestCase):
    def setUp(self):
        self.a=np.arange(5)


    def test(self):
        self.assertEqual(len(self.a), 5)




if __name__ == '__main__':
	unittest.main()

