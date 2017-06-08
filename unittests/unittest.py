import unittest

class FooTests(unittest.TestCase):

    def testFoo(self):
        self.failUnless(False)

def main():
    unittest.main()

if __name__ == '__main__':
    main()