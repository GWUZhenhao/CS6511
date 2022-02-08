import igraph as ig
import unittest
import A_star
import numpy as np

class A_star_test(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        self.concontainers = np.array([2, 3, 6, 72, 999999])
        self.state = np.array([2., 0., 0., 0., 0.])
        self.target = 143
        self.g = ig.Graph()
        self.g.add_vertex(name = 'start',
                     state = self.state,
                     goal = (self.target == 0),
                     g_score = 1,
                     h_score = 4,
                     f_score = 5,
                     is_leaf = True)

    def test_read_file(self):
        data = A_star.read_file('data.txt')
        self.assertEqual(data,['2,5,6,72', '143'])

    def test_calculate_h_score(self):
        h_score = A_star.calculate_h_score(self.concontainers, self.state, self.target)
        self.assertEqual(h_score, 4)

    def test_check_graph(self):
        not_stop = A_star.check_graph(self.g)
        self.assertTrue(not_stop)

    def test_calculate_steps(self):
        step = A_star.calculate_steps(self.g)
        self.assertFalse(step)

    def test_construct_graph(self):
        output = A_star.construct_graph(self.concontainers, self.target)
        self.assertEqual(output, 7)


if __name__ == '__main__':
    unittest.main()