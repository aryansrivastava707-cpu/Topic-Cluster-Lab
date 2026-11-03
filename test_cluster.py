import unittest
from cluster import cluster,tfidf,tokenize
class ClusterTests(unittest.TestCase):
 def test_tokenizer_removes_common_words(self):self.assertEqual(tokenize('The solar panel produces energy'),['solar','panel','produces','energy'])
 def test_empty_text_has_empty_vector(self):self.assertEqual(tfidf(['','']),[{},{}])
 def test_seeded_clusters_repeat(self):
  docs=['solar panels generate electricity from sunlight','wind turbines generate clean electricity','football teams score goals on the field','players train for the football match']
  a=cluster(docs,2,seed=4);b=cluster(docs,2,seed=4);self.assertEqual(a,b);self.assertEqual(len(a[1]),2)
 def test_invalid_cluster_count(self):
  with self.assertRaises(ValueError):cluster(['one'],2)
if __name__=='__main__':unittest.main()
