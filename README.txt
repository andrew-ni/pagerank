Usage: 
  python pagerank.py
  (or) ./pagerank.py

-uses numpy, scipy, sklearn, pandas packages
-uses dataset of 'enwiki-2013.txt.gz', which is an edge list of node id's that tells us which websites point to which other websites. 'enwiki-2013.txt.gz' has a cooresponding names file, which maps node id to website.

To manage a dataset on this scale, we have to use pandas dataframes and sparse matrices. For 'enwiki-2013.txt.gz', our sparse matrix ends up being 4206289x4206289.

This is an implementation of PageRank algorithm using hubs and authorities scoring, and power method. The program returns a quality of solution, 5 best authority pages, 5 best hub pages, and 5 best pagerank pages

Comparing this implementation and sklearn's svd function: 
  -Equivalent quality of solution
  -This implementation has about a 40% faster runtime 

