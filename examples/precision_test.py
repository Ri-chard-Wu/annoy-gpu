from __future__ import print_function
import random, time

from annoy_gpu import AnnoyIndex



try:
    xrange
except NameError:
    # Python 3 compat
    xrange = range

n, f = 100000, 768

# t = AnnoyIndex(f, 'angular')

# t.fill_items('test-1e5.tree')

# for i in xrange(n):
#     print(f"{i} / {n}")
#     v = []
#     for z in xrange(f):
#         v.append(random.gauss(0, 1))
#     t.add_item(i, v)

# t.save_items()




q = AnnoyIndex(f, 'angular')


q.load_items('test-1e5.tree')

q.build(5)

limits = [10, 100, 1000, 10000]
k = 10
prec_sum = {}
prec_n = 10
time_sum = {}

for i in xrange(prec_n):
    j = random.randrange(0, n)
        
    closest = set(q.get_nns_by_item(j, k, n))
    for limit in limits:
        t0 = time.time()
        toplist = q.get_nns_by_item(j, k, limit)
        T = time.time() - t0
            
        found = len(closest.intersection(toplist))
        hitrate = 1.0 * found / k
        prec_sum[limit] = prec_sum.get(limit, 0.0) + hitrate
        time_sum[limit] = time_sum.get(limit, 0.0) + T

for limit in limits:
    print('limit: %-9d precision: %6.2f%% avg time: %.6fs'
          % (limit, 100.0 * prec_sum[limit] / (i + 1),
             time_sum[limit] / (i + 1)))
