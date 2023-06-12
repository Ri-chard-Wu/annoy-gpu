
"""
#######################################################

768, , 5, TBrain data, cpu

n trees built: 1 / 5
n trees built: 2 / 5
n trees built: 3 / 5
n trees built: 4 / 5
n trees built: 5 / 5

Done building in 353 secs.
limit: 10        precision:  45.00% avg time: 0.001838s
limit: 100       precision:  45.00% avg time: 0.001726s
limit: 1000      precision:  58.00% avg time: 0.003799s
limit: 10000     precision:  93.00% avg time: 0.025689s

#######################################################

768, , 5, max item num 1e6, TBrain data, cpu + gpu

Done building in 70 secs.
limit: 10        precision:  49.00% avg time: 0.001806s
limit: 100       precision:  49.00% avg time: 0.001728s
limit: 1000      precision:  62.00% avg time: 0.003204s
limit: 10000     precision:  86.00% avg time: 0.024875s


#######################################################

768, , 5, max item num 1.9e6, TBrain data, cpu + gpu

Done building in 58 secs.
limit: 10        precision:  44.00% avg time: 0.001258s
limit: 100       precision:  46.00% avg time: 0.001272s
limit: 1000      precision:  66.00% avg time: 0.003273s
limit: 10000     precision:  89.00% avg time: 0.021272s


#######################################################

768, 5e6, 5, gaussian data, cpu + gpu

Done building in 337 secs.
limit: 10        precision:  10.00% avg time: 0.002013s
limit: 100       precision:  10.00% avg time: 0.001815s
limit: 1000      precision:  10.00% avg time: 0.003787s
limit: 10000     precision:  10.00% avg time: 0.031639s

#######################################################

768, 1e6, 5, max item num 1e6, gaussian data, cpu + gpu

Done building in 35 secs.
limit: 10        precision:  10.00% avg time: 0.001551s
limit: 100       precision:  10.00% avg time: 0.001456s
limit: 1000      precision:  10.00% avg time: 0.003029s
limit: 10000     precision:  11.00% avg time: 0.025877s

"""






# from annoy import AnnoyIndex
# import numpy as np 
# import os 
# import random, time
# import os
# os.environ["CUDAHOME"] = "/usr/local/cuda-10.2"


# f = 768


# def fill_items():

#     t = AnnoyIndex(f, 'angular')
#     t.fill_items('TBrain.tree2')

#     dir = 'TBrain_data'

#     vec_list = []
#     for file in os.listdir(dir):
#         path = os.path.join(dir, file)
#         a = np.load(path)
#         print(path, a.shape)
#         vec_list += list(a)

#     for i, vec in enumerate(vec_list):
        
#         if(i % 1000 == 0): print(f"{i} / {len(vec_list)}")
#         t.add_item(i, list(vec))


#     t.save_items()



# def precision_test(t):

#     limits = [10, 100, 1000, 10000]
#     k = 10
#     prec_sum = {}
#     prec_n = 10
#     time_sum = {}

#     for i in range(prec_n):
#         j = random.randrange(0, t.get_n_items())
            
#         closest = set(t.get_nns_by_item(j, k, t.get_n_items()))
#         for limit in limits:
#             t0 = time.time()
#             toplist = t.get_nns_by_item(j, k, limit)
#             T = time.time() - t0
                
#             found = len(closest.intersection(toplist))
#             hitrate = 1.0 * found / k
#             prec_sum[limit] = prec_sum.get(limit, 0.0) + hitrate
#             time_sum[limit] = time_sum.get(limit, 0.0) + T

#     for limit in limits:
#         print('limit: %-9d precision: %6.2f%% avg time: %.6fs'
#             % (limit, 100.0 * prec_sum[limit] / (i + 1),
#                 time_sum[limit] / (i + 1)))



# # fill_items()

# t = AnnoyIndex(f, 'angular')
# t.load_items('./data/TBrain.tree')
# t.build(5)
# precision_test(t)



# -------------------------------




from __future__ import print_function
import random, time
try:
    xrange
except NameError:
    xrange = range


import sys


from annoy_gpu import AnnoyIndex
import os
os.environ["CUDAHOME"] = "/usr/local/cuda-10.2"






f = 768


def fill_items():

    n = 1000000
    t = AnnoyIndex(f, 'angular')
    t.fill_items('testPy-f768-n1e6.tree')

    for i in xrange(n):

        if(i%1000==0): print(f"{i} / {n}")

        v = []
        for z in xrange(f):
            v.append(random.gauss(0, 1))
        t.add_item(i, v)

    t.save_items()


def precision_test(t):
    limits = [10, 100, 1000, 10000]
    k = 10
    prec_sum = {}
    prec_n = 10
    time_sum = {}

    for i in xrange(prec_n):
        j = random.randrange(0, t.get_n_items())
            
        closest = set(t.get_nns_by_item(j, k, t.get_n_items()))
        for limit in limits:
            t0 = time.time()
            toplist = t.get_nns_by_item(j, k, limit)
            T = time.time() - t0
                
            found = len(closest.intersection(toplist))
            hitrate = 1.0 * found / k
            prec_sum[limit] = prec_sum.get(limit, 0.0) + hitrate
            time_sum[limit] = time_sum.get(limit, 0.0) + T

    for limit in limits:
        print('limit: %-9d precision: %6.2f%% avg time: %.6fs'
            % (limit, 100.0 * prec_sum[limit] / prec_n, time_sum[limit] / prec_n))


# fill_items()

t = AnnoyIndex(f, 'angular')

t.set_print_redirection(sys.stdout)

t.load_items('./test/data/testPy-f768-n1e5.tree')
t.build(5)
precision_test(t)


