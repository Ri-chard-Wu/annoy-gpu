annoy-gpu
-----

This project is derived from `Annoy <https://github.com/spotify/annoy/tree/main>`_. The original project can use multi-thread to accelerate build process. In this project GPU is used to accelerate the build process. This project is still under development. Currently it only support the 'angular' metrics.

Install
-------

First set the environment variable "CUDAHOME" to CUDA installation location, e.g. ``export CUDAHOME=/usr/local/cuda-10.2``, then run ``pip install annoy_gpu`` to pull down the latest version from `PyPI <https://test.pypi.org/project/annoy-gpu>`_.

Python code example
-------------------

Add items and then save at 'test-1e5.tree'.

.. code-block:: python

  from annoy_gpu import AnnoyIndex
  import random, time

  n, f = 100000, 768

  t = AnnoyIndex(f, 'angular')

  t.fill_items('test-1e5.tree')

  for i in xrange(n):
      print(f"{i} / {n}")
      v = []
      for z in xrange(f):
          v.append(random.gauss(0, 1))
      t.add_item(i, v)

  t.save_items()


Load the saved items 'test-1e5.tree' and build 5 index trees, then perform precision test.

.. code-block:: python

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


Currently it only allows you to build on disk.