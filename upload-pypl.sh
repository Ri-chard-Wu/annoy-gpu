
rm -r build
rm -r dist
rm -r annoy_gpu.egg-info

export CUDAHOME=/usr/local/cuda-10.2
python setup.py sdist
# ls dist

python -m twine upload --repository-url https://test.pypi.org/legacy/ dist/*


