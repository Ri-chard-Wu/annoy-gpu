
# # clear
# rm main

# # rm compile.log
# annoy_build_flag=${1}_BUILD

# nvcc -o main main.cu -D${annoy_build_flag} #2> compile.log
# # code compile.log

# # rm run.log
# ./main #> run.log
# # code run.log

#----------------------------------

# # kill -9 `jobs -ps`

# cd ~/fnlPrj/annoy/src
# rm ./annoy/annoylib.so


# annoy_build_flag=ANNOYLIB_${1}_BUILD
# # rm compile.log
# nvcc --shared -o annoylib.so annoymodule.cu --compiler-options '-fPIC' -I/usr/include/python3.6 -D${annoy_build_flag} #2> compile.log
# # code compile.log
# mv annoylib.so ./annoy

# python3 main.py #> run.log
# # code run.log


#----------------------------------

# pip3 install numpy
# sudo apt-get install python3-dev



#----------------------------------

# rm main


# # nvcc -o main main.cu

# # cuda_path=/usr/local/cuda-10.2/include
# g++ -o main main.cu --library=gpu --library-path=/usr/local/cuda-10.2/include \
#     --library=cudadevrt --library=cudart

# ./main 
#----------------------------------



#---------------------------

pip3 uninstall -y annoy-gpu
python setup.py sdist
pip3 install .


python3 ./test/main.py 

