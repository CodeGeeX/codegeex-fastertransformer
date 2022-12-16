cd /workspace/FasterTransformer/
mkdir build
cd build
cmake -DSM=80 -DCMAKE_BUILD_TYPE=Release -DBUILD_PYT=ON -DBUILD_MULTI_GPU=ON ..
make -j12
cd ..
./build/bin/codegeex_gemm 1 1 32 64 64  16348 50048 1 1 # 10B
# ./build/bin/gpt_gemm 1 1 32 16 64  4096 50048 0 1 # large
