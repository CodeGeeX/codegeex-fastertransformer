cd /workspace/FasterTransformer/
mkdir build
cd build
# Change DSM to the correspoding version of GPUs (e.g. 80 for A100, RTX 3090; 75 for RTX TITAN)
cmake -DSM=80 -DCMAKE_BUILD_TYPE=Release -DBUILD_PYT=ON -DBUILD_MULTI_GPU=ON ..
make -j12
cd ..
./build/bin/codegeex_gemm 1 1 32 64 64  16348 50048 1 1