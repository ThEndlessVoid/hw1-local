add_executable (benchmark benchmark.c)

target_link_libraries (benchmark LINK_PUBLIC matmul matmul-naive)
