add_library (matmul dgemm-naive.c)

target_link_libraries (matmul LINK_PUBLIC HW1)