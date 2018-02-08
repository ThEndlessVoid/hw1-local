add_library (matmul dgemm-blocked.c)
add_library (matmul-naive dgemm-naive.c)


target_include_directories (matmul PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories (matmul-naive PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
