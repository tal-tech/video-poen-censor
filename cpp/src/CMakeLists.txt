cmake_minimum_required(VERSION 3.10)
project(image_porn_censor)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -frtti")

add_library(image_porn_censor SHARED image_porn.h image_porn.cpp)

target_include_directories(image_porn_censor PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}>)

target_link_libraries(image_porn_censor
                      ${TRITON_INFER_LIBRARY} 
                      ${OPENCV_LIBRARY} 
                      ${BOOST_LIBRARY} 
                      -ldl 
                      -lpthread
                    )