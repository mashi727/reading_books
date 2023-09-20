cmake_minimum_required(VERSION 2.8)

SET(COMPILE_DEFINITIONS -Werror)
SET(CMAKE_CXX_FLAGS "-O3")
SET(CMAKE_C_FLAGS "-O3")
find_package( OpenCV REQUIRED )

include_directories(/opt/vc/include)
include_directories(/opt/vc/include/interface/vcos)
include_directories(/opt/vc/include/interface/vcos/pthreads)
include_directories(/opt/vc/include/interface/vmcs_host)
include_directories(/opt/vc/include/interface/vmcs_host/linux)

link_directories(/opt/vc/lib /usr/local/lib)

add_library(input_raspicam_opencv SHARED capture.cpp opencv_func.cpp RaspiCamControl.c RaspiCLI.c RaspiPreview.c)

target_link_libraries(input_raspicam_opencv mmal_core mmal_util mmal_vc_client vcos bcm_host ${OpenCV_LIBS} -lrt -lpthread -lwebsockets)
