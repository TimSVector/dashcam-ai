add_library(roadcam_core SHARED src/roadcam_core.cpp)
add_library(roadcam::core ALIAS roadcam_core)

target_compile_features(roadcam_core PUBLIC cxx_std_17)
target_include_directories(roadcam_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
set_target_properties(roadcam_core PROPERTIES
    CXX_EXTENSIONS OFF
    VERSION 0.1.0
    SOVERSION 0
)

add_executable(roadcam_core_tests tests/roadcam_core_tests.cpp)
target_link_libraries(roadcam_core_tests PRIVATE roadcam_core)
add_test(NAME roadcam_core_tests COMMAND roadcam_core_tests)
