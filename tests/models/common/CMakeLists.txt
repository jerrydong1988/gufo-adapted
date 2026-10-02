add_executable(model_registry_test registry_test.cpp)
target_link_libraries(model_registry_test PRIVATE gufo_model_packages)
add_test(NAME model_registry_test COMMAND model_registry_test)
set_tests_properties(model_registry_test PROPERTIES
  LABELS "cpu;models;registry")
