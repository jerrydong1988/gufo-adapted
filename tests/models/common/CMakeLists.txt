add_executable(model_registry_test registry_test.cpp)
target_link_libraries(model_registry_test PRIVATE gufo_model_packages)
add_test(NAME model_registry_test COMMAND model_registry_test)
set_tests_properties(model_registry_test PROPERTIES
  LABELS "cpu;models;registry")

add_executable(model_ops_test ops_test.cpp)
target_link_libraries(model_ops_test PRIVATE gufo_model_packages)
add_test(NAME model_ops_test COMMAND model_ops_test)
set_tests_properties(model_ops_test PROPERTIES
  LABELS "cpu;models;ops")

add_executable(model_validate_test validate_test.cpp)
target_link_libraries(model_validate_test PRIVATE gufo_model_packages)
add_test(NAME model_validate_test COMMAND model_validate_test)
set_tests_properties(model_validate_test PROPERTIES
  LABELS "cpu;models;validate")
