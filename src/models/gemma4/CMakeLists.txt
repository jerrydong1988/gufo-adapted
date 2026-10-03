target_sources(gufo_core PRIVATE protocol.cpp image.cpp)
if(ENGINE_ENABLE_HIP)
  target_sources(gufo_core PRIVATE runtime.hip vision.hip)
endif()
