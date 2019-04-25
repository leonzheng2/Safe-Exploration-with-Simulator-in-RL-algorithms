CMAKE_MINIMUM_REQUIRED (VERSION 2.6)

PROJECT (SwimmerEnv)

ADD_EXECUTABLE(TestAcceleration TestComputeAcceleration.cpp SwimmerEnvironment.cpp)
TARGET_LINK_LIBRARIES(TestAcceleration rlutils rlglue)