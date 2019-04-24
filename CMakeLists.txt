CMAKE_MINIMUM_REQUIRED (VERSION 2.6)

PROJECT (SwimmerEnv)

ADD_EXECUTABLE(SwimmerEnv SwimmerEnvironment.cpp)
TARGET_LINK_LIBRARIES(SwimmerEnv rlutils rlglue)
