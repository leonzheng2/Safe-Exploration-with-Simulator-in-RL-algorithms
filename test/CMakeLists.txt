CMAKE_MINIMUM_REQUIRED (VERSION 2.6)

PROJECT (SwimmerEnv)

ADD_EXECUTABLE(TestAcceleration TestComputeAcceleration.cpp ../src/environment/SwimmerEnvironment.cpp)
TARGET_LINK_LIBRARIES(TestAcceleration rlutils rlglue)

# ADD_EXECUTABLE(TestEuler TestEuler.cpp SwimmerEnvironment.cpp)
# TARGET_LINK_LIBRARIES(TestEuler rlutils rlglue)

ADD_EXECUTABLE(TestUpdate TestUpdateState.cpp ../src/environment/SwimmerEnvironment.cpp)
TARGET_LINK_LIBRARIES(TestUpdate rlutils rlglue)
