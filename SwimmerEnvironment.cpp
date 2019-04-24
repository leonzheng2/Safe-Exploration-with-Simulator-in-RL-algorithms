#include <iostream>

// env_ function prototypes types 
#include <rlglue/Environment_common.h>	  

// helpful functions for allocating structs and cleaning them up 
#include <rlglue/utils/C/RLStruct_util.h>   

static observation_t this_observation;
static reward_observation_terminal_t this_reward_observation;

int main(int argc, char const *argv[])
{
	std::cout << "Hello world!" << std::endl;
	return 0;
}