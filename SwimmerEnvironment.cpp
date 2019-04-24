#include "SwimmerEnvironment.h"

/*
	RLGlue methods for environment
*/
const char* env_init()
{    
	std::string s = "SwimmerEnvironment(C/C++) by Leon Zheng";
	const char *task_spec_string = s.c_str();

	/* Allocate the observation variable */
	const int numVar = 2*(2+(n_seg-1)); // A_0 is the head of the swimmer, 2D point; and there are n_seg-1 angles. We want also the derivatives.
	allocateRLStruct(&this_observation,0,numVar,0);

	/* Setup the reward_observation variable */
	this_reward_observation.observation=&this_observation;
	this_reward_observation.reward=0;
	this_reward_observation.terminal=0;

   return task_spec_string;
}

const observation_t *env_start()
{ 
	if(default_start_state){
		// Default position: everything is at 0
		for(int i=0; i++; i<this_observation.numDoubles){
			this_observation.doubleArray[i] = 0;
		}
	} else {
		//TODO random positions of the head and angles, initial speed is 0
	}
	return &this_observation;
}

const reward_observation_terminal_t *env_step(const action_t *this_action)
{
	/* Make sure the action is valid */
	assert(this_action->numDoubles == n_seg-1);
	for(int i=0; i<n_seg-1; i++){
		assert(abs(this_action->doubleArray[i]) < max_u);
	}

	updateState(this_observation, this_action->doubleArray);
	this_reward_observation.observation = &this_observation;
	this_reward_observation.reward = calculate_reward(this_observation);
	this_reward_observation.terminal = check_terminal(this_observation);

	return &this_reward_observation;
}

void env_cleanup()
{
	clearRLStruct(&this_observation);
}

/*
	Helper methods
*/
void updateState(observation_t& state, const double* torques)
{

}

double calculate_reward(const observation_t& state)
{
	return 0;
}

int check_terminal(const observation_t& state)
{
	return 0;
}

/*
	Main for checking compiling
*/

int main(int argc, char const *argv[])
{
	std::cout << "Hello world!" << std::endl;
	return 0;
}