#include "SwimmerEnvironment.h"

// Global variables for RLGlue methods
static observation_t this_observation;
static reward_observation_terminal_t this_reward_observation;

// Parameters
//TODO put the parameters has an input of the file and don't recompile at each time
static int n_seg = 3;
static double max_u = 1;

/*
	RLGlue methods for environment
*/
const observation_t *env_start()
{ 
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

int main(int argc, char const *argv[])
{
	std::cout << "Hello world!" << std::endl;
	return 0;
}