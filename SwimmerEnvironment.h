/*
	Swimmer environment implementation in RLGLue framework.
	The model used is based on the one used by Remi Coulom in his thesis.
*/

#include <iostream>
#include <cmath>
#include <assert.h> /*assert*/
// env_ function prototypes types 
#include <rlglue/Environment_common.h>	  
// helpful functions for allocating structs and cleaning them up 
#include <rlglue/utils/C/RLStruct_util.h>   

const observation_t *env_start();
const reward_observation_terminal_t *env_step(const action_t *this_action);
void updateState(observation_t& state, const double* torques);
double calculate_reward(const observation_t& state);
int check_terminal(const observation_t& state);

