/*
	Swimmer environment implementation in RLGLue framework.
	The model used is based on the one used by Remi Coulom in his thesis.
*/

#include <iostream>
#include <cmath>
#include <stdio.h>
#include <string.h>
#include <assert.h> /*assert*/
#include "Eigen/Dense" // Eigen library
// env_ function prototypes types 
#include <rlglue/Environment_common.h>	  
// helpful functions for allocating structs and cleaning them up 
#include <rlglue/utils/C/RLStruct_util.h>   

using namespace Eigen;

// Global variables for RLGlue methods
static observation_t this_observation;
static reward_observation_terminal_t this_reward_observation;

/* Used if a message is sent to the environment to use default start states */
static int default_start_state = 1;

// Parameters
// TODO put the parameters has an input of the file and don't recompile at each time
static int n_seg = 3;
static double max_u = 5;
static double l_i = 1;
static double k = 10;
static double m_i = 1;

// Methods
const char* env_init();
const observation_t *env_start();
const reward_observation_terminal_t *env_step(const action_t *this_action);
void env_cleanup();
const char* env_message(const char * message);

void computeAccelerations(const double* torque, const Vector2d p_head, const Vector2d v_head, const double* p_angle, const double* v_angle, 
							double* a_angle, Vector2d& a_head);
void updateState(observation_t& state, const action_t* action);
double calculate_reward(const observation_t& state);
int check_terminal(const observation_t& state);

