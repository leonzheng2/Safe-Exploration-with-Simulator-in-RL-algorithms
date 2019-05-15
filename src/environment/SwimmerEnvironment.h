/*
	Swimmer environment implementation in RLGLue framework.
	The model used is based on the one used by Remi Coulom in his thesis.
*/

#include <iostream>
#include <fstream>
#include <sstream>
#include <cmath>
#include <vector>
#include <stdio.h>
#include <string.h>
#include <assert.h> /*assert*/
#include "../../Eigen/Dense" // Eigen library
// env_ function prototypes types 
#include <rlglue/Environment_common.h>	  
// helpful functions for allocating structs and cleaning them up 
#include <rlglue/utils/C/RLStruct_util.h>   

using namespace Eigen;

// Global variables for RLGlue methods
static observation_t this_observation;
static observation_t saved_observation;
static reward_observation_terminal_t this_reward_observation;

// Parameters
// TODO put the parameters has an input of the file and don't recompile at each time
extern Vector2d direction;
extern size_t n_seg;
extern double max_u;
extern double l_i;
extern double k;
extern double m_i;
extern double h_global;

// Methods
const char* env_init();
const observation_t *env_start();
const reward_observation_terminal_t *env_step(const action_t *this_action);
void env_cleanup();
const char* env_message(const char * message);

void compute_accelerations(const std::vector<double> &torque, const Vector2d p_head, const Vector2d v_head, const std::vector<double> &p_angle, const std::vector<double> &v_angle, 
							Vector2d& a_head, std::vector<double> &a_angle);
void semi_implicit_euler(double h, Vector2d& p_head, std::vector<double> &p_angle, Vector2d& v_head, std::vector<double> &v_angle, const Vector2d& a_head, const std::vector<double> &a_angle);
void updateState(observation_t& state, const action_t* action);
double calculate_reward(const observation_t& state);
int check_terminal(const observation_t& state);
void save_state();
void load_state();
void print_state(const observation_t &state);
void set_parameters();