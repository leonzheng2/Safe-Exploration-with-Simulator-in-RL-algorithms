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

void compute_accelerations(const std::vector<double> &torque, const Vector2d &G_dot, const std::vector<double> &theta, const std::vector<double> &theta_dot,
							Vector2d &G_dotdot, std::vector<double> &theta_dotdot);
void semi_implicit_euler(double h, std::vector<double> &theta, Vector2d& G_dot, std::vector<double> &theta_dot, 
							const Vector2d& G_dotdot, const std::vector<double> &theta_dotdot);
void updateState(observation_t &state, const action_t* action);
double calculate_reward(const observation_t& state);
void compute_friction(const Vector2d &G_dot, const std::vector<double> &theta, const std::vector<double> &theta_dot, 
							std::vector<Vector2d> &F_friction, std::vector<double> &M_friction);
int check_terminal(const observation_t& state);
void save_state();
void load_state();
void print_state(const observation_t &state);
void set_parameters(const std::string &param_file);