#include "SwimmerEnvironment.h"

void print_state(const observation_t &state){
	Vector2d p_head(state.doubleArray[0], state.doubleArray[1]);
	Vector2d v_head(state.doubleArray[2], state.doubleArray[3]);
	double* p_angle = &state.doubleArray[4];
	double* v_angle = &state.doubleArray[4 + n_seg];

	std::cout << "p_head = (" << p_head[0] << "; " << p_head[1] << ")" << std::endl;
	std::cout << "v_head = (" << v_head[0] << "; " << v_head[1] << ")" << std::endl;
	std::string p_s = "p_angle = {";
	std::string v_s = "v_angle = {";
	for(int i=0; i<n_seg; i++){
		p_s += std::to_string(p_angle[i]);
		v_s += std::to_string(v_angle[i]);
		if(i==n_seg-1){
			p_s += "}";
			v_s += "}";
		} else {
			p_s += "; ";
			v_s += "; ";
		}
	}
	std::cout << p_s << std::endl << v_s << std::endl;
}

// void test_updateState(observation_t &state, const action_t* action)
// {
// 	// Extract the informations
// 	const double* torque = action->doubleArray;
// 	Vector2d p_head(state.doubleArray[0], state.doubleArray[1]);
// 	Vector2d v_head(state.doubleArray[2], state.doubleArray[3]);
// 	double* p_angle = &state.doubleArray[4];
// 	double* v_angle = &state.doubleArray[4 + n_seg];

// 	std::cout << "State before update: " << std::endl;
// 	print_state(state);

// 	// Compute accelerations
// 	double a_angle[n_seg];
// 	Vector2d a_head;
// 	compute_accelerations(torque, p_head, v_head, p_angle, v_angle, a_angle, a_head);

// 	std::string a_s = "{";
// 	for(int i=0; i<n_seg; i++){
// 		a_s += std::to_string(a_angle[i]);
// 		if(i==n_seg-1){
// 			a_s += "}";
// 		} else {
// 			a_s += "; ";
// 		}
// 	}
// 	std::cout << "Computed accelerations: " << a_s << std::endl;

// 	// Semi-implicit Euler
// 	//TODO ISSUE what is the time interval??
// 	const double h = h_global;
// 	semi_implicit_euler(h, p_head, p_angle, v_head, v_angle, a_head, a_angle);

// 	std::cout << "State after update: " << std::endl;
// 	print_state(state);

// 	// Return new state
// 	state.doubleArray[0] = p_head(0);
// 	state.doubleArray[1] = p_head(1);
// 	state.doubleArray[2] = v_head(0);
// 	state.doubleArray[3] = v_head(1);
// 	// For the angles since we used pointers it is already updated
// }

int main(int argc, char const *argv[])
{

	// env_init();

	const int numVar = 2*(2+n_seg); // A_0 is the head of the swimmer, 2D point; and there are n_seg angles. We want also the derivatives.
	allocateRLStruct(&this_observation,0,numVar,0);

	std::cout << "Size of this_observation: " << this_observation.numDoubles << std::endl;

	action_t this_action;
	double torque[n_seg-1];
	this_action.doubleArray = torque;

	torque[0] = {0.57343072};


	this_observation.doubleArray[0] = 5.41090895;
	this_observation.doubleArray[1] = -5.62143928;
	this_observation.doubleArray[2] = -1.8534326;
	this_observation.doubleArray[3] = 4.72296265;
	this_observation.doubleArray[4] = 23.39752352;
	this_observation.doubleArray[5] = 40.82946899;
	this_observation.doubleArray[6] = -1.29783677;
	this_observation.doubleArray[7] = 7.57271494;


	std::cout << "Updating state..." << std::endl;

	std::cout << "State before update: " << std::endl;
	print_state(this_observation);

	// // Extract the informations
	// Vector2d p_head(this_observation.doubleArray[0], this_observation.doubleArray[1]);
	// Vector2d v_head(this_observation.doubleArray[2], this_observation.doubleArray[3]);
	// double* p_angle = &this_observation.doubleArray[4];
	// double* v_angle = &this_observation.doubleArray[4 + n_seg];

	// // Compute accelerations
	// double a_angle[n_seg];
	// Vector2d a_head;
	// compute_accelerations(torque, p_head, v_head, p_angle, v_angle, a_angle, a_head);

	// // Semi-implicit Euler
	// //TODO ISSUE what is the time interval??
	// const double h = h_global;
	// semi_implicit_euler(h, p_head, p_angle, v_head, v_angle, a_head, a_angle);

	// // Return new state
	// this_observation.doubleArray[0] = p_head(0);
	// this_observation.doubleArray[1] = p_head(1);
	// this_observation.doubleArray[2] = v_head(0);
	// this_observation.doubleArray[3] = v_head(1);
	// // For the angles since we used pointers it is already updated

	updateState(this_observation, &this_action);

	std::cout << "State after update: " << std::endl;
	print_state(this_observation);

	std::cout << "Done" << std::endl;

	env_cleanup();

	return 0;
}