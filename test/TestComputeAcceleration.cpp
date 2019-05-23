#include "../src/environment/SwimmerEnvironment.h"

int main(int argc, char const *argv[])
{
	set_parameters("../../src/parameters.txt");
	const size_t n_state = 2+2*n_seg;
	const size_t n_action = n_seg-1;

 	allocateRLStruct(&this_observation,0,n_state,0);
	std::cout << "Size of this_observation: " << this_observation.numDoubles << std::endl;

	// Fix control
	action_t this_action;
	double torque[n_action];
	this_action.doubleArray = torque;
	for(size_t i=0; i<n_seg-1; i++){
		torque[i] = max_u/2;
	}

	// States
	double state[n_state] = {-0.0453422, 1.33766e-11, -1.35003, -1.4868, 1.5708, -1.88179e-15, -1.79156, 1.4868};
	this_observation.doubleArray = state;

	std::cout << "Extract state..." << std::endl;

	// Extract the informations
	std::vector<double> vec_torque;
	for (size_t i = 0; i < n_action; i++) {
		vec_torque.push_back(torque[i]);
	}

	Vector2d G_dot(this_observation.doubleArray[0], this_observation.doubleArray[1]);
	std::vector<double> theta;
	std::vector<double> theta_dot;
	for (size_t i = 0; i < n_seg; i++) {
		theta.push_back(this_observation.doubleArray[2 + 2*i]);
		theta_dot.push_back(this_observation.doubleArray[2 + 2*i + 1]);
	}
	std::cout << "Information extracted" << std::endl;

	// Printing informations
	std::cout << "State extracted is: " << std::endl;
	print_state(this_observation);

	// Compute accelerations
	Vector2d G_dotdot(0., 0.);
	std::vector<double> theta_dotdot;
	std::cout << "Computing accelerations..." << std::endl;
	compute_accelerations(vec_torque, G_dot, theta, theta_dot, G_dotdot, theta_dotdot);
	// std::cout << "Accelerations computed" << std::endl;

	// Printing accelerations
	std::cout << "G_dotdot = (" << G_dotdot(0) << "; " << G_dotdot(1) << ")" << std::endl;
	std::cout << "Angles acceleration: {";
	for (int i = 0; i < n_seg; i++) {
		std::cout << theta_dotdot[i];
		if(i<n_seg-1)
			std::cout << "; ";
		else
			std::cout<< "}" << std::endl;
	}

	std::cout << "Done" << std::endl;

	return 0;
}
