#include "../src/environment/SwimmerEnvironment.h"

int main(int argc, char const *argv[])
{

	n_seg = 2;
	h_global = 0.01;
	max_u = 5.;
	l_i = 1.;
	k = 10.;
	m_i = 1.;
 
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

	updateState(this_observation, &this_action);

	std::cout << "State after update: " << std::endl;
	print_state(this_observation);

	std::cout << "Done" << std::endl;

	return 0;
}