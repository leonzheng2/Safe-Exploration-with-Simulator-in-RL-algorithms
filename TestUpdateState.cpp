#include "SwimmerEnvironment.h"

int main(int argc, char const *argv[])
{
	const int numVar = 2*(2+n_seg); // A_0 is the head of the swimmer, 2D point; and there are n_seg angles. We want also the derivatives.
	allocateRLStruct(&this_observation,0,numVar,0);

	action_t this_action;
	double torque[n_seg-1] = {1,-1,2};
	this_action.doubleArray = torque;

	this_observation.doubleArray[0] = 0.;
	this_observation.doubleArray[1] = 0.;
	this_observation.doubleArray[2] = 1.0;
	this_observation.doubleArray[3] = -0.5;
	this_observation.doubleArray[4] = 0.1;
	this_observation.doubleArray[5] = 0.2;
	this_observation.doubleArray[6] = 0.3;
	this_observation.doubleArray[7] = 2;
	this_observation.doubleArray[8] = -1;
	this_observation.doubleArray[9] = 0.5;

	std::cout << "Updating state..." << std::endl;
	updateState(this_observation, &this_action);
	std::cout << "Done" << std::endl;

	for(int i=0; i<numVar; i++){
		std::cout << this_observation.doubleArray[i] << std::endl;
	}

	return 0;
}