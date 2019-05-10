#include "SwimmerEnvironment.h"

int main(int argc, char const *argv[])
{
	const int numVar = 2*(2+n_seg); // A_0 is the head of the swimmer, 2D point; and there are n_seg angles. We want also the derivatives.
	allocateRLStruct(&this_observation,0,numVar,0);

	action_t this_action;
	double torque[n_seg-1] = {1.34981995};
	this_action.doubleArray = torque;

	// this_observation.doubleArray[0] = 14.00317796;
	// this_observation.doubleArray[1] = -3.74752228;
	// this_observation.doubleArray[2] = 4.32255351;
	// this_observation.doubleArray[3] = -5.31158584;
	// this_observation.doubleArray[4] = -3.94527853;
	// this_observation.doubleArray[5] = 14.58028602;
	// this_observation.doubleArray[6] = -22.38938012;
	// this_observation.doubleArray[7] = 36.87059979;


	torque[0] = {0.00303806};

	this_observation.doubleArray[0] = 11.5715;
	this_observation.doubleArray[1] = -1.17902;
	this_observation.doubleArray[2] = -24.3165;
	this_observation.doubleArray[3] = 25.685;
	this_observation.doubleArray[4] = -11.0167;
	this_observation.doubleArray[5] = -22.38938012;
	this_observation.doubleArray[6] = -70.714;
	this_observation.doubleArray[7] = 77.2129;



         
     
	// for(size_t i=0; i<this_observation.numDoubles; i++){
	// 	this_observation.doubleArray[8] = 0.1;
	// }

	std::cout << "Updating state..." << std::endl;
	updateState(this_observation, &this_action);
	std::cout << "Done" << std::endl;

	for(int i=0; i<numVar; i++){
		std::cout << this_observation.doubleArray[i] << std::endl;
	}

	return 0;
}