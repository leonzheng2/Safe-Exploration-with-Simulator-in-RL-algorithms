#include "SwimmerEnvironment.h"

int main(int argc, char const *argv[])
{
	std::cout << "Hello world!" << std::endl;

	const double torque[n_seg-1] = {1,-1,2};
	Vector2d p_head(0, 0);
	Vector2d v_head(1.0, -0.5);
	double p_angle[n_seg] = {0.1, 0.2, 0.3};
	double v_angle[n_seg] = {2,-1,0.5};

	double a_angle[n_seg];
	Vector2d a_head;
	computeAccelerations(torque, p_head, v_head, p_angle, v_angle, a_angle, a_head);

	std::cout << "Head acceleration: (" << a_head(0) << ";" << a_head(1) << ")" << std::endl;
	std::cout << "Angles acceleration: " << std::endl;
	for(int i=0; i<n_seg; i++){
		std::cout << a_angle[i] << std::endl;
	}

	return 0;
}