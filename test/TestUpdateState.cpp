#include "../src/environment/SwimmerEnvironment.h"

void print_remy(const double* state)
{
	const size_t n_remy = 2+2*n_seg;
	std::string s = "Remy state = {";
	for(size_t i=0; i<n_remy; i++){
		s += std::to_string(state[i]);
		if(i==n_remy-1){
			s += "}";
		} else {
			s += "; ";
		}
	}
	std::cout << s << std::endl;
}

void convert_remy_to_glue(const double* remystate, double* gluestate)
{
	// Head position
	gluestate[0] = 0.;
	gluestate[1] = 0.;
	// Head velocity
	const Vector2d Gdot(remystate[0], remystate[1]);
	Vector2d Adot = Gdot;
	for(int i=1; i<n_seg; i++){
		Vector2d sum;
		for(int j=0; j<i; j++){
			const double e = (j==i-1) ? 0.5 : 1.;
			sum += e * remystate[2+2*j+1] * Vector2d(-sin(remystate[2+2*j]), cos(remystate[2+2*j]));
		}
		Adot -= l_i/n_seg * sum;
	}
	gluestate[2] = Adot(0);
	gluestate[3] = Adot(1);
	// Angles
	for(size_t i=0; i<n_seg; i++){
		gluestate[4+i] = remystate[2+2*i];
	}
	// Angles speed
	for(size_t i=0; i<n_seg; i++){
		gluestate[4+n_seg+i] = remystate[2+2*i+1];
	}
}

void convert_glue_to_remy(const double *gluestate, double *remystate)
{
	// Barycenter velocity
	const Vector2d Adot(gluestate[2], gluestate[3]);
	Vector2d Gdot = Adot;
	for(int i=1; i<n_seg; i++){
		Vector2d sum;
		for(int j=0; j<i; j++){
			const double e = (j==i-1) ? 0.5 : 1;
			sum += e * gluestate[4+n_seg+j] * Vector2d(-sin(gluestate[4+j]), cos(gluestate[4+j]));
		}
		Gdot += l_i/n_seg * sum;
	}
	remystate[0] = Gdot(0);
	remystate[1] = Gdot(1);
	// Angles
	for(size_t i=0; i<n_seg; i++){
		remystate[2+2*i] = gluestate[4+i];
	}
	// Angles speed
	for(size_t i=0; i<n_seg; i++){
		remystate[2+2*i+1] = gluestate[4+n_seg+i];
	}
}

int main(int argc, char const *argv[])
{
	set_parameters("../../src/parameters.txt");
	const size_t n_remy = 2+2*n_seg;
	const size_t n_state = 4+2*n_seg;

 	allocateRLStruct(&this_observation,0,n_state,0);
	std::cout << "Size of this_observation: " << this_observation.numDoubles << std::endl;

	// Fix control
	action_t this_action;
	double torque[n_seg-1];
	this_action.doubleArray = torque;
	for(size_t i=0; i<n_seg-1; i++){
		torque[i] = max_u/2;
	}

	// States
	double remystate[n_remy] = {0.0579709, -1.70959e-11, 0.457804, -1.75632, 0.677357, 0.757031, 2.48826, 0.119707, 1.5708, 1.15949e-15, 0.653328, -0.119707, 2.46424, -0.757031, 2.68379, 1.75632};
	double gluestate[n_state];

	convert_remy_to_glue(remystate, gluestate);
	this_observation.doubleArray = gluestate;

	std::cout << "Updating state..." << std::endl;

	std::cout << "State before update: " << std::endl;
	print_remy(remystate);
	print_state(this_observation);

	updateState(this_observation, &this_action);

	std::cout << "State after update: " << std::endl;
	print_state(this_observation);
	convert_glue_to_remy(gluestate, remystate);
	print_remy(remystate);

	std::cout << "Done" << std::endl;

	return 0;
}