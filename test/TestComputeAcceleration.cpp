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

Vector2d compute_Gdotdot(const double *gluestate, const std::vector<double> &a_angle, const Vector2d &a_head)
{
	// Barycenter velocity
	Vector2d Gdotdot(a_head(0), a_head(1));
	for(int i=1; i<n_seg; i++){
		Vector2d sum;
		for(int j=0; j<i; j++){
			const double e = (j==i-1) ? 0.5 : 1;
			sum += e * (a_angle[j] * Vector2d(-sin(gluestate[4+j]), cos(gluestate[4+j])) - pow(gluestate[4+n_seg+j],2) * Vector2d(cos(gluestate[4+j]), -sin(gluestate[4+j])));
		}
		Gdotdot += l_i/n_seg * sum;
	}
	return Vector2d(Gdotdot(0), Gdotdot(1));
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
	std::vector<double> torque_vec;
	this_action.doubleArray = torque;
	for(size_t i=0; i<n_seg-1; i++){
		torque[i] = max_u/2;
		torque_vec.push_back(max_u/2);
	}

	// States
	double remystate[n_remy] = {0.074309, -2.1914e-11, 0.705283, -1.85751, 0.569522, 0.836742, 2.50831, 0.25463, 1.5708, 1.11088e-15, 0.633281, -0.25463, 2.57207, -0.836742, 2.43631, 1.85751};
	double gluestate[n_state];

	convert_remy_to_glue(remystate, gluestate);
	this_observation.doubleArray = gluestate;

	std::cout << "Extract state..." << std::endl;

	// Extract the informations
	Vector2d p_head(gluestate[0], gluestate[1]);
	Vector2d v_head(gluestate[2], gluestate[3]);
	std::vector<double> p_angle;
	std::vector<double> v_angle;
	for(size_t i=0; i<n_seg; i++){
		p_angle.push_back(gluestate[4+i]);
		v_angle.push_back(gluestate[4+n_seg+i]);
	}
	std::cout << "Information extracted" << std::endl;

	// Printing informations
	std::cout << "State extracted is: " << std::endl;
	print_remy(remystate);
	print_state(this_observation);

	// Compute accelerations
	Vector2d a_head;
	std::vector<double> a_angle;
	std::cout << "Computing accelerations..." << std::endl;
	compute_accelerations(torque_vec, p_head, v_head, p_angle, v_angle, a_head, a_angle);

	// Printing accelerations
	std::cout << "Head acceleration: (" << a_head(0) << "; " << a_head(1) << ")" << std::endl;
	const Vector2d Gdotdot = compute_Gdotdot(gluestate, a_angle, a_head); 
	std::cout << "Barycenter acceleration: (" << Gdotdot(0) << "; " << Gdotdot(1) << ")" << std::endl;

	std::cout << "Angles acceleration: {";
	for(int i=0; i<n_seg; i++){
		std::cout << a_angle[i];
		if(i<n_seg-1)
			std::cout << "; ";
		else
			std::cout<< "}" << std::endl;
	}

	std::cout << "Done" << std::endl;

	return 0;
}
