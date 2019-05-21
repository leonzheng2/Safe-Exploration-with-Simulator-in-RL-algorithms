#include "SwimmerEnvironment.h"

Vector2d direction;
size_t n_seg;
double max_u;
double l_i;
double k;
double m_i;
double h_global;

/*
	RLGlue methods for environment
*/
const char* env_init()
{
	std::cout << "Initialize environment" << std::endl;
	/* Allocate the observation variable */
	const int numVar = 2*(2+n_seg); // A_0 is the head of the swimmer, 2D point; and there are n_seg angles. We want also the derivatives.
	allocateRLStruct(&this_observation,0,numVar,0);
	allocateRLStruct(&saved_observation,0,numVar,0);

	std::cout << "Initialization: size of observation is " << this_observation.numDoubles << std::endl;

	/* Setup the reward_observation variable */
	this_reward_observation.observation=&this_observation;
	this_reward_observation.reward=0;
	this_reward_observation.terminal=0;

	static std::string task_spec_string = "VERSION RL-Glue-3.0 PROBLEMTYPE continuing DISCOUNTFACTOR 0.9 OBSERVATIONS DOUBLES (" + std::to_string(2*(2+n_seg)) + " UNSPEC UNSPEC) ACTIONS DOUBLES (" + std::to_string(n_seg-1) + " " + std::to_string(-max_u) + " " + std::to_string(max_u) + ") REWARDS (UNSPEC UNSPEC) EXTRA SwimmerEnvironment(C++) by Leon Zheng";

	return task_spec_string.c_str();
}

const observation_t *env_start()
{
	std::cout << "Starting environment..." << std::endl;
	// Default position: random
	for(size_t i=0; i<this_observation.numDoubles; i++){
		// this_observation.doubleArray[i] = 0.0000001 * (double) rand() / (RAND_MAX);
		this_observation.doubleArray[i] = 0.001;
	}

	save_state();

	print_state(this_observation);

	std::cout << "Environment started!" << std::endl;

	return &this_observation;
}

const reward_observation_terminal_t *env_step(const action_t *this_action)
{
	/* Make sure the action is valid */
	assert(this_action->numDoubles == n_seg-1);
	for(size_t i=0; i<n_seg-1; i++){
		assert(abs(this_action->doubleArray[i]) <= max_u);
	}

	updateState(this_observation, this_action);
	this_reward_observation.observation = &this_observation;
	this_reward_observation.reward = calculate_reward(this_observation);
	this_reward_observation.terminal = check_terminal(this_observation);

	return &this_reward_observation;
}

void env_cleanup()	
{
	clearRLStruct(&this_observation);
	clearRLStruct(&saved_observation);
}

const char* env_message(const char * message)
{
    if(strcmp(message, "what is your name?")==0){
    	return "My name is swimmer_environment, C++ edition!";
    }
    if(strcmp(message, "save state")==0){
    	save_state();
    	return "saved_observation has the value of this_observation";
    }
    if(strcmp(message, "load state")==0){
    	load_state();
    	return "this_observation has the value of saved_observation";
    }
    if(strcmp(message, "set parameters")==0){
    	set_parameters("../parameters.txt");
    	static std::string s = "Environment parameters are: n_seg=" + std::to_string(n_seg) + "; max_u=" + std::to_string(max_u) + "; l_i=" + std::to_string(l_i) + "; k=" + std::to_string(k) + "; m_i=" + std::to_string(m_i) + "; h_global=" + std::to_string(h_global);
    	std::cout << s << std::endl;
    	return s.c_str();
    }

   	return "SwimmerEnvironment(C++) does not respond to that message.";
}

/*
	Helper methods
*/
void updateState(observation_t &state, const action_t* action)
{
	// Extract the informations
	std::vector<double> torque;
	for(size_t i=0; i<n_seg-1; i++){
		torque.push_back(action->doubleArray[i]);
	}
	Vector2d p_head(state.doubleArray[0], state.doubleArray[1]);
	Vector2d v_head(state.doubleArray[2], state.doubleArray[3]);
	std::vector<double> p_angle;
	std::vector<double> v_angle;
	for(size_t i=0; i<n_seg; i++){
		p_angle.push_back(state.doubleArray[4+i]);
		v_angle.push_back(state.doubleArray[4+n_seg+i]);
	}
	// std::cout << "Information extracted" << std::endl;

	// Compute accelerations
	Vector2d a_head;
	std::vector<double> a_angle;
	compute_accelerations(torque, p_head, v_head, p_angle, v_angle, a_head, a_angle);
	// std::cout << "Accelerations computed" << std::endl;

	// Semi-implicit Euler
	const double h = h_global;
	semi_implicit_euler(h, p_head, p_angle, v_head, v_angle, a_head, a_angle);
	// std::cout << "Semi-implicit Euler done" << std::endl;

	// Return new state
	state.doubleArray[0] = p_head(0);
	state.doubleArray[1] = p_head(1);
	state.doubleArray[2] = v_head(0);
	state.doubleArray[3] = v_head(1);
	for(size_t i=0; i<n_seg; i++){
		state.doubleArray[4+i] = p_angle[i];
		state.doubleArray[4+n_seg+i] = v_angle[i];
	}
	// std::cout << "New state returned" << std::endl;
}

void compute_accelerations(const std::vector<double> &torque, const Vector2d p_head, const Vector2d v_head, const std::vector<double> &p_angle, const std::vector<double> &v_angle,
							Vector2d &a_head, std::vector<double> &a_angle)
{
	// Computes direction unit vectors
	std::vector<Vector2d> p_i;
	std::vector<Vector2d> n_i;
	for (size_t i=0; i<n_seg; i++) {
		p_i.push_back(Vector2d( cos(p_angle[i]), sin(p_angle[i])));
		n_i.push_back(Vector2d(-sin(p_angle[i]), cos(p_angle[i])));
	}

	// Compute point's positions and speed
	std::vector<Vector2d> p_points;
	std::vector<Vector2d> v_points;
	p_points.push_back(p_head);
	v_points.push_back(v_head);
	for (size_t i=0; i<n_seg; i++) {
		p_points.push_back(p_points[i] + l_i*p_i[i]);
		v_points.push_back(v_points[i] + l_i*v_angle[i]*n_i[i]);
	}

	// And also for the mass centers
	std::vector<Vector2d> p_center;
	std::vector<Vector2d> v_center;
	for (size_t i=0; i<n_seg; i++) {
		p_center.push_back((p_points[i]+p_points[i+1])/2);
		v_center.push_back((v_points[i]+v_points[i+1])/2);
	}

	// Compute friction forces and torques
	std::vector<Vector2d> F_friction;
	std::vector<double> M_friction;
	for (size_t i=0; i<n_seg; i++) {
		F_friction.push_back(-k*l_i*v_center[i].dot(n_i[i])*n_i[i]);
		M_friction.push_back(-k*v_angle[i]*pow(l_i,3)/12);
		// std::cout << "M_friction[" << i << "] = " << M_friction[i] << std::endl;
	}

	// Matrix and vector of the linear system
	MatrixXd A = MatrixXd::Zero(5*n_seg+2, 5*n_seg+2);
	VectorXd B = VectorXd::Zero(5*n_seg+2);

	// Dynamic equations: lines 0 to n_seg-1
	for (size_t i = 1; i <= n_seg; i++) { // angles..
		A(i-1, i-1)             = m_i*l_i/12;
		A(i-1, n_seg + 2*i + 0) = +l_i/2*sin(p_angle[i-1]); // f_(i, x)
		A(i-1, n_seg + 2*i + 2) = +l_i/2*sin(p_angle[i-1]); // f_(i+1, x)
		A(i-1, n_seg + 2*i + 1) = -l_i/2*cos(p_angle[i-1]); // f_(i, y)
		A(i-1, n_seg + 2*i + 3) = -l_i/2*cos(p_angle[i-1]); // f_(i+1, y)

		B(i-1) = M_friction[i-1];
		if (i-2>=0)	B(i-1) += torque[i-2];
		if (i-1<n_seg-1) B(i-1) -= torque[i-1];
		// std::cout << "B[" << i-1 << "] = " << B(i-1) << std::endl;
		// if(i-1<n_seg-1) std::cout << "torque[" << i-1 << "] = " << torque[i-1] << std::endl;
	}


	// Equations on f_i: lines n_seg to 3*n_seg+3
	// lines n_seg to n_seg+1
	A(n_seg, n_seg) = 1;
	A(n_seg+1, n_seg+1) = 1;
	// lines n_seg+2 to 3*n_seg+1
	for (size_t i = 1; i <= n_seg; i++) {
		// Equation on x/y direction (x:d=0, y:d=1)
		for (int d = 0; d < 2; d++) {
			A(n_seg+2 + 2*(i-1) + d, d + n_seg + 2*(i-1)) = +1; //f_(i-1,x)
			A(n_seg+2 + 2*(i-1) + d, d + n_seg + 2*i)     = -1; //f_(i,x)
			A(n_seg+2 + 2*(i-1) + d, d + 3*n_seg + 2*i)   = m_i; //G.._(i,x)

			B(n_seg+2 + 2*(i-1) + d) = F_friction[i-1](d);
		}
	}
	// lines 3*n_seg+2 to 3*n_seg+3
	A(3*n_seg+2, 3*n_seg)   = 1;
	A(3*n_seg+3, 3*n_seg+1) = 1;

	// Equations on G.._i: lines 3*n_seg+4 to 5*n_seg+1
	for (size_t i = 1; i < n_seg; i++) {
		// Equation on x/y direction (x:d=0, y:d=1)
		for (int d = 0; d < 2; d++) {
			A(3*n_seg+4 + 2*(i-1) + d, d + 3*n_seg+2*(i+0)) = +1; //G.._(i,x)
			A(3*n_seg+4 + 2*(i-1) + d, d + 3*n_seg+2*(i+1)) = -1; //G.._(i+1,x)

			if (d == 0) {
				A(3*n_seg+4 + 2*(i-1) + d, i-1) = -l_i/2*sin(p_angle[i-1]); //theta.._i
				A(3*n_seg+4 + 2*(i-1) + d, i-0) = -l_i/2*sin(p_angle[i-0]); //theta.._(i+1)

				B(3*n_seg+4 + 2*(i-1) + d)      = +l_i/2*(cos(p_angle[i-1])*pow(v_angle[i-1],2) +
				                                          cos(p_angle[i-0])*pow(v_angle[i-0],2));
			} else {
				A(3*n_seg+4 + 2*(i-1) + d, i-1) = +l_i/2*cos(p_angle[i-1]); //theta.._i
				A(3*n_seg+4 + 2*(i-1) + d, i-0) = +l_i/2*cos(p_angle[i-0]); //theta.._(i+1)

				B(3*n_seg+4 + 2*(i-1) + d)      = +l_i/2*(sin(p_angle[i-1])*pow(v_angle[i-1],2) +
				                                          sin(p_angle[i-0])*pow(v_angle[i-0],2));
			}
		}
	}

	// std::cout << "-----------------Matrix A-----------------" << std::endl << A << std::endl << "------------------------------------------" << std::endl;
	// std::cout << "-----------------Vector B-----------------" << std::endl << B << std::endl << "------------------------------------------" << std::endl;


	// Solve linear equation, extract second derivatives
	VectorXd X = A.colPivHouseholderQr().solve(B);
	for(size_t i=0; i<n_seg; i++){
		a_angle.push_back(X(i));
	}
	a_head = Vector2d(X(3*n_seg+2), X(3*n_seg+3)) - l_i/2*(a_angle[0]*Vector2d(-sin(p_angle[0]), cos(p_angle[0])) - pow(v_angle[0],2)*Vector2d(cos(p_angle[0]), sin(p_angle[0])));
}

void semi_implicit_euler(double h, Vector2d& p_head, std::vector<double> &p_angle, Vector2d& v_head, std::vector<double> &v_angle, 
							const Vector2d& a_head, const std::vector<double> &a_angle)
{
	v_head = v_head + h*a_head;
	p_head = p_head + h*v_head;
	for(size_t i=0; i<n_seg; i++){
		v_angle[i] = v_angle[i] + h*a_angle[i];
		p_angle[i] = p_angle[i] + h*v_angle[i];
	}
}

double calculate_reward(const observation_t& state)
{
	const Vector2d v_head(state.doubleArray[2], state.doubleArray[3]);
	const double* p_angle = &state.doubleArray[4];
	const double* v_angle = &state.doubleArray[4 + n_seg];
	std::vector<Vector2d> v_points;
	v_points.push_back(v_head);
	for(size_t i=1; i<n_seg+1; i++){
		v_points.push_back(v_points[i-1] + l_i*v_angle[i-1]*Vector2d(-sin(p_angle[i-1]), cos(p_angle[i-1])));
	}

	Vector2d v_barycenter(0., 0.);
	for(size_t i=0; i<n_seg; i++){
		v_barycenter += 1./n_seg * (v_points[i]+v_points[i+1])/2;
	}

	return v_barycenter.dot(direction);
}

int check_terminal(const observation_t& state)
{
	return 0;
}

void save_state()
{
	saved_observation.numInts = this_observation.numInts;
	saved_observation.numDoubles = this_observation.numDoubles;
	saved_observation.numChars = this_observation.numChars;
	for(unsigned int i=0; i<saved_observation.numInts; i++){
		saved_observation.intArray[i] = this_observation.intArray[i];
	}
	for(unsigned int i=0; i<saved_observation.numDoubles; i++){
		saved_observation.doubleArray[i] = this_observation.doubleArray[i];
	}
	for(unsigned int i=0; i<saved_observation.numChars; i++){
		saved_observation.charArray[i] = this_observation.charArray[i];
	}
}

void load_state()
{
	this_observation.numInts = saved_observation.numInts;
	this_observation.numDoubles = saved_observation.numDoubles;
	this_observation.numChars = saved_observation.numChars;
	for(unsigned int i=0; i<this_observation.numInts; i++){
		this_observation.intArray[i] = saved_observation.intArray[i];
	}
	for(unsigned int i=0; i<this_observation.numDoubles; i++){
		this_observation.doubleArray[i] = saved_observation.doubleArray[i];
	}
	for(unsigned int i=0; i<this_observation.numChars; i++){
		this_observation.charArray[i] = saved_observation.charArray[i];
	}
}

void print_state(const observation_t &state) 
{
	Vector2d p_head(state.doubleArray[0], state.doubleArray[1]);
	Vector2d v_head(state.doubleArray[2], state.doubleArray[3]);
	const double* p_angle = &state.doubleArray[4];
	const double* v_angle = &state.doubleArray[4 + n_seg];

	std::cout << "p_head = (" << p_head[0] << "; " << p_head[1] << ")" << std::endl;
	std::cout << "v_head = (" << v_head[0] << "; " << v_head[1] << ")" << std::endl;
	std::string p_s = "p_angle = {";
	std::string v_s = "v_angle = {";
	for(size_t i=0; i<n_seg; i++){
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

void set_parameters(const std::string &param_file)
{
	using namespace std;
	string line;
	ifstream inFile(param_file);

	if (inFile.is_open()) {
		cout << "File is opened" << endl;
		while (getline(inFile, line)) {
	    	stringstream ss(line);
	    	string varName;
	    	ss >> varName;
			if(varName=="n_seg") ss >> n_seg;
			else if(varName=="max_u") ss >> max_u;
			else if(varName=="l_i") ss >> l_i;
			else if(varName=="k") ss >> k;
			else if(varName=="m_i") ss >> m_i;
			else if(varName=="h_global") ss >> h_global;
			else if(varName=="direction"){
				double x;
				double y;
				ss >> x;
				ss >> y;
				direction = Vector2d(x, y);
			}
	    }
	    inFile.close();
	}
	else cout << "Unable to open file for setting environment parameters" << std::endl; 
}
