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
	const int n_obs = 2 + 2*n_seg; // G_dot_x, G_dot_y, theta_i, theta_dot_i
	const int n_action = n_seg-1;
	allocateRLStruct(&this_observation, 0, n_obs, 0);
	allocateRLStruct(&saved_observation, 0, n_obs, 0);

	std::cout << "Initialization: size of observation is " << this_observation.numDoubles << std::endl;

	/* Setup the reward_observation variable */
	this_reward_observation.observation=&this_observation;
	this_reward_observation.reward=0;
	this_reward_observation.terminal=0;

	static std::string task_spec_string = "VERSION RL-Glue-3.0 PROBLEMTYPE continuing DISCOUNTFACTOR 0.9 OBSERVATIONS DOUBLES (" + std::to_string(n_obs) + " UNSPEC UNSPEC) ACTIONS DOUBLES (" + std::to_string(n_action) + " " + std::to_string(-max_u) + " " + std::to_string(max_u) + ") REWARDS (UNSPEC UNSPEC) EXTRA SwimmerEnvironment(C++) by Leon Zheng";

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

	print_state(this_observation);
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
	Vector2d G_dot(state.doubleArray[0], state.doubleArray[1]);
	std::vector<double> theta;
	std::vector<double> theta_dot;
	for(size_t i=0; i<n_seg; i++){
		theta.push_back(state.doubleArray[2 + 2*i]);
		theta_dot.push_back(state.doubleArray[2 + 2*i + 1]);
	}
	// std::cout << "Information extracted" << std::endl;

	// Compute accelerations
	Vector2d G_dotdot(0., 0.);
	std::vector<double> theta_dotdot;
	compute_accelerations(torque, G_dot, theta, theta_dot, G_dotdot, theta_dotdot);
	// std::cout << "Accelerations computed" << std::endl;

	// Semi-implicit Euler
	const double h = h_global;
	semi_implicit_euler(h, theta, G_dot, theta_dot, G_dotdot, theta_dotdot);
	// std::cout << "Semi-implicit Euler done" << std::endl;

	// Return new state
	state.doubleArray[0] = G_dot(0);
	state.doubleArray[1] = G_dot(1);
	for(size_t i=0; i<n_seg; i++){
		state.doubleArray[2 + 2*i] = theta[i];
		state.doubleArray[2 + 2*i + 1] = theta_dot[i];
	}
	// std::cout << "New state returned" << std::endl;
}

void compute_accelerations(const std::vector<double> &torque, const Vector2d &G_dot, const std::vector<double> &theta, const std::vector<double> &theta_dot,
							Vector2d &G_dotdot, std::vector<double> &theta_dotdot)
{
	// Compute friction forces and torques
	std::vector<Vector2d> F_friction;
	std::vector<double> M_friction;
	compute_friction(G_dot, theta, theta_dot, F_friction, M_friction);

	// Matrix and vector of the linear system
	MatrixXd A = MatrixXd::Zero(5*n_seg+2, 5*n_seg+2);
	VectorXd B = VectorXd::Zero(5*n_seg+2);

	// Dynamic equations: lines 0 to n_seg-1
	for (size_t i = 1; i <= n_seg; i++) { // angles..
		A(i-1, i-1)             = m_i*pow(l_i,2)/12; // angular momentum of the segment, around the mass center
		A(i-1, n_seg + 2*i + 0) = +l_i/2*sin(theta[i-1]); // f_(i, x)
		A(i-1, n_seg + 2*i + 2) = +l_i/2*sin(theta[i-1]); // f_(i+1, x)
		A(i-1, n_seg + 2*i + 1) = -l_i/2*cos(theta[i-1]); // f_(i, y)
		A(i-1, n_seg + 2*i + 3) = -l_i/2*cos(theta[i-1]); // f_(i+1, y)

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
				A(3*n_seg+4 + 2*(i-1) + d, i-1) = -l_i/2*sin(theta[i-1]); //theta.._i
				A(3*n_seg+4 + 2*(i-1) + d, i-0) = -l_i/2*sin(theta[i-0]); //theta.._(i+1)

				B(3*n_seg+4 + 2*(i-1) + d)      = +l_i/2*(cos(theta[i-1])*pow(theta_dot[i-1],2) +
				                                          cos(theta[i-0])*pow(theta_dot[i-0],2));
			} else {
				A(3*n_seg+4 + 2*(i-1) + d, i-1) = +l_i/2*cos(theta[i-1]); //theta.._i
				A(3*n_seg+4 + 2*(i-1) + d, i-0) = +l_i/2*cos(theta[i-0]); //theta.._(i+1)

				B(3*n_seg+4 + 2*(i-1) + d)      = +l_i/2*(sin(theta[i-1])*pow(theta_dot[i-1],2) +
				                                          sin(theta[i-0])*pow(theta_dot[i-0],2));
			}
		}
	}

	// std::cout << "-----------------Matrix A-----------------" << std::endl << A << std::endl << "------------------------------------------" << std::endl;
	// std::cout << "-----------------Vector B-----------------" << std::endl << B << std::endl << "------------------------------------------" << std::endl;

	// Solve linear equation, extract second derivatives
	VectorXd X = A.colPivHouseholderQr().solve(B);

	// std::cout << "-----------------Matrix inv_A-----------------" << std::endl << A.inverse() << std::endl << "------------------------------------------" << std::endl;
	// std::cout << "-----------------Vector X-----------------" << std::endl << X << std::endl << "------------------------------------------" << std::endl;

	VectorXd X_ = A.inverse()*B;
	// std::cout << "-----------------Vector X_-----------------" << std::endl << X_ << std::endl << "------------------------------------------" << std::endl;


	for (size_t i = 1; i <= n_seg; i++) {
		theta_dotdot.push_back(X(i-1)); 
		G_dotdot += 1./n_seg * Vector2d(X(3*n_seg + 2*i), X(3*n_seg + 2*i + 1));
	}
}

void semi_implicit_euler(double h, std::vector<double> &theta, Vector2d& G_dot, std::vector<double> &theta_dot, 
							const Vector2d& G_dotdot, const std::vector<double> &theta_dotdot)
{
	G_dot = G_dot + h*G_dotdot;
	for(size_t i=0; i<n_seg; i++){
		theta_dot[i] = theta_dot[i] + h*theta_dotdot[i];
		theta[i] = theta[i] + h*theta_dot[i];
	}
}

void compute_friction(const Vector2d &G_dot, const std::vector<double> &theta, const std::vector<double> &theta_dot, 
							std::vector<Vector2d> &F_friction, std::vector<double> &M_friction)
{
	// Computes direction unit vectors
	std::vector<Vector2d> n_i;
	for (size_t i=0; i<n_seg; i++) {
		n_i.push_back(Vector2d(-sin(theta[i]), cos(theta[i])));
	}

	// Compute G1_dot
	Vector2d G1_dot = G_dot;
	for (size_t i = 1; i <= n_seg; i++) {
		Vector2d sum(0., 0.);
		for(size_t j=0; j<i; j++){
			const double e = (j==0 || j==i-1) ? 0.5 : 1.;
			sum += e * theta_dot[j] * n_i[j];
		}
		G1_dot -= l_i/n_seg * sum;
	}

	// Compute mass centers speed
	std::vector<Vector2d> G_i_dot;
	G_i_dot.push_back(G1_dot);
	for (size_t i = 1; i < n_seg; i++) {
		G_i_dot.push_back(G_i_dot[i-1] + l_i/2*theta_dot[i-1]*n_i[i-1] + l_i/2*theta_dot[i]*n_i[i]);
	}

	// Compute friction forces and torques
	for (size_t i = 0; i < n_seg; i++) {
		F_friction.push_back(-k*l_i*G_i_dot[i].dot(n_i[i])*n_i[i]);
		M_friction.push_back(-k*theta_dot[i]*pow(l_i,3)/12.);
		// std::cout << "M_friction[" << i << "] = " << M_friction[i] << std::endl;
	}
}

double calculate_reward(const observation_t& state)
{
	const Vector2d G_dot(state.doubleArray[0], state.doubleArray[1]);
	return G_dot.dot(direction);
}

int check_terminal(const observation_t& state)
{
	return 0;
}

void print_state(const observation_t &state) 
{
	std::cout << "State = {";
	for (size_t i = 0; i < state.numDoubles; i++) {
		std::cout << state.doubleArray[i];
		if (i < state.numDoubles-1) {
			std::cout << "; ";
		} else {
			std::cout << "}" << std::endl; 
		}
	}
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