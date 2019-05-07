#include "SwimmerEnvironment.h"

/*
	RLGlue methods for environment
*/
const char* env_init()
{    
	std::string s = "SwimmerEnvironment(C++) by Leon Zheng";
	const char *task_spec_string = s.c_str();

	/* Allocate the observation variable */
	const int numVar = 2*(2+n_seg); // A_0 is the head of the swimmer, 2D point; and there are n_seg angles. We want also the derivatives.
	allocateRLStruct(&this_observation,0,numVar,0);

	/* Setup the reward_observation variable */
	this_reward_observation.observation=&this_observation;
	this_reward_observation.reward=0;
	this_reward_observation.terminal=0;

   return task_spec_string;
}

const observation_t *env_start()
{ 
	if(default_start_state){
		// Default position: everything is at 0
		for(size_t i=0; i<this_observation.numDoubles; i++){
			this_observation.doubleArray[i] = 0;
		}
	} else {
		//TODO random positions of the head and angles, initial speed is 0
	}
	return &this_observation;
}

const reward_observation_terminal_t *env_step(const action_t *this_action)
{
	/* Make sure the action is valid */
	assert(this_action->numDoubles == n_seg-1);
	for(size_t i=0; i<n_seg-1; i++){
		assert(abs(this_action->doubleArray[i]) < max_u);
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
}

const char* env_message(const char * message)
{
	/*	Message Description
 	 * 'set-default-start-state'
	 * Action: Set flag to do default starting states (the default)
	 */
	if(strcmp(message,"set-default-start-state")==0){
        default_start_state=1;
        return "Message understood.  Using default start state.";
    }

   	return "SwimmerEnvironment(C++) does not respond to that message.";
}

/*
	Helper methods
*/
void computeAccelerations(const double* torque, const Vector2d p_head, const Vector2d v_head, const double* p_angle, const double* v_angle, 
							double* a_angle, Vector2d& a_head)
{
	// Computes direction unit vectors
	Vector2d p_i[n_seg];
	Vector2d n_i[n_seg];
	for(size_t i=0; i<n_seg; i++){
		p_i[i] = Vector2d(cos(p_angle[i]), sin(p_angle[i]));
		n_i[i] = Vector2d(-sin(p_angle[i]), cos(p_angle[i]));
	}

	// Compute point's positions and speed
	Vector2d p_points[n_seg+1];
	Vector2d v_points[n_seg+1];
	p_points[0] = p_head;
	v_points[0] = v_head;
	for(size_t i=1; i<n_seg+1; i++){
		p_points[i] = p_points[i-1] + l_i*p_i[i-1];
		v_points[i] = v_points[i-1] + l_i*v_angle[i-1]*n_i[i-1];
	}

	// And also for the mass centers
	Vector2d p_center[n_seg];
	Vector2d v_center[n_seg];
	for(size_t i=0; i<n_seg; i++){
		p_center[i] = (p_points[i]+p_points[i+1])/2;
		v_center[i] = (v_points[i]+v_points[i+1])/2;
	}

	// Compute friction forces and torques
	Vector2d F_friction[n_seg];
	double M_friction[n_seg];
	for(size_t i=0; i<n_seg; i++){
		F_friction[i] = -k*l_i*v_center[i].dot(n_i[i])*n_i[i];
		M_friction[i] = -k*v_angle[i]*pow(l_i,3)/12;
	}

	// Matrix and vector of the linear system
	MatrixXd A = MatrixXd::Zero(5*n_seg+2, 5*n_seg+2);
	VectorXd B = VectorXd::Zero(5*n_seg+2);

	// Dynamic equations: lines 0 to n_seg-1
	for(size_t i=1; i<n_seg+1; i++){ // angles..
		A(i-1, i-1) = m_i*l_i/12;
		A(i-1, n_seg + 2*i) = l_i/2*sin(p_angle[i-1]); // f_(i, x)
		A(i-1, n_seg + 2*(i+1)) = l_i/2*sin(p_angle[i-1]); // f_(i+1, x)
		A(i-1, n_seg + 2*i + 1) = -l_i/2*cos(p_angle[i-1]); // f_(i, y)
		A(i-1, n_seg + 2*(i+1) + 1) = -l_i/2*cos(p_angle[i-1]); // f_(i+1, y)
		B(i-1) = M_friction[i-1]-torque[i-1];
		if(i-2>=0)	B(i-1) += torque[i-2];
	}

	// Equations on f_i: lines n_seg to 3*n_seg+3
	// lines n_seg to n_seg+1
	A(n_seg, n_seg) = 1;
	A(n_seg+1, n_seg+1) = 1;
	// lines n_seg+2 to 3*n_seg+1
	for(size_t i=1; i<n_seg+1; i++){
		// Equation on x direction
		A(n_seg+2 + 2*(i-1), n_seg + 2*(i-1)) = 1; //f_(i-1,x)
		A(n_seg+2 + 2*(i-1), n_seg + 2*i) = -1; //f_(i,x)
		A(n_seg+2 + 2*(i-1), 3*n_seg + 2*i) = m_i; //G.._(i,x)
		B(n_seg+2 + 2*(i-1)) = F_friction[i-1](0);
		// Equation on y direction
		A(n_seg+2 + 2*(i-1) + 1, n_seg + 2*(i-1) + 1) = 1; //f_(i-1,y)
		A(n_seg+2 + 2*(i-1) + 1, n_seg + 2*i + 1) = -1; //f_(i,y)
		A(n_seg+2 + 2*(i-1) + 1, 3*n_seg + 2*i + 1) = m_i; //G.._(i,y) //TODO DEBUG
		B(n_seg+2 + 2*(i-1) + 1) = F_friction[i-1](1);
	}
	// lines 3*n_seg+2 to 3*n_seg+3
	A(3*n_seg+2, 3*n_seg) = 1;
	A(3*n_seg+3, 3*n_seg+1) = 1;

	// Equations on G.._i: lines 3*n_seg+4 to 5*n_seg+1
	for(size_t i=1; i<n_seg; i++){
		// Equation on x direction
		A(3*n_seg+4 + 2*(i-1), 3*n_seg+2*i) = 1; //G.._(i,x)
		A(3*n_seg+4 + 2*(i-1), 3*n_seg+2*(i+1)) = -1; //G.._(i+1,x)
		A(3*n_seg+4 + 2*(i-1), i-1) = -l_i/2*sin(p_angle[i-1]); //theta.._i
		A(3*n_seg+4 + 2*(i-1), i) = -l_i/2*sin(p_angle[i]); //theta.._(i+1)
		B(3*n_seg+4 + 2*(i-1)) = l_i/2*(cos(p_angle[i-1])*pow(v_angle[i-1],2) + cos(p_angle[i])*pow(v_angle[i],2));
		// Equation on y direction
		A(3*n_seg+4 + 2*(i-1) + 1, 3*n_seg+2*i+1) = 1; //G.._(i,y)
		A(3*n_seg+4 + 2*(i-1) + 1, 3*n_seg+2*(i+1)+1) = -1; //G.._(i+1,y)
		A(3*n_seg+4 + 2*(i-1) + 1, i-1) = l_i/2*cos(p_angle[i-1]); //theta.._i
		A(3*n_seg+4 + 2*(i-1) + 1, i) = l_i/2*cos(p_angle[i]); //theta.._(i+1)
		B(3*n_seg+4 + 2*(i-1) + 1) = l_i/2*(sin(p_angle[i-1])*pow(v_angle[i-1],2) + sin(p_angle[i])*pow(v_angle[i],2));
	}

	// Solve linear equation, extract second derivatives
	VectorXd X = A.colPivHouseholderQr().solve(B);
	for(size_t i=0; i<n_seg; i++){
		a_angle[i] = X(i);
	}
	a_head = Vector2d(X(3*n_seg+2), X(3*n_seg+3)) - l_i/2*p_i[0];
}

void semiImplicitEuler(double h, Vector2d& p_head, double* p_angle, Vector2d& v_head, double* v_angle, const Vector2d& a_head, const double* a_angle)
{
	v_head = v_head + h*a_head;
	p_head = p_head + h*v_head;
	for(size_t i=0; i<n_seg; i++){
		v_angle[i] = v_angle[i] + h*a_angle[i];
		p_angle[i] = p_angle[i] + h*v_angle[i];
	}
}

void updateState(observation_t& state, const action_t* action)
{
	// Extract the informations
	const double* torque = action->doubleArray;
	Vector2d p_head(state.doubleArray[0], state.doubleArray[1]);
	Vector2d v_head(state.doubleArray[2], state.doubleArray[3]);
	double* p_angle = &state.doubleArray[4];
	double* v_angle = &state.doubleArray[4 + n_seg];

	// Compute accelerations
	double a_angle[n_seg];
	Vector2d a_head;
	computeAccelerations(torque, p_head, v_head, p_angle, v_angle, a_angle, a_head);

	// Semi-implicit Euler
	//TODO ISSUE what is the time interval??
	const double h = h_global;
	semiImplicitEuler(h, p_head, p_angle, v_head, v_angle, a_head, a_angle);

	// Return new state
	state.doubleArray[0] = p_head(0);
	state.doubleArray[1] = p_head(1);
	state.doubleArray[2] = v_head(0);
	state.doubleArray[3] = v_head(1);
	// For the angles since we used pointers it is already updated
}


double calculate_reward(const observation_t& state)
{
	const Vector2d v_head(state.doubleArray[2], state.doubleArray[3]);
	const double* p_angle = &state.doubleArray[4];
	const double* v_angle = &state.doubleArray[4 + n_seg];
	Vector2d v_points[n_seg+1];
	v_points[0] = v_head;
	for(size_t i=1; i<n_seg+1; i++){
		v_points[i] = v_points[i-1] + l_i*v_angle[i-1]*Vector2d(-sin(p_angle[i-1]), cos(p_angle[i-1]));
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