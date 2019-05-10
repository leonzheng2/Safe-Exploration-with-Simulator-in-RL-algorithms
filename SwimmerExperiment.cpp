#include <iostream>   // for cout
#include <stdio.h>
#include <string.h>

#include <rlglue/RL_glue.h> /* RL_ function prototypes and RL-Glue types */
	
int whichEpisode=0;

// /* Run One Episode of length maximum cutOff*/
// void runEpisode(int stepLimit) {        
//     int terminal=RL_episode(stepLimit);
// 	printf("Episode %d\t %d steps \t%f total reward\t %d natural end \n",whichEpisode,RL_num_steps(),RL_return(), terminal);
// 	whichEpisode++;
// }


// Parameters for the experience
const size_t n_it = 100;

// Parameters from the agent
const size_t H = 500;
const size_t N = 1;

// Variables for experience
const char* task_spec;
const char* responseMessage;
const reward_observation_action_terminal_t *stepResponse;
const observation_action_t *startResponse;		

void sendBasicMessages(){
	std::cout << "\n\n----------Sending some sample messages----------\n" << std::endl;
	/*Talk to the agent and environment a bit...*/
	responseMessage=RL_agent_message("what is your name?");
	std::cout << "Agent responded to \"what is your name?\" with: " << responseMessage << std::endl;
	responseMessage=RL_agent_message("If at first you don't succeed; call it version 1.0");
	std::cout << "Agent responded to \"If at first you don't succeed; call it version 1.0\" with: " << responseMessage << std::endl;

	responseMessage=RL_env_message("what is your name?");
	std::cout << "Environment responded to \"what is your name?\" with: " << responseMessage << std::endl;
	responseMessage=RL_env_message("If at first you don't succeed; call it version 1.0");
	std::cout << "Environment responded to \"If at first you don't succeed; call it version 1.0\" with: " << responseMessage << std::endl;
}

void runOneTrainingIteration(){
	const size_t steps = 2*H*N;
	for(size_t i=0; i<steps; i++){
		stepResponse=RL_step();
		if(i%100==0){
			// Print head position
			std::cout << "(" << stepResponse->observation->doubleArray[0] << "; " << stepResponse->observation->doubleArray[1] << ") and reward is " << stepResponse->reward << std::endl; 
		}
	}
}

void run_training(){
	std::cout << "\n\n----------Augmented Random Search training----------\n" << std::endl;
	responseMessage=RL_env_message("activate training");
	std::cout << "Agent responded to \"activate training\" with: " << responseMessage << std::endl;
	RL_init();

	std::cout << "Starting the training..." << std::endl;
	startResponse = RL_start();
	std::cout << "First observation is (only head): (" << startResponse->observation->doubleArray[0] << "; " << startResponse->observation->doubleArray[1] << ")" << std::endl; 

	std::cout << "Running training iterations..." << std::endl;
	for(size_t i=0; i<n_it; i++){
		runOneTrainingIteration();
	}
	std::cout << "End of the training" << std::endl;
}

void runOneEvaluationIteration(size_t current_it){
	const size_t steps = H;
	for(size_t i=0; i<steps; i++){
		stepResponse=RL_step();
	}
	std::cout << "Reward for one rollout with policy at iteration " << current_it << ": " << stepResponse->reward << std::endl;
}

void run_evaluation(){
	std::cout << "\n\n----------Evaluation----------\n" << std::endl;
	responseMessage=RL_env_message("activate evaluation");
	std::cout << "Agent responded to \"activate evaluation\" with: " << responseMessage << std::endl;
	RL_init();

	std::cout << "Starting the evaluation..." << std::endl;
	startResponse = RL_start();
	std::cout << "First observation is (only head): (" << startResponse->observation->doubleArray[0] << "; " << startResponse->observation->doubleArray[1] << ")" << std::endl; 

	std::cout << "Running evaluation iterations..." << std::endl;
	for(size_t i=0; i<n_it; i++){
		runOneEvaluationIteration(i);
	}
	std::cout << "End of the evaluation" << std::endl;

}

int main(int argc, char *argv[]) {
	std::cout << "\n\nExperiment starting up!\n" << std::endl;

	task_spec=RL_init();
	std::cout << "RL_init called, the environment sent task spec: " << task_spec << std::endl;

	sendBasicMessages();
	run_training();
	run_evaluation();

	RL_cleanup();

	return 0;
}
