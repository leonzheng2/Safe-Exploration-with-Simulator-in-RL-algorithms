#include <iostream>   // for cout
#include <fstream>
#include <stdio.h>
#include <string.h>
#include <sstream>  

#include <rlglue/RL_glue.h> /* RL_ function prototypes and RL-Glue types */
	
// Parameters for the experience
size_t n_it = 1000;

// Parameters from the agent
size_t N;
size_t H;

// Variables for experience
const char* task_spec;
const char* responseMessage;
const reward_observation_action_terminal_t *stepResponse;
const observation_action_t *startResponse;		

// Output results
std::ofstream results;

void sendBasicMessages()
{
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

void setParameters()
{
	std::cout << "\n\n----------Setting environment and agent parameters----------\n" << std::endl;
	responseMessage=RL_env_message("set parameters");
	std::cout << "Environment response: " << responseMessage << std::endl;
	responseMessage=RL_agent_message("set parameters");
	std::cout << "Agent response: " << responseMessage << std::endl;

	std::string line;
	std::ifstream inFile("../parameters.txt");

	if (inFile.is_open()) {
		while (getline(inFile, line)) {
	    	std::stringstream ss(line);
	    	std::string varName;
	    	ss >> varName;
			if(varName=="N") ss >> N;
			else if(varName=="H") ss >> H;
	    }
	    inFile.close();
	}
	else std::cout << "Unable to open file for setting parameters for experiment"; 
    std::cout << "Experiment parameters are: N=" + std::to_string(N) + "; H=" + std::to_string(H) << std::endl;
}

void runOneTrainingIteration(size_t current_it)
{
	RL_agent_message("unfreeze training");
	for(size_t i=0; i<2*H*N; i++){
		if(i%H == 0){
			RL_env_message("load state");
		}
		stepResponse=RL_step();
	}
	RL_agent_message("freeze training");
	RL_env_message("load state");
	for(size_t i=0; i<H; i++){
		stepResponse=RL_step();
	}
	responseMessage = RL_agent_message("get total_reward");
	const double total_reward = std::stod(responseMessage);

	std::cout << "Reward for one rollout with policy at iteration " << current_it << ": " << total_reward << std::endl;
	results << "Reward for one rollout with policy at iteration " << current_it << ": " << total_reward << "\n";
}

void run_training()
{
	std::cout << "\n\n----------Augmented Random Search training----------\n" << std::endl;


	std::cout << "Starting the training..." << std::endl;
	startResponse = RL_start();
	std::cout << "First observation is (only head): (" << startResponse->observation->doubleArray[0] << "; " << startResponse->observation->doubleArray[1] << ")" << std::endl; 

	std::cout << "Running training iterations..." << std::endl;
	for(size_t i=0; i<n_it; i++){
		runOneTrainingIteration(i);
	}
	std::cout << "End of the training" << std::endl;
}

int main(int argc, char *argv[])
{
	results.open("../plot/results.txt");
	std::cout << "\nExperiment starting up!\n" << std::endl;
	if(argc > 1){
		n_it = std::stoi(argv[1]);
		std::cout << "Iteration number is " << n_it << std::endl;
	}

	setParameters();
	task_spec=RL_init();
	std::cout << "RL_init called, the environment sent task spec: " << task_spec << std::endl;

	sendBasicMessages();
	run_training();

	RL_cleanup();
	results.close();

	return 0;
}
