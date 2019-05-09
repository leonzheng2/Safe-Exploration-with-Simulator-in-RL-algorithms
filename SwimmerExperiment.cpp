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
const size_t n_it = 10;

// Parameters from the agent
const size_t H = 500;
const size_t N = 5;

// Variables for experience
const char* task_spec;
const char* responseMessage;
const reward_observation_action_terminal_t *stepResponse;
const observation_action_t *startResponse;		

void runOneIteration(){
	const size_t steps = 2*H*N;
	for(size_t i=0; i<steps; i++){
		stepResponse=RL_step();
		if(i%100==0){
			// Print head position
			std::cout << "(" << stepResponse->observation->doubleArray[0] << "; " << stepResponse->observation->doubleArray[1] << ") and reward is " << stepResponse->reward << std::endl; 
		}
	}
}

int main(int argc, char *argv[]) {

	std::cout << "\n\nExperiment starting up!\n" << std::endl;

	task_spec=RL_init();
	std::cout << "RL_init called, the environment sent task spec: " << task_spec << std::endl;

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

	std::cout << "\n\n----------Augmented Random Search training----------\n" << std::endl;
	std::cout << "Starting the environment..." << std::endl;
	startResponse = RL_start();
	std::cout << "First observation is (only head): (" << startResponse->observation->doubleArray[0] << "; " << startResponse->observation->doubleArray[1] << ")" << std::endl; 

	std::cout << "Running iterations..." << std::endl;
	for(size_t i=0; i<n_it; i++){
		runOneIteration();
	}


	// runEpisode(100);
	// runEpisode(100);
	// runEpisode(100);
	// runEpisode(100);
	// runEpisode(100);
	// runEpisode(1);
	// /* Remember that stepLimit of 0 means there is no limit at all!*/
	// runEpisode(0);
	// RL_cleanup();

	// printf("\n\n----------Stepping through an episode----------\n");
	// /*We could also start over and do another experiment */
	// task_spec=RL_init();

	// /*We could run one step at a time instead of one episode at a time */
	// /*Start the episode */
	// startResponse=RL_start();
	// printf("First observation and action were: %d %d\n",startResponse->observation->intArray[0],startResponse->action->intArray[0]);

	// /*Run one step */
	// stepResponse=RL_step();
	
	// /*Run until the episode ends*/
	// while(stepResponse->terminal!=1){
	// 	stepResponse=RL_step();
	// 	if(stepResponse->terminal!=1){
	// 		/*Could optionally print state,action pairs */
	// 		/*printf("(%d,%d) ",stepResponse.o.intArray[0],stepResponse.a.intArray[0]);*/
	// 	}
	// }
	
	// printf("\n\n----------Summary----------\n");
	

	// printf("It ran for %d steps, total reward was: %f\n",RL_num_steps(),RL_return());
	RL_cleanup();


	return 0;
}
