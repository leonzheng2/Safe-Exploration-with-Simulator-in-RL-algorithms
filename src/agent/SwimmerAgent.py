"""
	ARS implementation for learning the swimmer task. RL Agent implementation in the RL Glue framework.
"""

import numpy as np
import sys
import copy
from statistics import stdev

from rlglue.agent.Agent import Agent
from rlglue.agent import AgentLoader as AgentLoader
from rlglue.types import Action
from rlglue.types import Observation
from rlglue.utils import TaskSpecVRLGLUE3

### TODO: V2 version

class SwimmerARSAgent(Agent):

	freeze = False

	"""
		RLGlue agent Methods
	"""
	
	def agent_init(self,taskSpec):

		print("Reading taskSpec: " + taskSpec.decode())

		# Parse taskSpec
		TaskSpec = TaskSpecVRLGLUE3.TaskSpecParser(taskSpec)
		if TaskSpec.valid:
			print("Parsing task spec...")
			self.max_u = TaskSpec.getDoubleActions()[0][1]
			self.n_action = len(TaskSpec.getDoubleActions())
			self.n_obs = len(TaskSpec.getDoubleObservations())
			print(f"Number of actions: {self.n_action}")
			print(f"Number of obs: {self.n_obs}")
			print("Task spec parsed!")
		else:
			print("Task Spec could not be parsed: "+taskSpecString)

		print("Initialization of training...")

		# Variables
		self.initial_state = Observation() # Iteration initial state: fixed during one iteration
		self.states = [] # States encountered from the start of the training
		self.agentPolicy = np.zeros((self.n_action, self.n_obs))
		self.deltas = [2*np.zeros((self.n_action, self.n_obs))-1 for i in range(self.N)]
		self.deltaPolicies = [self.agentPolicy for i in range(2*self.N)] # 2N policies for the 2N rollouts
		self.rewards = [0. for i in range(2*self.N)] # Rewards obtained at the end of the 2N rollouts
		self.count = 0 # Counter which increments only after one agent step
		self.ev_count = 0 # Counter for evaluation
		
		print("Training initialized!")

	def agent_start(self,observation):
		print("Starting agent...")
		assert len(observation.doubleArray) == self.n_obs

		self.sample_deltas() # Choose the deltas directions
		self.initial_state = copy.deepcopy(observation) # Fix the observation at the begining of the iteration
		for i in range(len(self.rewards)):
			self.rewards[i] = 0. # Reset rewards
		self.total_reward = 0.

		thisPolicy = self.fix_policy() # Select the policy for this agent step
		thisAction = self.select_action(thisPolicy, observation) # Select action given the policy and the observation
		print("Agent started!")

		return thisAction
	
	def agent_step(self,reward, observation):
		assert len(observation.doubleArray) == self.n_obs
		thisObservation = copy.deepcopy(observation) # In general case, the observation used to select the action is the last observation given by the environment
		self.total_reward += reward

		if not self.freeze:
			# New rollout
			if self.count%self.H == 0:
				idx = (self.count//self.H - 1)%(2*self.N) 
				self.rewards[idx] = self.total_reward # Update reward. The current reward is the one received at the end of the previous rollout.
				self.total_reward = 0.
				thisObservation = copy.deepcopy(self.initial_state) # At the begining of a new rollout, we start again with the initial observation.

		else:
			# New rollout
			if self.ev_count == 0:
				self.total_reward = 0.
				thisObservation = copy.deepcopy(self.initial_state) # At the begining of a new rollout, we start again with the initial observation.

		thisPolicy = self.fix_policy() # Select the policy for this agent step
		# print(f"Policy: {thisPolicy}")
		thisAction = self.select_action(thisPolicy, thisObservation) # Select action given the policy and the observation

		if not self.freeze:
			self.states.append(observation)
			self.count += 1
			# print(f"Count: {self.count}")

			# New iteration
			if self.count%(2*self.N*self.H) == 0:
				order = self.order_directions()
				self.update_policy(order) # Use the previous rewards to update the policy. Only after the first iteration.
				self.sample_deltas() # Choose the deltas directions
				for i in range(len(self.rewards)):
					self.rewards[i] = 0. # Reset rewards

		else:
			self.ev_count += 1
			# print(f"Evaluation Count: {self.ev_count}")
		
		return thisAction

	def agent_end(self,reward):
		pass
	
	def agent_cleanup(self):
		pass
	
	def agent_message(self,inMessageByte):
		inMessage = inMessageByte.decode()
		if inMessage=="what is your name?":
			return "my name is swimmer_agent, Python edition!";
		if inMessage=="freeze training":
			self.freeze = True
			self.ev_count = 0
			# print("===========Training is freezed===========")
			return "training is freezed"
		if inMessage=="unfreeze training":
			self.freeze = False
			# print("===========Training is unfreezed===========")
			return "training is unfreezed"
		if inMessage=="set parameters":
			self.set_parameters()
			s = f"Agent parameters are: N={self.N}; b={self.b}, H={self.H}, alpha={self.alpha}, nu={self.nu}"
			print(s)
			return s
		if inMessage=="get total_reward":
			print(f"Total reward is: {self.total_reward}")
			return str(self.total_reward)
		else:
			return "I don't know how to respond to your message";


	""" 
		Helper methods 
	"""	
	def fix_policy(self):
		"""
			Compute the policy given the current count
		"""
		if self.freeze:
			return self.agentPolicy

		# V1 and V1-t ARS policy
		idx = self.count%(2*self.N*self.H)
		rollout_idx = idx//self.H
		return self.deltaPolicies[rollout_idx]

		# TODO V2 and V2-t ARS policy

	def select_action(self, policy, state):
		"""
			Action selection based on linear policies
		"""
		state_vector = np.array(state.doubleArray)
		action_vector = np.matmul(policy, state_vector)

		# print(f"State: {state_vector}")
		# print(f"Action: {action_vector}")

		#Constraint
		for i in range(len(action_vector)):
			if action_vector[i] > self.max_u:
				action_vector[i] = self.max_u
			elif action_vector[i] < -self.max_u:
				action_vector[i] = -self.max_u

		action_selected = Action(numDoubles=action_vector.size)
		action_selected.doubleArray = action_vector.tolist()

		return action_selected

	def sample_deltas(self):
		"""
			Sample the N directions for rollouts with iid standard normal entries
		"""
		for i in range(len(self.deltas)):
			self.deltas[i] = np.random.rand(self.agentPolicy.shape[0], self.agentPolicy.shape[1])
			self.deltaPolicies[2*i] = self.agentPolicy + self.nu*self.deltas[i]
			self.deltaPolicies[2*i+1] = self.agentPolicy - self.nu*self.deltas[i]

	def order_directions(self):
		"""
			Sort the directions delta_k by max(reward[k,+], reward[k,-]). Return the array of indices corresponding to ordering from largest to smallest direction.
		"""
		# TODO
		return range(len(self.deltas))

	def update_policy(self, order):
		"""
			Update the policy after the end of the previous iteration
		"""
		used_rewards = []
		for i in range(self.b):
			used_rewards += [self.rewards[2*order[i]], self.rewards[2*order[i]+1]]
		sigma_r = stdev(used_rewards)
		
		grad = np.zeros(self.agentPolicy.shape)
		for i in range(self.b):
			grad += (self.rewards[2*order[i]]-self.rewards[2*order[i]+1]) * self.deltas[order[i]]
		grad /= (self.b*sigma_r)

		self.agentPolicy += self.alpha * grad

		print(f"Policy updated: {self.agentPolicy}")

	def set_parameters(self):
		f = open("../parameters.txt", "r")
		for line in f:
			tokens = line.split(" ")
			if(tokens[0]=="N"):
				self.N = int(tokens[1])
			elif(tokens[0]=="H"):
				self.H = int(tokens[1])
			elif(tokens[0]=="b"):
				self.b = int(tokens[1])
			elif(tokens[0]=="alpha"):
				self.alpha = float(tokens[1])
			elif(tokens[0]=="nu"):
				self.nu = float(tokens[1])
		f.close() 

if __name__=="__main__":
	AgentLoader.loadAgent(SwimmerARSAgent())
