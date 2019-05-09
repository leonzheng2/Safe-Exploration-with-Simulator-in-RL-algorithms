"""
	ARS implementation for learning the swimmer task. RL Agent implementation in the RL Glue framework.
"""

import numpy as np
import sys
import copy
from statistics import 	stdev

from rlglue.agent.Agent import Agent
from rlglue.agent import AgentLoader as AgentLoader
from rlglue.types import Action
from rlglue.types import Observation
from rlglue.utils import TaskSpecVRLGLUE3

# Parameters
n_seg = 3
N = 10
H = 1000
b = N
alpha = 0.1

### TODO: V2 version

class SwimmerAgent(Agent):
	"""
		Variables
	"""
	
	# Iteration initial state: fixed during one iteration
	iterationObs = Observation()

	# States encountered from the start of the training
	states = []

	# Linear policy
	agentPolicy = np.zeros((n_seg-1, 2*(2+n_seg)))
	# n_seg-1 action variables
	# 2*(2+n_seg) state variables
	# A_0 is the head of the swimmer, 2D point; and there are n_seg angles. We want also the derivatives.

	# Perturbations for the 2N rollouts
	deltas = []

	# Rewards obtained at the end of the 2N rollouts
	rewards = []

	# Counter which increments only after one agent step
	count = 0


	"""
		RLGlue agent Methods
	"""
	
	def agent_init(self,taskSpec):

		# TODO Parse taskSpec
		TaskSpec = TaskSpecVRLGLUE3.TaskSpecParser(taskSpec)
		if TaskSpec.valid:
			print("Parsing task spec...")
			# assert len(TaskSpec.getIntObservations())==1, "expecting 1-dimensional discrete observations"
			# assert len(TaskSpec.getDoubleObservations())==0, "expecting no continuous observations"
			# assert not TaskSpec.isSpecial(TaskSpec.getIntObservations()[0][0]), " expecting min observation to be a number not a special value"
			# assert not TaskSpec.isSpecial(TaskSpec.getIntObservations()[0][1]), " expecting max observation to be a number not a special value"
			# self.numStates=TaskSpec.getIntObservations()[0][1]+1;

			# assert len(TaskSpec.getIntActions())==1, "expecting 1-dimensional discrete actions"
			# assert len(TaskSpec.getDoubleActions())==0, "expecting no continuous actions"
			# assert not TaskSpec.isSpecial(TaskSpec.getIntActions()[0][0]), " expecting min action to be a number not a special value"
			# assert not TaskSpec.isSpecial(TaskSpec.getIntActions()[0][1]), " expecting max action to be a number not a special value"
			# self.numActions=TaskSpec.getIntActions()[0][1]+1;
			
			# self.value_function=[self.numActions*[0.0] for i in range(self.numStates)]

		else:
			print("Task Spec could not be parsed: "+taskSpecString)

		# Initialization of variables
		self.iterationObs = Observation()
		self.states = []
		self.agentPolicy = np.zeros((n_seg-1, 2*(2+n_seg)))
		self.deltas = [np.zeros((n_seg-1, 2*(2+n_seg))) for i in range(N)]
		self.rewards = [0. for i in range(2*N)]
		self.count = 0
		
	def agent_start(self,observation):
		self.sample_deltas() # Choose the deltas directions
		iterationObs = copy.deepcopy(observation) # Fix the observation at the begining of the iteration
		for i in range(len(self.rewards)):
			self.rewards[i] = 0. # Reset rewards

		thisPolicy = self.fix_policy() # Select the policy for this agent step
		thisAction = self.select_action(thisPolicy, thisObservation) # Select action given the policy and the observation
		self.states.append(observation)
		self.count += 1

		return thisAction
	
	def agent_step(self,reward, observation):
		thisObservation = copy.deepcopy(observation) # In general case, the observation used to select the action is the last observation given by the environment

		# New rollout
		if self.count%H == 0:
			idx = self.count//H - 1 
			self.rewards[idx] = reward # Update reward. The current reward is the one received at the end of the previous rollout.
			thisObservation = copy.deepcopy(iterationObs) # At the begining of a new rollout, we start again with the initial observation.

		# New iteration
		if self.count%(2*N*H) == 0:
			order = self.order_directions()
			self.update_policy(order) # Use the previous rewards to update the policy. Only after the first iteration.
			self.sample_deltas() # Choose the deltas directions
			iterationObs = copy.deepcopy(observation) # Fix the observation at the begining of the iteration
			for i in range(len(self.rewards)):
				self.rewards[i] = 0. # Reset rewards
		
		thisPolicy = self.fix_policy() # Select the policy for this agent step
		thisAction = self.select_action(thisPolicy, thisObservation) # Select action given the policy and the observation
		self.states.append(observation)
		self.count += 1

		return thisAction
	
	def agent_end(self,reward):
		pass
	
	def agent_cleanup(self):
		pass
	
	def agent_message(self,inMessage):
		if inMessage=="what is your name?":
			return "my name is swimmer_agent, Python edition!";
		else:
			return "I don't know how to respond to your message";


	""" 
		Helper methods 
	"""	
	def fix_policy(self):
		"""
			Compute the policy given the current count
		"""

		# V1 and V1-t ARS policy
		idx = self.count%(2*N*H)
		rollout_idx = idx//H
		if rollout_idx%2==0:
			return self.agentPolicy + nu*self.deltas[rollout_idx//2]
		else:
			return self.agentPolicy - nu*self.deltas[rollout_idx//2]

		# TODO V2 and V2-t ARS policy

	def select_action(self, policy, state):
		"""
			Action selection based on linear policies
		"""
		state_vector = np.array(state.doubleArray)
		action_vector = np.matmul(policy*state_vector)
		action_selected = Action(numDoubles=action_vector.size)
		action_selected.doubleArray = action_vector.tolist()
		return action_selected

	def sample_deltas(self):
		"""
			Sample the N directions for rollouts with iid standard normal entries
		"""
		for i in range(len(self.deltas)):
			deltas[i] = np.random.rand(self.agentPolicy.shape[0], self.agentPolicy.shape[1])

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
		for i in range(b):
			used_rewards += [self.rewards[2*order[i]], self.rewards[2*order[i]+1]]
		sigma_r = stdev(used_rewards)
		
		grad = np.zeros(self.agentPolicy.shape)
		for i in range(b):
			grad += (self.rewards[2*order[i]]-self.rewards[2*order[i]+1]) * self.deltas[order[i]]
		grad /= (b*sigma_r)

		self.agentPolicy += alpha * grad

if __name__=="__main__":
	AgentLoader.loadAgent(SwimmerAgent())