# 
# Copyright (C) 2008, Brian Tanner
# 
#http://rl-glue-ext.googlecode.com/
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import numpy as np
import random
import sys
import copy
from rlglue.agent.Agent import Agent
from rlglue.agent import AgentLoader as AgentLoader
from rlglue.types import Action
from rlglue.types import Observation

from random import Random

# Parameters
n_seg = 3
N = 10
H = 1000

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
	deltas = [np.zeros((n_seg-1, 2*(2+n_seg))) for i in range(N)]

	# Rewards obtained at the end of the 2N rollouts
	rewards = [0. for i in range(2*N)]

	# Counter which increments only after one agent step
	count = 0


	"""
		RLGlue agent Methods
	"""
	
	def agent_init(self,taskSpec):
		#See the sample_sarsa_agent in the mines-sarsa-example project for how to parse the task spec
		self.lastAction=Action()
		self.lastObservation=Observation()

		# Reset count
		count = 0
		
	def agent_start(self,observation):
		raise NotImplementedError
	
	def agent_step(self,reward, observation):
		thisPolicy = fix_policy() # Select the policy for this agent step
		thisObservation = copy.deepcopy(observation) # In general case, the observation used to select the action is the last observation given by the environment

		if new_rollout():
			idx = (count-H)/H
			rewards[idx] = reward # Update reward. The current reward is the one received at the end of the previous rollout.
			thisObservation = copy.deepcopy(iterationObs) # At the begining of a new rollout, we start again with the initial observation.

		if new_iteration():
			if count>0:
				update_policy() # Use the previous rewards to update the policy. Only after the first iteration.
			sample_deltas() # Choose the deltas directions
			iterationObs = copy.deepcopy(observation) # Fix the observation at the begining of the iteration
			for i in range(len(rewards)):
				rewards[i] = 0. # Reset rewards
		
		thisAction = select_action(thisPolicy, thisObservation) # Select action given the policy and the observation
		states.append(observation)
		count += 1

		return thisAction
	
	def agent_end(self,reward):
		pass
	
	def agent_cleanup(self):
		pass
	
	def agent_message(self,inMessage):
		if inMessage=="what is your name?":
			return "my name is skeleton_agent, Python edition!";
		else:
			return "I don't know how to respond to your message";


	""" 
		Helper methods 
	"""	
	def new_rollout(self):
		return count % H == 0

	def new_iteration(self):
		return count % (2*N*H) == 0

	def fix_policy(self):
		"""
			Compute the policy given the current count
		"""
		raise NotImplementedError

	def select_action(self, policy, state):
		"""
			Action selection based on linear policies
		"""
		raise NotImplementedError

	def sample_deltas(self):
		"""
			Sample the N directions for rollouts with iid standard normal entries
		"""
		raise NotImplementedError

	def update_policy(self):
		"""
			Update the policy after the end of the previous iteration
		"""
		raise NotImplementedError


if __name__=="__main__":
	AgentLoader.loadAgent(SwimmerAgent())