from typing import Dict, Optional
from agents import Agent
from environment import Environment
from mdp import MarkovDecisionProcess, S, A
import json
import numpy as np
from copy import deepcopy

from helpers.utils import NotImplemented

# This is a class for a generic Policy Iteration agent
class PolicyIterationAgent(Agent[S, A]):
    mdp: MarkovDecisionProcess[S, A] # The MDP used by this agent for training
    policy: Dict[S, A]
    utilities: Dict[S, float] # The computed utilities
                                # The key is the string representation of the state and the value is the utility
    discount_factor: float # The discount factor (gamma)

    def __init__(self, mdp: MarkovDecisionProcess[S, A], discount_factor: float = 0.99) -> None:
        super().__init__()
        self.mdp = mdp
        # This initial policy will contain the first available action for each state,
        # except for terminal states where the policy should return None.
        self.policy = {
            state: (None if self.mdp.is_terminal(state) else self.mdp.get_actions(state)[0])
            for state in self.mdp.get_states()
        }
        self.utilities = {state:0 for state in self.mdp.get_states()} # We initialize all the utilities to be 0
        self.discount_factor = discount_factor
    
    # Given the utilities for the current policy, compute the new policy
    def update_policy(self):
        for state in self.mdp.get_states(): # get all utilities of all states
            if (self.mdp.is_terminal(state)): self.policy[state] = None # if state is terminal it has no action so action = None
            else:
                max_utitlity = float('-inf')
                best_action = None
                for action in self.mdp.get_actions(state): # for all actions with all states
                    sum_of_values = 0
                    for next_state,probability in self.mdp.get_successor(state,action).items(): # for all next states apply U(s)=max(sum(p*(R+gamma*U(s`))))
                        sum_of_values += probability*(self.mdp.get_reward(state,action,next_state)+self.discount_factor*self.utilities[next_state])
                    if sum_of_values > max_utitlity: # check for maximum utility
                        max_utitlity = sum_of_values 
                        best_action = action # pick the best action based on the best utility value
                self.policy[state] =best_action # change policy with the new action

    
    # Given the current policy, compute the utilities for this policy
    # Hint: you can use numpy to solve the linear equations. We recommend that you use numpy.linalg.lstsq
    #  linear_equations * utilities = y
    def update_utilities(self):
        n_of_states = len(self.mdp.get_states())
        linear_equations =[[0 for _ in range(n_of_states)] for _ in range(n_of_states)] # create initial matrix of coffienets all zeros
        y = [[0]for _ in range(n_of_states)] # same LHS values

        state_dict ={}
        for i,state in enumerate(self.mdp.get_states()): # create a dict of key:state value:index to access elements in matrices
            state_dict[state]=i
            linear_equations[i][i]=1 # make coffienet matrix an identity matrix in the same loop

        for state in self.mdp.get_states():
            action = self.policy[state] # use tha action taken from the policy
            if (action != None): # account for if state is terminal
                for next_state,probability in self.mdp.get_successor(state,action).items(): # get all successors
                    linear_equations[state_dict[state]][state_dict[next_state]]+=-1*self.discount_factor*probability # insert their cofficent based on their position in the matrix coff = -1*gamma*p(s`|s,policy(a))
                    y[state_dict[state]][0] +=self.mdp.get_reward(state,action,next_state)*probability # insert in RHS the constants where const = R*p
        
        ans = np.linalg.lstsq(linear_equations, y, rcond=None)[0] # calculate the unknowns matrix
        state_dict = { state_dict[k]:k for k in state_dict} # change state_dict to key:index, value:state
        for i in range(n_of_states): 
            self.utilities[state_dict[i]] = ans[i][0] # associate each state with its index to update the correct utility values

        
    # Applies a single utility update followed by a single policy update
    # then returns True if the policy has converged and False otherwise
    def update(self) -> bool:
        self.update_utilities() # first updating the utility values
        temp_policy = deepcopy(self.policy) # saving a copy of the orignal policy to check for converge
        self.update_policy() # update policy
        if temp_policy.items() == self.policy.items(): return True # check if converged
        else: return False

    # This function applies value iteration starting from the current utilities stored in the agent and stores the new utilities in the agent
    # NOTE: this function does incremental update and does not clear the utilities to 0 before running
    # In other words, calling train(M) followed by train(N) is equivalent to just calling train(N+M)
    def train(self, iterations: Optional[int] = None) -> int:
        iteration = 0
        while iterations is None or iteration < iterations:
            iteration += 1
            if self.update():
                break
        return iteration
    
    # Given an environment and a state, return the best action as guided by the learned utilities and the MDP
    # If the state is terminal, return None
    def act(self, env: Environment[S, A], state: S) -> A:
        # apply bellaman with all actions in the environement of the current state to produce the best action
        if (self.mdp.is_terminal(state)): return None
        max_utitlity = float('-inf')
        best_action = None
        for action in env.actions(): # loop on actions in environment
            sum_of_values = 0
            for next_state,probability in self.mdp.get_successor(state,action).items(): #apply U(s)=max(sum(p*(R+gamma*U(s`))))
                sum_of_values += probability*(self.mdp.get_reward(state,action,next_state)+self.discount_factor*self.utilities[next_state])
            if sum_of_values > max_utitlity: # check for maximum utility
                max_utitlity = sum_of_values 
                best_action = action # pick the best action based on the best utility value
        return best_action
        
    
    # Save the utilities to a json file
    def save(self, env: Environment[S, A], file_path: str):
        with open(file_path, 'w') as f:
            utilities = {self.mdp.format_state(state): value for state, value in self.utilities.items()}
            policy = {
                self.mdp.format_state(state): (None if action is None else self.mdp.format_action(action)) 
                for state, action in self.policy.items()
            }
            json.dump({
                "utilities": utilities,
                "policy": policy
            }, f, indent=2, sort_keys=True)
    
    # loads the utilities from a json file
    def load(self, env: Environment[S, A], file_path: str):
        with open(file_path, 'r') as f:
            data = json.load(f)
            self.utilities = {self.mdp.parse_state(state): value for state, value in data['utilities'].items()}
            self.policy = {
                self.mdp.parse_state(state): (None if action is None else self.mdp.parse_action(action)) 
                for state, action in data['policy'].items()
            }
