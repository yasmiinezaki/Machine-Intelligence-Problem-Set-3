from typing import Dict, Optional
from agents import Agent
from environment import Environment
from mdp import MarkovDecisionProcess, S, A
import json

from helpers.utils import NotImplemented

# This is a class for a generic Value Iteration agent
class ValueIterationAgent(Agent[S, A]):
    mdp: MarkovDecisionProcess[S, A] # The MDP used by this agent for training 
    utilities: Dict[S, float] # The computed utilities
                                # The key is the string representation of the state and the value is the utility
    discount_factor: float # The discount factor (gamma)

    def __init__(self, mdp: MarkovDecisionProcess[S, A], discount_factor: float = 0.99) -> None:
        super().__init__()
        self.mdp = mdp
        self.utilities = {state:0 for state in self.mdp.get_states()} # We initialize all the utilities to be 0
        self.discount_factor = discount_factor
    
    # Given a state, compute its utility using the bellman equation
    # if the state is terminal, return 0
    def compute_bellman(self, state: S) -> float:
        if (self.mdp.is_terminal(state)): return 0
        max_utitlity = float('-inf')
        for action in self.mdp.get_actions(state):
            sum_of_values = 0 # used to sum all q_values for the current state and action
            for next_state,probability in self.mdp.get_successor(state,action).items():
                # average of all next_state utilities
                sum_of_values += probability*(self.mdp.get_reward(state,action,next_state)+self.discount_factor*self.utilities[next_state])
            max_utitlity = max(sum_of_values,max_utitlity) # choose maximum to apply as utility for the current state
        return max_utitlity

    # Applies a single utility update
    # then returns True if the utilities has converged (the maximum utility change is less or equal the tolerance)
    # and False otherwise
    def update(self, tolerance: float = 0) -> bool:
        diff = 0
        update_utilites = {}
        for state in self.mdp.get_states():
            old_utility = self.utilities[state]
            new_utility = self.compute_bellman(state) # compute bellman once for each state 
            diff = max(abs(old_utility - new_utility),diff) # calculate the difference between each utility and the next and take the overall maximum difference
            update_utilites[state] = new_utility # update the utility with the new one
        self.utilities = update_utilites # apply changes
        if diff <= tolerance: return True # if values converged return true
        else: return False 

    # This function applies value iteration starting from the current utilities stored in the agent and stores the new utilities in the agent
    # NOTE: this function does incremental update and does not clear the utilities to 0 before running
    # In other words, calling train(M) followed by train(N) is equivalent to just calling train(N+M)
    def train(self, iterations: Optional[int] = None, tolerance: float = 0) -> int:
        iteration = 0
        while iterations is None or iteration < iterations:
            iteration += 1
            if self.update(tolerance):
                break
        return iteration
    
    # Given an environment and a state, return the best action as guided by the learned utilities and the MDP
    # If the state is terminal, return None
    def act(self, env: Environment[S, A], state: S) -> A:
        # apply bellaman with all actions in the environement of the current state to produce the best action
        if (self.mdp.is_terminal(state)): return None
        max_utitlity = float('-inf')
        best_action = None
        for action in env.actions():
            sum_of_values = 0
            for next_state,probability in self.mdp.get_successor(state,action).items():
                sum_of_values += probability*(self.mdp.get_reward(state,action,next_state)+self.discount_factor*self.utilities[next_state])
            if sum_of_values > max_utitlity: # check for maximum utility
                max_utitlity = sum_of_values 
                best_action = action # pick the best action based on the best utility value
        return best_action
        
    
    # Save the utilities to a json file
    def save(self, env: Environment[S, A], file_path: str):
        with open(file_path, 'w') as f:
            utilities = {self.mdp.format_state(state): value for state, value in self.utilities.items()}
            json.dump(utilities, f, indent=2, sort_keys=True)
    
    # loads the utilities from a json file
    def load(self, env: Environment[S, A], file_path: str):
        with open(file_path, 'r') as f:
            utilities = json.load(f)
            self.utilities = {self.mdp.parse_state(state): value for state, value in utilities.items()}
