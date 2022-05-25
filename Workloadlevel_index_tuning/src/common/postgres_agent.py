# SPG
import os, sys
from re import A
head, tail = os.path.split(__file__)
sys.path.insert(0, os.path.join(head, '../')) # /src

# DQN
from rlgraph.agents import Agent

sys.path.insert(0, os.path.join(head, '../../..')) # lift
from lift.lift.rl_model.task import Task # update
from lift.lift.rl_model.task_graph import TaskGraph
import numpy as np

import torch

class PostgresAgent():
    """
        Wrapper for DQN agent (from rlgraph) or SPG agent,
        a standard api that can be called by system controller
    
    """

    def __init__(self, agent_config=None, experiment_config=None, schema=None):
        
        self.agent_config = agent_config
        


            # update: wrapper wraps (Lift) TaskGraph which has a (Lift) Task which has a (Rlgraph) Agent
        self.task_graph = TaskGraph()
        task = Task(self.agent_config, 
                        state_space=schema.get_states_spec(), 
                        action_space=schema.get_actions_spec())
        self.task_graph.add_task(task)

            # TODO DELETE
            # self.agent = Agent.from_spec(
            #     self.agent_config,
            #     state_space=states_spec,
            #     action_space=actions_spec
            # )
        
            
    def get_action(self, agent_state):

        return self.task_graph.act_task("", states=np.asarray(agent_state.get_value()), apply_preprocessing=True)
            #self.agent.get_action(agent_state.as_array(), apply_preprocessing=True)



    def observe(self, agent_state, agent_action, agent_reward, next_agent_state=None, terminal=None):
        """
        Args:
            agent_state: both
            agent_action: both
            agent_reward: both
            next_agent_state: DQN-specific
            terminal: DQN-specific
        """
        

        self.task_graph.observe_task("", np.asarray(agent_state.get_value()), agent_action, [], agent_reward, np.asarray(next_agent_state.get_value()), terminal)
        (task_loss, _)=self.task_graph.update_task(name="")
        return task_loss
            #self.agent.observe(agent_state.as_array(), agent_action, [], agent_reward, next_agent_state.as_array(), terminal)
        

    

    def load_model(self, path):
        """
        TODO clean up
        """
  
        self.task_graph.load_model('', path) # path to agent



    def save_model(self, path):
        """
        TODO
        """

        self.task_graph.store_model('', os.path.join(path, 'dqn')) # path to agent dir
