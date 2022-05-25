import os, sys 
head, tail = os.path.split(__file__)
sys.path.insert(0, os.path.join(head, '../../..'))  # lift
from lift.lift.rl_model.converter import Converter
from lift.lift.rl_model.state import State
from sklearn import preprocessing

import numpy as np


class PostgresConverter(Converter):

    def __init__(self, experiment_config, schema):
        """
        Converts between action / reward / state representations for
        rlgraph agent (i.e. inputs / outputs of DQN) and for db

        Boilerplated from lift/case_studies/{mongodb,mysql}/ 

        Args:
            experiment_config (dict): paramters for a particular experiment 
            schema (PostgresSchema): representations required by agent 
        """
        self.experiment_config = experiment_config
        self.schema = schema

        # reward
        self.max_size = experiment_config['max_size']
        self.size_weight = experiment_config['size_weight']
        self.max_runtime = experiment_config['max_runtime']
        self.runtime_weight = experiment_config['runtime_weight']
        self.reward_penalty = experiment_config['reward_penalty']
        
        self.system_spec = schema.get_system_spec()
        self.actions_spec = schema.get_actions_spec()
        self.max_columns_per_index = len(self.actions_spec)

        # tokens
        self.pad = self.system_spec['pad_token']
        self.index_token = self.system_spec['index_token']
        self.vocab = self.system_spec['vocab']
        self.noop_idx = self.system_spec['noop_idx']

    def system_to_agent_action(self):

        pass

    def agent_to_system_action(self, actions, meta_data=None):
        """
        Translates agent's Q-network actions into specific index fields to index on
        (to be executed by environment)

        ASC / DESC not taken into account

        Args: 
            actions (dict): TODO
        meta_data (dict): TODO

        """

        index_cols = [] # query attributes to index on
        query_cols = meta_data["query_cols"] # query attributes

        # for candidate index column
        for key in self.actions_spec.keys():
            action_val = actions[key]
        
            # check for batch_dim 
            if isinstance(action_val, (list, np.ndarray)):
                action_val = action_val[0]

            if action_val != self.noop_idx:

                # action space per candidate index column is an integer in [0,3] for 3 query attributes
                # suppose query has 1 query attribute, but Q-network returns an action of 2. 
                # this is not a valid action! 
                # we want to filter these out, so Q-network will train over time not to do this  
                if len(query_cols) > action_val - 1:
                    col = query_cols[action_val - 1]
                    if col not in index_cols: 
                        index_cols.append(col)
        
        return index_cols 

    def system_to_agent_reward(self, meta_data):
        """
        Reward as a weighted average of (index) space and (query under index) time
        """

        runtime = meta_data["runtime"]
        index_size = meta_data["index_size"]
        adj = meta_data["adj"]

        reward = - (self.runtime_weight * runtime) - (self.size_weight * index_size)

        # TODO experimental
        if adj: 
            reward *= 1.25

        if runtime > self.max_runtime:
            reward -= self.reward_penalty

        if index_size > self.max_size:
            reward -= self.reward_penalty

        return reward


    def workload_concat(self,workload):
    
        w_0 = workload[0]

        for q in workload:
            w_0 = w_0+q

        return w_0


    def system_to_agent_state(self, workload, system_context):
        """
        Converts a SQL query and a system context to an agent state,
        i.e. into an integer representation that's input to the agent's Q-network

        Args:
            query (SQLQuery): query with tokenized query representation 
            system_context (dict): currently indexed columns
        Returns:
            State 
        """

        tokened_w = [query.as_tokens().copy() for query in workload]
        workload_tokens = self.workload_concat(tokened_w)
        
        query_cols = [query.query_cols for query in workload]
        workload_cols = self.workload_concat(query_cols)
        workload_cols = list(set(workload_cols)) #TO DO, the order of state tokens
        
        # TBD/TO BE DELETED 
        # system_context is an iterable of compound index attributes e.g. [[foo], [bar,baz]],
        # currently, state tokenization strategy for system_context is to include [[foo], [bar,baz]] as [foo_idx, bar_idx, baz_idx]
        # flattening system_context like this throws away important information! 

        # context_tokens = [] 
        # for index in system_context["indices"]:
        #     context_tokens.extend(list(map(lambda col: col + '_idx', index))) 
        
        # Updates:
        # 1. state should contain only context (i.e. compound indices) with columns represented in query columns.
        # 2. token to separate compound indices
        #        not sure whether this actually allows structure to be recognized
        # e.g. 
        #     query.query_cols == [foo]
        #     system_context == [[baz], [foo, bar], [foo]]
        #     token representation of state (context component of state) is just [idx foo bar idx foo]
        # TODO do even further filtering based on prefix
        # i.e. do we bother including index [foo, bar, baz] if query.query_cols == [baz]?
        
        context_tokens = []
        for index in system_context['indices']:
            if [col for col in index if col in workload_cols] != []:
                context_tokens.append(self.index_token)
                context_tokens.extend(list(map(lambda col: col + '_idx', index)))
            

        input_tokens = workload_tokens + context_tokens
        indexed_input_tokens = [self.vocab[input_token] for input_token in input_tokens]
        
        pad = self.schema.input_sequence_size - len(indexed_input_tokens)
        if pad > 0:
            state = indexed_input_tokens + [self.vocab[self.pad]] * pad
        else:
            state = indexed_input_tokens[:self.schema.input_sequence_size]
        
        return State(value=np.asarray(state), meta_data=dict(query_columns=workload_cols))
        
    
