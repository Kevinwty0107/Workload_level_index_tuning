#!/usr/local/bin/python3

from curses import flash
import os, sys
from tpch_workload import sort_workload
head, tail = os.path.split(__file__)
sys.path.insert(0, os.path.join(head, '../../..')) # lift
from lift.lift.controller.system_controller import SystemController

sys.path.insert(0, os.path.join(head, '..')) # src
from agent.postgres_converter import PostgresConverter as PostgresDQNConverter
from agent.postgres_schema import PostgresSchema as PostgresDQNSchema


from workload_processor import workload_processor
from postgres_agent import PostgresAgent
from postgres_data_source import PostgresDataSource
from postgres_system_environment import Action, PostgresSystemEnvironment
from tpch_workload import TPCHWorkload, sort_workload
from tpch_util import tpch_table_columns

import copy
import csv
import itertools
import numpy as np
import pdb
import pickle
import time
from tensorboardX import SummaryWriter

class PostgresSystemController(SystemController):


    def __init__(self, 
                 agent_config, 
                 experiment_config, 
                 schema_config,
                 result_dir):
        """
        
        Args:
            dqn (bool): DQN agent, else SPG agent; required where rlgraph breaks abstraction barrier (e.g. resetting)
            ...

             TODO 'config' and 'spec' conflated throughout code
        """


        super().__init__(agent_config=agent_config, 
                         experiment_config=experiment_config,
                         result_dir=result_dir)    

        #
        # schema, for agent state and action representations 
        # 

        self.agent_config =  agent_config
        self.experiment_config = experiment_config
        self.result_dir = result_dir

        schema_config['tables'] = experiment_config['tables']
        self.schema_config = schema_config
        

        self.schema = PostgresDQNSchema(schema_config)

        self.states_spec = self.schema.get_states_spec()
        self.actions_spec = self.schema.get_actions_spec()
        self.system_spec = self.schema.get_system_spec()
        
        self.agent_config['network_spec'][0]['vocab_size'] = self.system_spec['vocab_size'] # TODO

        #
        # converter
        #
        self.converter = PostgresDQNConverter(experiment_config=experiment_config, schema=self.schema)


        #
        # workload
        # 
        n_selections = experiment_config.get("n_selections", 3)
                
        self.n_executions = experiment_config['n_executions']
        self.n_train_episodes = experiment_config["n_train_episodes"]
        self.n_test_episodes = experiment_config["n_test_episodes"]
        self.n_workloads_per_episode = experiment_config["n_workloads_per_episode"]
        self.n_queries_per_workload = experiment_config["n_queries_per_workload"]

        workload_spec = {
            "tables": self.schema_config['tables'],
            "scale_factor": 1 , # TODO specify this somewhere, not just in tpch_util 
            "n_selections": n_selections
        }
        self.workload = TPCHWorkload(spec=workload_spec)

        #
        # workload ser/des
        #
        self.data_source = PostgresDataSource(workload_spec)

        #
        # environment
        #
        self.system_environment = PostgresSystemEnvironment(tbls=self.experiment_config['tables'])
        
        self.logger.info('computing column selectivities...')
        start = time.monotonic()
        try:
            with open(os.path.join(result_dir, 'selectivities.pkl'), 'rb') as f:
                self.system_environment.tbl_2_col_2_sel = pickle.load(f)
        except:
            self.system_environment.compute_column_selectivity()
            with open(os.path.join(result_dir, 'selectivities.pkl'), 'wb') as f:
                pickle.dump(self.system_environment.tbl_2_col_2_sel, f)
        self.logger.info('...took {:.2f} seconds'.format(time.monotonic() - start))

        #
        # agent
        #
        self.agent_api = PostgresAgent(
            agent_config=self.agent_config,
            experiment_config=self.experiment_config,
            schema = self.schema
        )
        

        self.task_graph = self.agent_api.task_graph # have to have this exposed...
        self.set_update_schedule(self.agent_config["update_spec"])


    def workload_concat(self,workload):
        
        w_0 = workload[0]

        for q in workload:
            w_0 = w_0+q

        return w_0


    def train(self,workloads):

        # meta-data
        result_dir = os.path.join(self.result_dir, 'dqn')
        if not os.path.exists(result_dir): os.makedirs(result_dir)

        tensorboard = SummaryWriter(os.path.join(result_dir, 'tensorboard'))
        step = 0

        # record per step 
        times = [] # total time, roughly decomposes into...
        agent_times = [] # time spent retrieving action, updating based on retrieved action
        workload_times = [] # time spent on workloads
        query_times = []# time spent on queries
        avg_query_times = [] # avg time spent on queries
        system_times = [] # time spent on transitioning state (i.e. indexing)
        rewards = [] # for trend over time
        selectivities = []
        intersections = []
        losses = []

        index_set_stats = []

        self.task_graph.get_task("").unwrap().timesteps = 0 

        workloads = self.group_by_q_len(workloads)

        for episode_idx in range(self.n_train_episodes):

            self.logger.info('Starting episode {}/{}'.format(episode_idx+1, self.n_train_episodes))

            # meta-data
            action_results = dict.fromkeys(['noop', 'duplicate_index', 'index'], 0)
            action_results_szs = []  

            episode_workloads = workloads[episode_idx*self.n_workloads_per_episode:(episode_idx+1) * self.n_workloads_per_episode]
            episode_workloads = sort_workload(episode_workloads)

            terminal_workload_idx = len(episode_workloads)-1
            context = []

            self.task_graph.reset()
            self.system_environment.reset()

            for workload_idx, workload in enumerate(episode_workloads):

                start = time.monotonic()

                if workload_idx+1 %5 ==0:
                    self.logger.info('Completed {}/{} workloads'.format(workload_idx+1, self.n_workloads_per_episode))

                ##
                ## get state
                ##

                acting_start = time.monotonic()

                agent_state= self.converter.system_to_agent_state(workload=workload,
                                                                  system_context=dict(indices=context))

                agent_action = self.agent_api.get_action(agent_state)

                query_cols = [query.query_cols for query in workload]
                workload_cols = self.workload_concat(query_cols)
                workload_cols = list(set(workload_cols)) 
                

                system_action = self.converter.agent_to_system_action(actions = agent_action,
                                                                     meta_data=dict(query_cols=workload_cols))

                

                query_tbls = [query.query_tbl for query in workload]
                #workload_tbls = self.workload_concat(query_tbls)
                workload_tbls = list(set(query_tbls))

                #Take action

                system_time, action_result = self.system_environment.act(dict(index=system_action, table=workload_tbls[0] ))
                acting_time = time.monotonic()-acting_start
                action_results[action_result] += 1
                action_results_szs.append(len(system_action))

                workload_time, avg_query_runtime, query_time, explained_idx = self.system_environment.execute(workload)

                #record info about intxns
                existing_idx = self.get_intxn_opp(workload, context) # check before updating context
                intxn_opp = 1 if existing_idx != [] else 0
                intxn_opp_taken = 1 if system_action == [] and explained_idx in [existing_idx[:i+1] for i in range(len(existing_idx))] else 0 # TODO explained_idx is sourced from EXPLAIN ANALYZE
                                                                                                                                              # and its cols (i.e. cols after Index Cond:) come from a subset of or complete index actually used
                                                                                                                                              # so existing_idx and explained_idx can be wrongfully equal if >1 index share those cols, though this is rare
                intersections.append([intxn_opp, intxn_opp_taken])
                self.logger.info('workload cols: {} system action: {} explained_idx: {} existing_idx: {}'.format(list( workload_cols), system_action, explained_idx, existing_idx))
                # e.g. intxn_opp_taken == 1: INFO / postgres_controller / query cols: ['L_DISCOUNT', 'L_EXTENDEDPRICE'] system action: [] explained_idx: ['L_EXTENDEDPRICE'] existing_idx: ['L_EXTENDEDPRICE', 'L_SHIPINSTRUCT']
                
                # reward a couple specific cases
                context_unaware = True if system_action != [] and explained_idx in [existing_idx[:i+1] for i in range(len(existing_idx))] else False
                column_unaware = True if system_action != [] and explained_idx not in [system_action[:i+1] for i in range(len(system_action))] else False

                index_set_size, _ = self.system_environment.system_status()


                ##
                # update agent
                ##

                updating_start = time.monotonic()
                agent_reward = self.converter.system_to_agent_reward(meta_data=dict(runtime=workload_time,
                                                                                    index_size=index_set_size ,
                                                                                    adj = context_unaware or column_unaware))
                context.append(system_action)
                
                next_agent_state = terminal = None

                #if dqn

                terminal = workload_idx == terminal_workload_idx
                next_workload = episode_workloads[workload_idx+1 if not terminal else workload_idx]
                next_agent_state = self.converter.system_to_agent_state(workload=next_workload,system_context=dict(indices=context))

                loss = self.agent_api.observe(agent_state, agent_action, agent_reward, next_agent_state=next_agent_state, terminal=terminal)

                #if dqn
                losses.append(loss)
                tensorboard.add_scalar('scalars/loss', loss, step)
                step +=1

                #self.update_if_necessary()

                updating_time = time.monotonic()-updating_start

                #record results per step of episode

                times.append(time.monotonic()-start)
                agent_times.append(acting_time+updating_time)
                query_times.extend(query_time)
                avg_query_times.append(avg_query_runtime)
                workload_times.append(workload_time)
                system_times.append(system_time)


                rewards.append(agent_reward)
                selectivities.append((self.get_selectivity(workload_tbls[0], workload_cols), # avg selectivity of query cols
                                      self.get_selectivity(workload_tbls[0], system_action),    # avg selectivity of index cols in index suggested by agent
                                      self.get_selectivity(workload_tbls[0], explained_idx)))   # avg selectivity of index cols in index actually used

            # record results per episode
            index_set_stats.append((action_results, np.mean(action_results_szs),index_set_size )) 

            # pretty print a summary
            self.logger.info("Completed episode: " \
                             "actions taken: " + ('{}:{:.2f} ' * len(tuple(action_results.items()))).format(*tuple(field for tup in action_results.items() for field in tup)) + \
                             "avg reward per step: {:.2f} ".format(np.mean(rewards[-self.n_workloads_per_episode:])) + \
                             "avg workload runtime: {:.2f} ".format(np.mean(workload_times[-self.n_workloads_per_episode:])) + \
                             "avg query runtime: {:.2f} ".format(np.mean(avg_query_times[-self.n_workloads_per_episode:])) + \
                             "intxn opps: {}, opps taken: {} ".format(*tuple(np.sum(intersections[-self.n_workloads_per_episode:], axis=0))) + \
                             "index set size {}/{:.0f}".format(len(self.system_environment.index_set), index_set_size))  
            tensorboard.add_scalar('scalars/reward', np.mean(rewards[-self.n_workloads_per_episode:]), episode_idx)
            tensorboard.add_scalar('scalars/runtime', np.mean(workload_times[-self.n_workloads_per_episode:]), episode_idx)
            tensorboard.add_scalar('scalars/size', index_set_size, episode_idx)
            intxn_opps, intxn_opps_taken = np.sum(intersections[-self.n_workloads_per_episode:], axis=0)
            if intxn_opps != 0:
                tensorboard.add_scalar('scalars/intxns', intxn_opps_taken / intxn_opps, episode_idx)

        # record results for all episodes 
        np.savetxt(os.path.join(result_dir, 'train_times.txt'), np.asarray(times), delimiter=',') # TODO replace np.savetxt with np.save
        np.savetxt(os.path.join(result_dir, 'train_agent_times.txt'), np.asarray(agent_times), delimiter=',')
        np.savetxt(os.path.join(result_dir, 'train_system_times.txt'), np.asarray(system_times), delimiter=',')
        np.savetxt(os.path.join(result_dir, 'train_workload_times.txt'), np.asarray(workload_times), delimiter=',')
        np.savetxt(os.path.join(result_dir, 'train_query_times.txt'), np.asarray(query_times), delimiter=',')
        np.savetxt(os.path.join(result_dir, 'train_avg_query_times.txt'), np.asarray(avg_query_times), delimiter=',')
        np.savetxt(os.path.join(result_dir, 'train_rewards.txt'), np.asarray(rewards), delimiter=',')
        np.savetxt(os.path.join(result_dir, 'train_selectivity.txt'), np.asarray(selectivities), delimiter=',')
        np.savetxt(os.path.join(result_dir, 'train_intersections.txt'), np.asarray(intersections), delimiter=',')
        np.savetxt(os.path.join(result_dir, 'losses.txt'), np.asarray(losses), delimiter=',')
        
        with open(os.path.join(result_dir, 'train_index_set_stats.csv'), 'a') as f:
            writer = csv.writer(f)
            for episode in index_set_stats:
                writer.writerow([*tuple(episode[0].values()), episode[1], episode[2]])






    def get_selectivity(self, tbl, cols):
        """Compute selectivity of set of cols, to compare query, agent's suggested index, system's index
        """
        return np.mean([self.system_environment.tbl_2_col_2_sel[tbl][col] for col in cols]) if list(cols) != [] else 0.0





    def act(self, workloads):
        """
        Emulate an episode on queries (i.e. test queries) without training to accumulate system actions

        Args:
            queries (list): SQLQuery objects 

        Returns:
            system actions (dict): records actions taken per query / query_idx
        """

        context = []
        idx_creation_time =[]

        system_actions = {}

        for workload_idx, workload in enumerate(workloads):

            # start = time.monotonic()  

            agent_state = self.converter.system_to_agent_state(workload=workload, 
                                                               system_context=dict(indices=context))
            agent_action = self.agent_api.get_action(agent_state)

            query_cols = [query.query_cols for query in workload]
            workload_cols = self.workload_concat(query_cols)
            workload_cols = list(set(workload_cols)) 
                

            system_action = self.converter.agent_to_system_action(actions = agent_action,
                                                                     meta_data=dict(query_cols=workload_cols))

            query_tbls = [query.query_tbl for query in workload]
            #workload_tbls = self.workload_concat(query_tbls)

            workload_tbls = list(set(query_tbls))

            #Take action

            idx_creation,_ = self.system_environment.act(dict(index=system_action, table=workload_tbls[0] ))

            # end = time.monotonic()
            # TODO, Calculate the index creation time
            system_actions[workload_idx] = system_action
            context.append(system_action)
            idx_creation_time.append(idx_creation)
        
        return idx_creation_time, system_actions


    def restore_workload(self, path=None):
        """
            path (str): path to a workload to rerun an agent on, added ad hoc
        """
        train_workloads = self.data_source.import_data(data_dir=path, label="train")
        test_workloads = self.data_source.import_data(data_dir=path, label="test")

        return train_workloads, test_workloads

    def get_intxn_opp(self, workload, context):
        """Return index in context that query could use        
        
        a B-tree intersection opportunity occurs when any prefix of any permutation of columns is equal to any prefix of an existing index

        e.g. query with query.query_cols == [foo, bar] could use an index [bar, baz] (though probably the planner will only opt for this if 
        bar is sufficiently selective)
        """
        
        query_cols = [query.query_cols for query in workload]
        workload_col = self.workload_concat(query_cols)
        workload_col = list(set(workload_col)) 
        
        workload_cols, n_workload_cols = np.array(workload_col), len(workload_col)

        perm_idxs = itertools.permutations(range(n_workload_cols))
        
        # suppose query.query_cols == [foo, bar, baz]
        # for each permutation of cols e.g. [baz, bar, foo]
        for perm_idx in perm_idxs:
            cols = list(workload_cols[list(perm_idx)]) 

            # for each prefix of cols e.g. [baz, bar]
            for i in range(n_workload_cols):
                cols_prefix = cols[:i+1]

                # check if shared by existing index e.g. [baz, bar, qux]
                for idx in context:
                    if cols_prefix == idx[:i+1]:
                        return idx

        return []

    def group_by_q_len(self, workloads):

        return [workloads[i:i+self.n_queries_per_workload] for i in range(0,len(workloads),self.n_queries_per_workload)]

    def evaluate(self, workloads, baseline = None, export= False):
        """
        Evaluates the execution of a set of workloads (i.e. test workloads).
        Assumes indices of interest are already set up. UPDATED.

        Args:
            workloads (list): SQLworkload objects to execute and evaluate
            baseline (str): baseline identifier (e.g. default or full) 
        """

        workload_runtimes = []
        query_runtimes = []
        avg_query_runtimes = []
        index_set_sizes = []
        idx_creation_times = []
        eval_time = []

        workloads = self.group_by_q_len(workloads)
        

        start = time.monotonic()

        for episode_idx in range(self.n_test_episodes):

            episode_workloads = workloads[episode_idx*self.n_workloads_per_episode:(episode_idx+1) * self.n_workloads_per_episode]

            if baseline is None:
                self.system_environment.reset()
                idx_creation_time, actions = self.act(episode_workloads)
                idx_creation_times.append(idx_creation_time)

            for workload in episode_workloads:


                #query_runtimes_per_workload = []

                runtimes_per_workload = []
                runtimes_per_workload_by_query = []
                avg_runtimes_per_query = []

                for _ in range(self.n_executions):
                    
                    workload_time, avg_query_time,query_time, _ = self.system_environment.execute(workload)                
                    
                    runtimes_per_workload.append(workload_time)
                    avg_runtimes_per_query.append(avg_query_time)
                    runtimes_per_workload_by_query.append(query_time)

                runtimes_per_workload_by_query = list(map(list, zip(*runtimes_per_workload_by_query)))

                workload_runtimes.append(runtimes_per_workload)
                query_runtimes.extend(runtimes_per_workload_by_query)
                avg_query_runtimes.append(avg_runtimes_per_query)




            index_set_size, index_set = self.system_environment.system_status()
            index_set_sizes.append(index_set_size)
        
        eval_time.append(time.monotonic()-start)

        if export:
            tag = baseline if baseline is not None else 'dqn'  # yikes
            result_dir = os.path.join(self.result_dir, tag)
            if not os.path.exists(result_dir): os.makedirs(result_dir)


            # idx creation time

            np.savetxt(os.path.join(result_dir, 'test_idx_times.txt'), np.asarray(idx_creation_times), delimiter=',')


            # runtimes
            np.savetxt(os.path.join(result_dir, 'test_workload_times.txt'), np.asarray(workload_runtimes), delimiter=',')
            np.savetxt(os.path.join(result_dir, 'test_query_times.txt'), np.asarray(query_runtimes), delimiter=',')
            np.savetxt(os.path.join(result_dir, 'test_avg_query_times.txt'), np.asarray(avg_query_runtimes), delimiter=',')
            
            # index set sizes
            np.savetxt(os.path.join(result_dir, 'test_index_set_sizes.txt'), np.asarray(index_set_sizes), delimiter=',')


            # Eval time

            np.savetxt(os.path.join(result_dir, 'eval_time.txt'), np.asarray(eval_time), delimiter=',')

            # TODO - rn saves stats for the final test episode
            
            with open(os.path.join(result_dir, 'test_index_set_stats.csv'), 'wb') as f:
                pickle.dump([index_set_size, index_set], f)
             
            # queries w/ any action taken 
            with open(os.path.join(result_dir, 'test_by_query.csv'), 'a', newline='') as f:
                writer = csv.writer(f)
                for workload_idx, workload in enumerate(episode_workloads):

                    for query_idx, query in enumerate(workload):
                        query_string, query_string_args = query.sample_query()
                        if baseline is None:
                            writer.writerow([query_string % query_string_args, actions.get(workload_idx, ''), np.mean(query_runtimes[self.n_queries_per_workload*workload_idx:self.n_queries_per_workload*(workload_idx+1)][query_idx])])
                        else:
                            writer.writerow([query_string % query_string_args, np.mean(query_runtimes[self.n_queries_per_workload*workload_idx:self.n_queries_per_workload*(workload_idx+1)][query_idx])])

        return workload_runtimes, query_runtimes, avg_query_runtimes


def main(argv):

    import gflags
    import json
    import numpy as np
    import torch

    #
    # logging
    # 
    import logging # import before rlgraph
    format_ = '%(levelname)s / %(module)s / %(message)s\n'
    formatter = logging.Formatter(format_)
    if logging.root.handlers == []:
        handler = logging.StreamHandler(stream=sys.stdout) # no different from basicConfig()
        handler.setFormatter(formatter)
        handler.setLevel(logging.DEBUG)
        logging.root.addHandler(handler)
    else:
    # handler has set up by default in rlgraph
        logging.root.handlers[0].setFormatter(formatter)

    logger = logging.getLogger(__name__)
    logger.info('Starting controller script...')


    #
    # parsing - TODO command-line arg semantics aren't super clean
    #
    FLAGS = gflags.FLAGS
    
    # baselines rely on controller, which requires an agent, so set up dqn by default  
    #gflags.DEFINE_boolean('dqn', True, 'dqn or spg, dqn by default for non-agent')
    gflags.DEFINE_boolean('blackbox', False, 'true to train and test on same workload (e.g. like an Atari agent')
    gflags.DEFINE_string('config', 'conf/dqn.json', 'config for agent, agent representations')
    gflags.DEFINE_string('experiment_config', '', 'config for experiment')
    gflags.DEFINE_boolean('generate_workload', True, 'set True to build train and test workloads')
    gflags.DEFINE_integer('seed', 0, 'seed for workload')
    gflags.DEFINE_string('result_dir', '../res/', 'base directory for workload and workload results')
    gflags.DEFINE_boolean('with_agent', True, 'set True for agent, False for non-agent baseline')
    gflags.DEFINE_boolean('default_baseline', True, 'set when with_agent is False')
    gflags.DEFINE_boolean('get_seed', False, '')
    # TODO should have separate paths for workloads, agents, results
    gflags.DEFINE_string('reevaluate_agent', '', 'path to agent if dqn, agent dir if spg')
    gflags.DEFINE_string('reevaluate_workload', '', 'path to workload to reevaluate agent on')
    
    
    try:
        argv = FLAGS(argv)
    except gflags.FlagsError as e:
        print('%s\\nUsage: %s ARGS\\n%s' % (e, sys.argv[0], FLAGS))

    with open(FLAGS.config, 'r') as fh:
        config = json.load(fh)
    with open(FLAGS.experiment_config, 'r') as fh:
        experiment_config = json.load(fh)
    agent_config, schema_config = config['agent'], config['schema']
    logger.info('agent config: {}'.format(agent_config))
    logger.info('schema config: {}'.format(schema_config))
    logger.info('experiment config: {}'.format(experiment_config))

    result_dir = FLAGS.result_dir
                            
    # n.b. controller is required for agent-advised index or baseline index, b/c wraps system, workload
    controller = PostgresSystemController(
        #dqn=FLAGS.dqn,
        # blackbox=FLAGS.blackbox,
        agent_config=agent_config,
        schema_config=schema_config,
        experiment_config=experiment_config,
        result_dir=result_dir
    )


    # TODO crude way to suppress stdout from system_environment, this should be in a config somewhere 
    controller.system_environment.logger.setLevel(logging.WARNING) 


    seed = FLAGS.seed if FLAGS.seed != 0 else np.random.randint(1000)
    np.random.seed(seed)
    np.savetxt(os.path.join(result_dir, 'seed.txt'), np.asarray([seed]))

    logger.info('numpy seed: {}'.format(seed))
    logger.info('pytorch seed: {}'.format(torch.initial_seed())) # spg 


    # head, tail = os.path.split(__file__)
    # dir = os.path.join(head, '../../conf')
    path = result_dir  + '/data'
    if not os.path.exists(path): os.makedirs(path)

    #reset selectivities
    sel_path = path + '/selectivities.pkl'



    TPCH_Workloads_Processor = workload_processor(agent_config = agent_config,
            experiment_config=experiment_config,
            schema_config= schema_config,
            result_dir= path)

    
    if FLAGS.generate_workload:

        sel_path_exist_flag = os.path.exists(sel_path)
        if sel_path_exist_flag:
            os.remove(sel_path)
        TPCH_Workloads_Processor.reset_workload(path)
    
        train, test = TPCH_Workloads_Processor.gen_workload_stream(export = True)
    else:
        train, test = controller.restore_workload(path=path)
        # TODO! copy workload config?



    if not FLAGS.with_agent:
        controller.reset_system()

        if FLAGS.default_baseline:
            
            controller.evaluate(test, baseline='default', export=True)


        else:
            for tbl in experiment_config['tables']:
                for col in tpch_table_columns[tbl]:
                    controller.system_environment.act(dict(index=[col], table=tbl))
            controller.evaluate(test, baseline='full', export=True)

    elif FLAGS.with_agent:

        reeval = True if FLAGS.reevaluate_agent != '' else False

        if reeval:
            _ , test = controller.restore_workload(path=FLAGS.reevaluate_workload)
            controller.agent_api.load_model(FLAGS.reevaluate_agent)
            controller.reset_system() 
            controller.evaluate(test, export=True) # TODO this replaces old results

        else:
            logger.info('TRAINING')
            controller.train(train)
            controller.system_environment.reset()

            logger.info('TESTING')
            controller.evaluate(test, export=True)
        
            controller.agent_api.save_model(result_dir + ('/dqn'))

    logger.info('RAN TO COMPLETION')         

    if FLAGS.get_seed:

        seed_bound = 25

        controller.reset_system()

        logger.info('scanning seeds -- with no indices')

        avg_wo_idxs = []
        upper_pct_wo_idxs = []
        for seed in range(seed_bound):
            np.random.seed(seed)
            if (seed+1) % 5 == 0: logger.info('completed {}/{} seeds'.format(seed+1,seed_bound))
            _, test = TPCH_Workloads_Processor.gen_workload_stream(export=False) 
            _, per_query_latencies,_ = controller.evaluate(test, export=False)
            per_query_latency_μs = np.mean(per_query_latencies, axis=1) # (n_queries, n_executions_per_query)
            
            avg_wo_idxs.append(np.mean(per_query_latency_μs))
            upper_pct_wo_idxs.append(np.percentile(per_query_latency_μs, 90))

        for tbl in experiment_config['tables']:
            for col in tpch_table_columns[tbl]:
                controller.system_environment.act(dict(index=[col], table=tbl))

        logger.info('scanning seeds -- with indices')


        avg_w_idxs = []
        upper_pct_w_idxs = []
        for seed in range(seed_bound):
            np.random.seed(seed)
            if (seed+1) % 5 == 0: logger.info('completed {}/{} seeds'.format(seed+1,seed_bound))
            _, test = TPCH_Workloads_Processor.gen_workload_stream(export=False) 
            _, per_query_latencies,_ = controller.evaluate(test, export=False)
            per_query_latency_μs = np.mean(per_query_latencies, axis=1)
            
            avg_w_idxs.append(np.mean(per_query_latency_μs))
            upper_pct_w_idxs.append(np.percentile(per_query_latency_μs, 90))

        workload_difficulty_wo_idxs = np.array(upper_pct_wo_idxs) / np.array(avg_wo_idxs)
        workload_difficulty_w_idxs = np.array(upper_pct_w_idxs) / np.array(avg_w_idxs)
        
        avg_speedup = np.array(avg_wo_idxs) / np.array(avg_w_idxs)
        upper_pct_speedup = np.array(upper_pct_wo_idxs) / np.array(upper_pct_w_idxs)

        seeds = np.arange(seed_bound)

        res = np.concatenate((np.expand_dims(seeds, axis=1), 
                              np.expand_dims(workload_difficulty_wo_idxs, axis=1),
                              np.expand_dims(workload_difficulty_w_idxs, axis=1),
                              np.expand_dims(avg_speedup, axis=1),
                              np.expand_dims(upper_pct_speedup, axis=1)), axis=1)
        with open('seeds.txt', 'w') as f:
            np.savetxt(f, res, fmt='%.2f')


if __name__ == "__main__":
    main(sys.argv)


        




