import os, sys
head, tail = os.path.split(__file__)
sys.path.insert(0, os.path.join(head, '../../..')) # lift
from lift.lift.controller.system_controller import SystemController

sys.path.insert(0, os.path.join(head, '..')) # src
from agent.postgres_converter import PostgresConverter as PostgresDQNConverter
from agent.postgres_schema import PostgresSchema as PostgresDQNSchema

from postgres_agent import PostgresAgent
from postgres_data_source import PostgresDataSource
from postgres_system_environment import Action, PostgresSystemEnvironment
from tpch_workload import TPCHWorkload
from common.sql_query import SQLQuery
from common.sql_workload import SQLWorkload

import numpy as np
import pickle
import time
import os


class workload_processor(SystemController):
    """
    Abstraction for training + testing, Some of these are shared by DQN + SPG agents,
    
    Generate the stable stream of workloads
    """
    def __init__(self,agent_config, experiment_config,schema_config,result_dir):
       
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
        #workloads ser/des
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



    def generate_workloads(self):

        n_train_queries = self.n_queries_per_workload * self.n_workloads_per_episode * self.n_train_episodes
        n_test_queries = self.n_queries_per_workload * self.n_workloads_per_episode * self.n_test_episodes
        train_queries = [self.workload.generate_query_template(selectivities=self.system_environment.tbl_2_col_2_sel) 
                             for _ in range(n_train_queries)]
        test_queries = [self.workload.generate_query_template(selectivities=self.system_environment.tbl_2_col_2_sel) 
                            for _ in range(n_test_queries)]
        return train_queries, test_queries

    def gen_workload_stream(self, export=False):

        #total number of workloads needed
        # train_workload_number = self.n_workloads_per_episode * self.n_train_episodes
        # test_workload_number = self.n_workloads_per_episode * self.n_test_episodes

        train_dir = self.result_dir + '/train_workloads'
        test_dir = self.result_dir + '/test_workloads'

        train_queries, test_queries = self.generate_workloads()

        if export:

            self.data_source.export_data(train_queries, train_dir, label= 'train')
            self.data_source.export_data(test_queries, test_dir, label= 'test')


            # for k in range(train_workload_number):
                
                # self.data_source.export_data(train_queries[k*(self.n_queries_per_workload ): (k+1)*self.n_queries_per_workload ], train_dir, k, label= 'train')
            
            # for k in range(test_workload_number):

                # self.data_source.export_data(test_queries[k*(self.n_queries_per_workload ): (k+1)*self.n_queries_per_workload ], test_dir, k, label= 'test')

        return train_queries,test_queries

    
    def reset_workload(self,path):

        isExists=os.path.exists(path)



        train_path = path + '/train_workloads'
        train_path_exist_flag = os.path.exists(train_path)
        if not train_path_exist_flag:
            os.makedirs(train_path)

        test_path = path + '/test_workloads'
        test_path_exist_flag = os.path.exists(test_path)
        if not test_path_exist_flag:
            os.makedirs(test_path)

        if isExists:

            del_list = os.listdir(train_path)
            for f in del_list:
                file_path = os.path.join(train_path, f)
                if os.path.isfile(file_path):
                    os.remove(file_path)  

            del_list = os.listdir(test_path)
            for f in del_list:
                file_path = os.path.join(test_path, f)
                if os.path.isfile(file_path):
                    os.remove(file_path)  

            
        else:
            self.logger.info("Path does not exist!")

    def workload_embedding(self, queries):

        pass

    def workload_concat(self,workload):

        w_0 = workload.pop()

        for _ in range(len(workload)):
            q_tmp = w_0
            w_0 = workload.pop()
            w_0 = w_0 + q_tmp
        
        return w_0




    def read_workload_stream(self, train_queries, test_queries):

        train_workload_number = self.n_workloads_per_episode * self.n_train_episodes
        test_workload_number = self.n_workloads_per_episode * self.n_test_episodes
        train_workloads = []
        test_workloads = []

        for k in range(train_workload_number):      

            workload = train_queries[k*(self.n_queries_per_workload ): (k+1)*self.n_queries_per_workload ]
            #tokened_w = [row.as_tokens().copy() for row in workload]
            #w = self.workload_concat(tokened_w)
            train_workloads.append(workload)            

        for k in range(test_workload_number):      

            workload = test_queries[k*(self.n_queries_per_workload ): (k+1)*self.n_queries_per_workload ]
            #tokened_w = [row.as_tokens().copy() for row in workload]
            #w = self.workload_concat(tokened_w)
            test_workloads.append(workload)     
        
        return train_workloads, test_workloads

    def read_workload_stream_from_csv(self,dir,label):
    
        train_workload_number = self.n_workloads_per_episode * self.n_train_episodes
        test_workload_number = self.n_workloads_per_episode * self.n_test_episodes
        train_workloads = []
        test_workloads = []

        if label == "train":
            for k in range(train_workload_number):
                queries = self.data_source.import_data(dir, k, label="train",path=None)
                workload = queries[0:self.n_queries_per_workload]
                #tokened_w = [row.as_tokens().copy() for row in workload]
                #w = self.workload_concat(tokened_w)
                train_workloads.append(workload)

        if label == "test":
            for k in range(test_workload_number):
                queries = self.data_source.import_data(dir, k, label="test",path=None)
                workload = queries[0:self.n_queries_per_workload]
                #tokened_w = [row.as_tokens().copy() for row in workload]
                #w = self.workload_concat(tokened_w)
                test_workloads.append(workload)
        
        return train_workloads, test_workloads

                


                


if __name__ == "__main__":

    import json
    import numpy as np

    #hardcoded here
    #dir = "/Users/wangtaiyi/Documents/Graduate/Cambridge/Research/RL/Learning_Index_Selection/index_Code/Workloadlevel_index_tuning/conf"
    head, tail = os.path.split(__file__)
    dir = os.path.join(head, '../../conf')
    path = dir + '/../data'

    #reset selectivities
    sel_path = path + '/selectivities.pkl'
    sel_path_exist_flag = os.path.exists(sel_path)
    if sel_path_exist_flag:
        os.remove(sel_path)


    with open(dir+'/dqn.json', 'r') as fh:
        config = json.load(fh)

    with open(dir +'/experiment.json', 'r') as fh:
        experiment_config = json.load(fh)


    agent_config, schema_config = config['agent'], config['schema']



    TPCH_Workloads_Processor = workload_processor(agent_config = agent_config,
            experiment_config=experiment_config,
            schema_config= schema_config,
            result_dir= dir + '/../data')

    TPCH_Workloads_Processor.reset_workload(path)
    train, test = TPCH_Workloads_Processor.gen_workload_stream(export = True)
    #TPCH_Workloads_Processor.reset_workload(path)