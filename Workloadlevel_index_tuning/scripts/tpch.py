#!/usr/bin/python3

"""

    utility script for different tasks with tpch database 
    assumes you've run setup db script and added a super-user user

    Todos
        - hardcoded for jw2027
        - create queries from query templates

    References
        - http://www.tpc.org/tpc_documents_current_versions/current_specifications.asp
        - http://myfpgablog.blogspot.com/2016/08/tpc-h-queries-on-postgresql.html
"""

import argparse
import numpy as np
import os
import re
import subprocess
import sys
import time

import psycopg2 as pg
from tpch_util import TPCH_DIR, TPCH_TOOL_DIR, DATA_DIR, DSN, TPCH_DSN

class TPCHClient():
    """Encapsulates 2 cxns, one to a non-tpch default db and another to the tpch db. 

        Comments:
        - python psycopg driver has transaction-specific semantics that are handled cleanly in context managers. 
          I relied on cxn.set_session(autocommit=True), which ignores these implicit semantics, because of how I handled cxns.
        - The exception handling here is extremely brittle 

    """

    def __init__(self):
        self.cxn = self.__connect(DSN)
        try: 
            self.tpch_cxn = self.__connect(TPCH_DSN)
        except pg.OperationalError as e:
            self.tpch_cxn = None 
    
    def __connect(self, DSN):
        cxn = pg.connect(DSN)
        cxn.set_session(autocommit=True)
        return cxn

    def repopulate(self, scale_factor):
        """dbgen to generate data, build-tpch-tables.sql to create tables and copy data into tables

        Args:
            scale_factor (float) : -s, --scale_factor arg to dbgen, only certain values are TPCH compliant, but scale of 1 corresponds to 1 GB.

        """
        
        # drop db if exists, this isn't terribly expensive
        print('dropping db...')
        if self.tpch_cxn:
            self.tpch_cxn.close()
            with self.cxn.cursor() as cur:
                cur.execute('DROP DATABASE tpch')
            
        # create db, connection to db
        print('creating db... connecting to db...')
        with self.cxn.cursor() as cur:
            cur.execute('CREATE DATABASE tpch')
        self.tpch_cxn = self.__connect(TPCH_DSN)

        # generate data
        print('running dbgen... repopulating db...')
        tic=time.time()
        subprocess.run(['./dbgen.sh', str(scale_factor)]) # TODO whoops put absolute path

        # create tables, copy data into tables
        with open(os.path.join(TPCH_DIR, 'build-tpch-tables.sql'), 'r') as f, \
            self.tpch_cxn.cursor() as cur:

            cur.execute(f.read())
        
        toc = time.time()
        print('...took {} seconds'.format(round(toc-tic)))

        # rm generated data
        print('cleaning up...')
        subprocess.run(['rm', '-rf', DATA_DIR])

        print('saving scale_factor to tpch_sf.txt')
        np.savetxt('../../tpch_sf.txt', np.asarray([scale_factor]))
    
    def close(self):
        self.cxn.close()
        self.tpch_cxn.close()

    


##
# script utils
#
def parser():
    parser = argparse.ArgumentParser(description='utility script for different tasks with tpch database')
    parser.add_argument('-r', '--repopulate', action='store_true', help='')
    parser.add_argument('-s', '--scale_factor', required='-r' in sys.argv or '--repopulate' in sys.argv, action='store', type=float)
    return parser 

def main():

    args = parser().parse_args()

    tpch = TPCHClient()    

    if args.repopulate:
        tpch.repopulate(scale_factor=args.scale_factor)

if __name__ == "__main__":
    main()
