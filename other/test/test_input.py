import sys
import numpy as np
import pandas as pd
from os import path 
import os
import json
from pandas.testing import assert_series_equal

sys.path.append("../../GENERATOR/")
from model.simple_access.functions import Generator
from model.simple_access.functions import IOfunctions

##########################################test 1:
#ALL INPUT ELEMENTS PRESENT
#this elements are useful for different kinds of generations, the single generator will never use all of those

#copy of all_elements.json :

#{
#"type_of_regression": "log",
#"chunk": "False",
#"opt_level": "float64",
#"verbose": "True",
#"out_name": "test_input_out",
#"chunks": 1,
#"compression": "True",
#"compr_type": "bz",
#"csv_files": "True",
#"seed": 12, 
#"n_variables": 5,
#"error_variance": 3,
#"n_points": 30,
#"related_vars": 3,
#"n_scales": 1,
#"requested_mean": 2,
#"requested_odds": 2,
#"unuseful_vars": 1,
#"betas": [ 1.23, 1.1, 0.0, -0.8, 0.7, -1.3 ]
#}

#loaded input from all_elements.json
elements = IOfunctions.get_input("all_elements.json")

#expected result
exp_type_of_regression = "log"
exp_chunk = False
exp_opt_level = "float64"
exp_verbose = True
exp_output_filepath = os.path.join(os.getcwd(),"output","test_input_out")
exp_compression = True
exp_compr_type = "bz"
exp_csv_files= True 
exp_seed = 12
exp_n_variables=5
exp_error_variance=3
exp_n_points=30
exp_related_vars=3
exp_n_scales=1
exp_requested_odds=2
exp_betas=pd.Series([1.23, 1.1, 0.0, -0.8, 0.7, -1.3 ])
exp_betas.to_numpy(dtype='float64')
print(betas)
exp_unuseful_vars=1 #this input will be ignorated because betas values are present

#checks
if type_of_regression == exp_type_of_regression and chunk == exp_chunk and opt_level == exp_opt_level and verbose == exp_verbose and output_filepath == exp_output_filepath \
    and compression == exp_compression and compr_type == exp_compr_type and csv_files == exp_csv_files and seed == exp_seed and n_variables == exp_n_variables and error_variance == exp_error_variance \
    and n_points == exp_n_points and related_vars == exp_related_vars and n_scales == exp_n_scales and requested_odds == exp_requested_odds \
    and unuseful_vars == exp_unuseful_vars and betas.equals(exp_betas):
    print("test1 (all elements): ok")
else:
    print("test1 (all elements): failed")

##########################################test 2:
#INPUT ELEMENTS MISSING
#the correct behavior is that all choices not specified in .json file that are useful to generation are substituted with default values, 
#and 'seed' and 'betas' parameters, if not specified in .json file, must be randomized

#copy of elements_missing.json :

#{
#"n_variables": 5,
#"error_variance": 3,
#"n_points": 30,
#"related_vars": 3,
#"n_scales": 1,
#"requested_mean": 2,
#"unuseful_vars": 1
#}

#loaded input from elements_missing.json
elements = IOfunctions.get_input("elements_missing.json")

type_of_regression = elements.type_of_regression
chunk = elements.chunk
opt_level = elements.opt_level
verbose = elements.verbose
output_filepath = elements.output_filepath
chunks = elements.chunks
compression=elements.compression
compr_type=elements.compr_type
csv_files=elements.csv_files
seed=elements.params.seed
n_variables=elements.params.n_variables
error_variance=elements.params.error_variance
n_points=elements.params.n_points
related_vars=elements.params.related_vars
n_scales=elements.params.n_scales
requested_mean=elements.params.requested_mean
unuseful_vars=elements.params.unuseful_vars
betas=elements.params.betas

#expected result
exp_type_of_regression = "lr" #default value
exp_chunk = False #default value
exp_opt_level = "float64" #default value
exp_verbose = True #default value
exp_output_filepath = "./output/temp" #default value
exp_chunks = 1 # default value
exp_compression = False #default value
exp_compr_type = "bz" #default value
exp_csv_files= False #default value
#seed has been randomized, different value every times
exp_n_variables=5
exp_error_variance=3
exp_n_points=30
exp_related_vars=3
exp_n_scales=1
exp_requested_mean=2
exp_unuseful_vars=1
#betas been randomized, different value every times, just check it is not empty

#checks
if type_of_regression == exp_type_of_regression and chunk == exp_chunk and opt_level == exp_opt_level and verbose == exp_verbose and output_filepath == exp_output_filepath and chunks == exp_chunks \
    and compression == exp_compression and compr_type == exp_compr_type and csv_files == exp_csv_files and n_variables == exp_n_variables and error_variance == exp_error_variance \
    and n_points == exp_n_points and related_vars == exp_related_vars and n_scales == exp_n_scales and requested_mean == exp_requested_mean \
    and unuseful_vars == exp_unuseful_vars and not betas.empty:
    print("test2 (missing elements): ok")
else:
    print("test2 (missing elements): failed")