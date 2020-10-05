import sys
import os
sys.path.append("../../GENERATOR/")
from model.simple_access.functions import Generator
from model.utilities.sanity_check import Sanity_Check 

def create_path(folder_path): #correct path or exit
    if not os.path.exists(folder_path) :
        try:
            os.mkdir(folder_path)
        except OSError:
            print ("Creation of the directory %s failed" % folder_path)
            exit(1)
    return folder_path


#linear 
results_folder_name_lr = create_path(os.path.join(os.getcwd(),"output","lr_sanity_check"))
results = Generator.lr_separated_params(None,5,0,30,3,1,2,1,None,'float64',True) #n_points must be higher than 20 for this check
Sanity_Check().plot_all(results.features, results_folder_name_lr)

#logistic
results_folder_name_log = create_path(os.path.join(os.getcwd(),"output","log_sanity_check"))
results = Generator.log_separated_params(None,5,0,30,3,1,2,1,None,'float64',True) #n_points must be higher than 20 for this check
Sanity_Check().plot_all(results.features, results_folder_name_log)


#linear by chunk
results_folder_name_lr_by_chunk = create_path(os.path.join(os.getcwd(),"output","lr_sanity_check_by_chunk"))
#using the by_chunk generator is impossible to create an istance of results, 
#so the features we want to test are automatically taken by .csv files previously generated using the chunk generator 
input_file = create_path(os.path.join(os.getcwd(),"input","lr_sanity_check_by_chunk", "dataset.csv"))
Sanity_Check().plot_all(input_file, results_folder_name_lr_by_chunk)

#logistic by chunk
results_folder_name_log_by_chunk = create_path(os.path.join(os.getcwd(),"output","log_sanity_check_by_chunk"))
#using the by_chunk generator is impossible to create an istance of results, 
#so the features we want to test are automatically taken by .csv files previously generated using the chunk generator 
input_file = create_path(os.path.join(os.getcwd(),"input","log_sanity_check_by_chunk", "dataset.csv"))
Sanity_Check().plot_all(input_file, results_folder_name_log_by_chunk)