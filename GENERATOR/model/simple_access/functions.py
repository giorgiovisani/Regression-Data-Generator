import os
import json
#lr
from model.implementation_lr.lr_params import LR_Params
from model.implementation_lr.lr_generator import LR_Generator, LR_Chunk_Generator
#log
from model.implementation_log.log_params import Log_Params
from model.implementation_log.log_generator import Log_Generator, Log_Chunk_Generator

from model.utilities.csv_saver import Csv_Results_Saver

class Elements:
    """
        A class for input elements, if a choice is None, it will be replaced with default value
    """

    def __init__(self, params, type_of_regression = "lr", chunk = False, opt_level = 'float64', verbose = True, output_filepath = "./output/temp", chunks = 1, compression = False, compr_type =  'bz', csv_files = False):
        self.params = params
        if type_of_regression: self.type_of_regression = type_of_regression
        else: self.type_of_regression = "lr"
        if chunk: self.chunk = chunk
        else: self.chunk = False
        if opt_level: self.opt_level = opt_level
        else: self.opt_level = 'float64'
        if verbose: self.verbose = verbose
        else: self.verbose = True
        if output_filepath: self.output_filepath = output_filepath
        else: self.output_filepath = "./output/temp"
        if chunks: self.chunks = chunks
        else: self.chunks = 1
        if compression: self.compression = compression
        else: self.compression = False
        if compr_type: self.compr_type = compr_type
        else: self.compr_type = 'bz'
        if csv_files: self.csv_files = csv_files
        else: self.csv_files = False


class Generator:
    """
        Static class that hides the complexity of the Generation Process and gives access to all the features of the library 
    """
    
    # with separated params (you can call these functions without instanciate any object of the library)
    
    @staticmethod
    def lr_separated_params(seed=None,n_variables=None,error_variance=None,n_points=None,related_vars=None,n_scales=None,requested_mean=None,unuseful_vars=None,betas=None, opt_level='float64',verbose=True):
        """
        :param seed: int, the seed for initialize random generation. If not set, will be randomized
        :param n_variables: int, number of variables
        :param error_variance: int, error variance of error vector
        :param n_points: int, it represents the number of points that will be generated
        :param related_vars: int, number of related variables
        :param n_scales: int, number of scales
        :param requested_mean: int, mean of vector Y
        :param unuseful_vars: int, number of 0 betas
        :param betas: array. If not set, it will be randomized considering unuseful_vars
        :param opt_level: str that represent the float type for casting arrays. Can be 'float64', 'float32' or 'float16'. Default 'float64'\n
        'float64' -> default (no optimization)\n
        'float32' -> downcast to float32\n
        'float16' -> downcast to float16\n
        :param verbose: verbose output
        
        :return: instance of class DGP_Results
        """
        params = LR_Params(seed,n_variables, error_variance, n_points, related_vars, n_scales, requested_mean, unuseful_vars, betas)

        generator = LR_Generator()
        results = generator.generate_dataset(params, opt_level, verbose)
        return results       
    
    @staticmethod
    def log_separated_params(seed=None,n_variables=None,error_variance=None,n_points=None,related_vars=None,n_scales=None,requested_odds=None,unuseful_vars=None,betas=None, opt_level='float64',verbose=True): 
        """
        :param seed: int, the seed for initialize random generation. If not set, will be randomized
        :param n_variables: int, number of variables
        :param error_variance: int, error variance of error vector
        :param n_points: int, it represents the number of points that will be generated
        :param related_vars: int, number of related variables
        :param n_scales: int, number of scales
        :param requested_odds: int, odd of vector Y
        :param unuseful_vars: int, number of 0 betas
        :param betas: array. If not set, it will be randomized considering unuseful_vars
        :param opt_level: str that represent the float type for casting arrays. Can be 'float64', 'float32' or 'float16'. Default 'float64'\n
        'float64' -> default (no optimization)\n
        'float32' -> downcast to float32\n
        'float16' -> downcast to float16\n
        :param verbose: verbose output
        
        :return: instance of class DGP_Results
        """
        params = Log_Params(seed,n_variables, error_variance, n_points, related_vars, n_scales, requested_odds, unuseful_vars, betas)
        generator = Log_Generator()
        results = generator.generate_dataset(params, opt_level, verbose)
        return results   

    
    @staticmethod
    def lr_chunk_separated_params(seed=None,n_variables=None,error_variance=None,n_points=None,related_vars=None,n_scales=None,requested_mean=None,unuseful_vars=None,betas=None,output_filepath="./output/temp",opt_level='float64',verbose=True,chunks=1,compression=False,compr_type=None): 
        """
        :param seed: int, the seed for initialize random generation. If not set, will be randomized
        :param n_variables: int, number of variables
        :param error_variance: int, error variance of error vector
        :param n_points: int, it represents the number of points that will be generated
        :param related_vars: int, number of related variables
        :param n_scales: int, number of scales
        :param requested_mean: int, mean of vector Y
        :param unuseful_vars: int, number of 0 betas
        :param betas: array. If not set, it will be randomized considering unuseful_vars
        :param output_filepath: 
        :param opt_level: str that represent the float type for casting arrays. Can be 'float64', 'float32' or 'float16'. Default 'float64'\n
        'float64' -> default (no optimization)\n
        'float32' -> downcast to float32\n
        'float16' -> downcast to float16\n
        :param verbose: verbose output
        :param chunks: int, number of chunks that the dataset will have. It represents the number of total iterations
        :param compression: bool. Set true if you want compression (must not have compr_type = None)
        :param compr_type: str that represent the compression type. Works only if compression is set to True. Can be 3 values:\n\t
        'gz' -> using gunzip compression, file will be saved as <filename>.csv.gz\n\t
        'bz' -> using bzip compression, file will be saved as <filename>.csv.bz\n\t
        'xz' -> using xz compression, file will be saved as <filename>.csv.xz\n\t
        
        You may not assign this method to a variable, because it doesn't mantain anything in memory
        """
        params = LR_Params(seed,n_variables, error_variance, n_points, related_vars, n_scales, requested_mean, unuseful_vars, betas)
        saver = Csv_Results_Saver(output_filepath)
        
        generator = LR_Chunk_Generator()
        generator.generate_dataset(params, saver, chunks, verbose, opt_level, compression, compr_type)
        #don't return results, just saves those on files
      
    @staticmethod      
    def log_chunk_separated_params(seed=None,n_variables=None,error_variance=None,n_points=None,related_vars=None,n_scales=None,requested_odds=None,unuseful_vars=None,betas=None,output_filepath="./output/temp",opt_level='float64',verbose=True,chunks=1,compression=False,compr_type=None):
        params = Log_Params(seed,n_variables, error_variance, n_points, related_vars, n_scales, requested_odds, unuseful_vars, betas)
        """
        :param seed: int, the seed for initialize random generation. If not set, will be randomized
        :param n_variables: int, number of variables
        :param error_variance: int, error variance of error vector
        :param n_points: int, it represents the number of points that will be generated
        :param related_vars: int, number of related variables
        :param n_scales: int, number of scales
        :param requested_mean: int, odds of vector Y
        :param unuseful_vars: int, number of 0 betas
        :param betas: array. If not set, it will be randomized considering unuseful_vars
        :param output_filepath: 
        :param opt_level: str that represent the float type for casting arrays. Can be 'float64', 'float32' or 'float16'. Default 'float64'\n
        'float64' -> default (no optimization)\n
        'float32' -> downcast to float32\n
        'float16' -> downcast to float16\n
        :param verbose: verbose output
        :param chunks: int, number of chunks that the dataset will have. It represents the number of total iterations
        :param compression: bool. Set true if you want compression (must not have compr_type = None)
        :param compr_type: str that represent the compression type. Works only if compression is set to True. Can be 3 values:\n\t
        'gz' -> using gunzip compression, file will be saved as <filename>.csv.gz\n\t
        'bz' -> using bzip compression, file will be saved as <filename>.csv.bz\n\t
        'xz' -> using xz compression, file will be saved as <filename>.csv.xz\n\t
        
        You may not assign this method to a variable, because it doesn't mantain anything in memory
        """
        saver = Csv_Results_Saver(output_filepath)
        
        generator = Log_Chunk_Generator()
        generator.generate_dataset(params, saver, chunks, verbose, opt_level, compression, compr_type)
        #don't return results, just saves those on files
    
    # with just one object for all the params (you must create the params instance in the code that is calling these functions)
    
    @staticmethod
    def lr(params, opt_level='float64',verbose=True):
        """
        :param params: instance of LR_Params
        :param opt_level: str that represent the float type for casting arrays. Can be 'float64', 'float32' or 'float16'. Default 'float64'\n
        'float64' -> default (no optimization)\n
        'float32' -> downcast to float32\n
        'float16' -> downcast to float16\n
        :param verbose: verbose output
        
        :return: instance of class DGP_Results
        """
        generator = LR_Generator()
        results = generator.generate_dataset(params, opt_level, verbose)
        return results       
    
    @staticmethod
    def log(params, opt_level='float64',verbose=True): 
        """
        :param params: instance of Log_Params
        :param opt_level: str that represent the float type for casting arrays. Can be 'float64', 'float32' or 'float16'. Default 'float64'\n
        'float64' -> default (no optimization)\n
        'float32' -> downcast to float32\n
        'float16' -> downcast to float16\n
        :param verbose: verbose output
        
        :return: instance of class DGP_Results
        """
        generator = Log_Generator()
        results = generator.generate_dataset(params, opt_level, verbose)
        return results   

    @staticmethod
    def lr_chunk(params,output_filepath="./output/temp",opt_level='float64',verbose=True,chunks=1,compression=False,compr_type=None):
        """
        :param params: instance of LR_Params
        :param output_filepath: 
        :param opt_level: str that represent the float type for casting arrays. Can be 'float64', 'float32' or 'float16'. Default 'float64'\n
        'float64' -> default (no optimization)\n
        'float32' -> downcast to float32\n
        'float16' -> downcast to float16\n
        :param verbose: verbose output
        :param chunks: int, number of chunks that the dataset will have. It represents the number of total iterations
        :param compression: bool. Set true if you want compression (must not have compr_type = None)
        :param compr_type: str that represent the compression type. Works only if compression is set to True. Can be 3 values:\n\t
        'gz' -> using gunzip compression, file will be saved as <filename>.csv.gz\n\t
        'bz' -> using bzip compression, file will be saved as <filename>.csv.bz\n\t
        'xz' -> using xz compression, file will be saved as <filename>.csv.xz\n\t
        
        You may not assign this method to a variable, because it doesn't mantain anything in memory
        """
        saver = Csv_Results_Saver(output_filepath)
        
        generator = LR_Chunk_Generator()
        generator.generate_dataset(params, saver, chunks, verbose, opt_level, compression, compr_type)
        #don't return results, just saves those on files
        
    @staticmethod
    def log_chunk(params,output_filepath="./output/temp",opt_level='float64',verbose=True,chunks=1,compression=False,compr_type=None):
        """
        :param params: instance of Log_Params
        :param output_filepath: 
        :param opt_level: str that represent the float type for casting arrays. Can be 'float64', 'float32' or 'float16'. Default 'float64'\n
        'float64' -> default (no optimization)\n
        'float32' -> downcast to float32\n
        'float16' -> downcast to float16\n
        :param verbose: verbose output
        :param chunks: int, number of chunks that the dataset will have. It represents the number of total iterations
        :param compression: bool. Set true if you want compression (must not have compr_type = None)
        :param compr_type: str that represent the compression type. Works only if compression is set to True. Can be 3 values:\n\t
        'gz' -> using gunzip compression, file will be saved as <filename>.csv.gz\n\t
        'bz' -> using bzip compression, file will be saved as <filename>.csv.bz\n\t
        'xz' -> using xz compression, file will be saved as <filename>.csv.xz\n\t
        
        You may not assign this method to a variable, because it doesn't mantain anything in memory
        """
        saver = Csv_Results_Saver(output_filepath)
        
        generator = Log_Chunk_Generator()
        generator.generate_dataset(params, saver, chunks, verbose, opt_level, compression, compr_type)
        #don't return results, just saves those on files
    
    @staticmethod
    def by_input(elements):
        """
        :param elements: instance of Elements
        
        :return: instance of class DGP_Results, or nothing. It depends on what kind of generation is specified in the dictionary
        
        It is really useful just combined with the input method of the class IOfunctions
        """
        
        if elements.chunk == True :
            if elements.type_of_regression == "lr" : 
                Generator.lr_chunk(elements.params,elements.output_filepath,elements.opt_level,elements.verbose,elements.chunks,elements.compression,elements.compr_type)
            if elements.type_of_regression == "log" : 
                Generator.log_chunk(elements.params,elements.output_filepath,elements.opt_level,elements.verbose,elements.chunks,elements.compression,elements.compr_type)
            return None #don't return results, just saves those on files
        else:
            if elements.type_of_regression == "lr" :
                results = Generator.lr(elements.params, elements.opt_level, elements.verbose)
            if elements.type_of_regression == "log" : 
                results = Generator.log(elements.params, elements.opt_level, elements.verbose)
            return results

class IOfunctions:
    """
        Static class that help the user to put the correct elements into the generator and to get a correct presentation of results
    """
    
    @staticmethod
    def get_input(input_file=None): 
        """
        :param input_file: str, path of a .json input file or just the name of a .json file in "./input" folder
        
        :return instance of Elements, containing all the input elements
        
        It provides to get and validate input elements (loading these by file or interacting with the user by stdIO)
        """
        
        #CHECK on file input 
        def positive_check(requested_input, number):
            try:
                var = int(number)
                if var > 0 :
                    return True
                else:
                    print(requested_input + " must be positive")
                    return False
            except ValueError:
                print(requested_input + " must be a number")
                return False
                    
        def non_negative_check(requested_input, number):
            try:
                var = int(number)
                if var >= 0 :
                    return True
                else:
                    print(requested_input + " must be zero or positive")
                    return False
            except ValueError:
                print(requested_input + " must be a number")
                return False
                
        def integer_check(requested_input, number):
            try:
                var = int(number)
                return True
            except ValueError:
                print(requested_input + " must be a number")
                return False
                
        def string_check(requested_input, string, valid_strings):
            if string in valid_strings :
                return True
            else:
                print(requested_input + " must be one of those strings:")
                print(valid_strings)
                return False
                
        def bool_check(requested_input, string):
            valid_strings = ['True', 'False']
            if string in valid_strings :
                return True
            else:
                print(requested_input + " must be one of those strings:")
                print(valid_strings)
                return False
            
        def betas_check(betas, n_variables):
            if len(betas) != (n_variables + 1) :
                print("the length of betas array must be like 'n_variables' incremented of one")
                return False
            else :
                return True
                    
        #INPUT by stdio
        def positive_input(requested_input):
            ok = 0
            message = "- " + requested_input + "? "
            while(ok != 1):
                try:
                    var = int(input(message))
                    if var > 0 :
                        ok = 1
                    else:
                        print("It must be positive")
                except ValueError:
                    print("That was no valid number")
                
            return var
            
        def non_negative_input(requested_input):
            ok = 0
            message = "- " + requested_input + "? "
            while(ok != 1):
                try:
                    var = int(input(message))
                    if var >= 0 :
                        ok = 1
                    else:
                        print("It must be zero or positive")
                except ValueError:
                    print("That was no valid number")
                
            return var
            
        def integer_input(requested_input):
            ok = 0
            message = "- " + requested_input + "? "
            while(ok != 1):
                try:
                    var = int(input(message))
                    ok = 1
                except ValueError:
                    print("That was no valid number")
                
            return var
            
        def string_input(requested_input, valid_strings):
            ok = 0
            message = "- " + requested_input + "? "
            while(ok != 1):
                var = input(message)
                if var in valid_strings :
                    ok = 1
                else:
                    print("It is not a valid string")   
            return var
            
        def bool_input(requested_input):
            valid_strings = ["y", "n"]
            ok = 0
            message = "- " + requested_input + "(y or n)? "
            while(ok != 1):
                var = input(message)
                if var in valid_strings :
                    ok = 1
                else:
                    print("It is not a valid string")
            if var == "y":
                return True
            else: 
                return False
                
        def float_input(requested_input):
            ok = 0
            message = "- " + requested_input + "? "
            while(ok != 1):
                try:
                    var = float(input(message))
                    ok = 1
                except ValueError:
                    print("That was no valid number")
                
            return var
            
        def betas_input(n_variables):
            betas = []
            betas.append(float_input("intercept (beta0)"))
            for i in range(n_variables) :
                index = str(i + 1)
                betas.append(float_input("beta"+index))
            return betas
        
        #choices (all kinds of generation)
        type_of_regression=None;chunk=None;opt_level=None;chunks=None;compression=None;compr_type=None;
        csv_files=None;output_filepath=None;
        
        #params (all kinds of generation)
        seed=None;n_variables=None;error_variance=None;n_points=None;related_vars=None; 
        n_scales=None;requested_mean=None;requested_odds=None;unuseful_vars=None;betas=None;
        
        params=None
        
        if input_file:
            #CHECK INPUT FILE
            #if input_file is not an existing path or the name of a file in './input' folder, exit with 3 
            if os.path.exists(input_file) : 
                input_filepath = input_file
            else : 
                input_filepath = os.path.join(os.getcwd(),"input",input_file)
                if not os.path.exists(input_filepath) :
                    print("specified .json file doesn't exists ")
                    print(input_filepath)
                    exit(3)
            with open(input_filepath, 'r') as input_f:
                p_dict = json.load(input_f)
                
                
            type_of_regression = p_dict.get("type_of_regression")
            if type_of_regression and not string_check("type_of_regression", type_of_regression, ["lr","log"]) : exit(1)
                
            chunk = p_dict.get("chunk")
            if chunk and not bool_check("chunk", chunk) : exit(1)
            elif chunk : chunk = eval(chunk)
            
            opt_level = p_dict.get("opt_level")
            if opt_level and not string_check("opt_level", opt_level, ["float16", "float32", "float64"]) : exit(1)
            
            verbose = p_dict.get("verbose")
            if verbose and not bool_check("verbose", verbose) : exit(1)
            elif verbose : verbose = eval(verbose)
            
            out_name = p_dict.get("out_name")
            
            if chunk == True :
            
                chunks = p_dict.get("chunks")
                if chunks and not positive_check("chunks", chunks) : exit(1)
                
                compression = p_dict.get("compression")
                if compression and not bool_check("compression", compression) : exit(1)
                elif compression : compression = eval(compression)
                
                if compression == True:
                
                    compr_type = p_dict.get("compr_type")
                    if compr_type and not string_check("compr_type", compr_type, ['bz', 'gz', 'xz']) : exit(1)
                
            else:
            
                csv_files = p_dict.get("csv_files")
                if csv_files and not bool_check("csv_files", csv_files) : exit(1)
                elif csv_files : csv_files = eval(csv_files)
                
                if csv_files == True :
                
                    compression = p_dict.get("compression")
                    if compression and not bool_check("compression", compression) : exit(1)
                    elif compression : compression = eval(compression)
                    
                    if compression == True:
                    
                        compr_type = p_dict.get("compr_type")
                        if compr_type and not string_check("compr_type", compr_type, ['bz', 'gz', 'xz']) : exit(1)
                    
            if type_of_regression == None or type_of_regression == "lr" : #default is linear
                seed = p_dict.get("seed")
                if seed and not positive_check("seed", seed) : exit(1)
                n_variables = p_dict.get("n_variables")
                if n_variables and not positive_check("n_variables", n_variables) : exit(1)
                error_variance = p_dict.get("error_variance")
                if error_variance and not integer_check("error_variance", error_variance) : exit(1)
                n_points = p_dict.get("n_points")
                if n_points and not positive_check("n_points", n_points) : exit(1)
                related_vars = p_dict.get("related_vars")
                if related_vars and not non_negative_check("related_vars", related_vars) : exit(1)
                n_scales = p_dict.get("n_scales")
                if n_scales and not positive_check("n_scales", n_scales) : exit(1)
                requested_mean = p_dict.get("requested_mean")
                if requested_mean and not non_negative_check("requested_mean", requested_mean) : exit(1)
                unuseful_vars = p_dict.get("unuseful_vars")
                if unuseful_vars and not positive_check("unuseful_vars", unuseful_vars) : exit(1)
                betas = p_dict.get("betas")
                if betas and not betas_check(betas, n_variables) : exit(1)
                
                params = LR_Params(seed,n_variables, error_variance, n_points, related_vars, n_scales, requested_mean, unuseful_vars, betas)
                
            if type_of_regression == "log" :
                seed = p_dict.get("seed")
                if seed and not positive_check("seed", seed) : exit(1)
                n_variables = p_dict.get("n_variables")
                if n_variables and not positive_check("n_variables", n_variables) : exit(1)
                error_variance = p_dict.get("error_variance")
                if error_variance and not integer_check("error_variance", error_variance) : exit(1)
                n_points = p_dict.get("n_points")
                if n_points and not positive_check("n_points", n_points) : exit(1)
                related_vars = p_dict.get("related_vars")
                if related_vars and not non_negative_check("related_vars", related_vars) : exit(1)
                n_scales = p_dict.get("n_scales")
                if n_scales and not positive_check("n_scales", n_scales) : exit(1)
                requested_odds = p_dict.get("requested_odds")
                if requested_odds and not non_negative_check("requested_odds", requested_odds) : exit(1)
                unuseful_vars = p_dict.get("unuseful_vars")
                if unuseful_vars and not positive_check("unuseful_vars", unuseful_vars) : exit(1)
                betas = p_dict.get("betas")
                if betas and not betas_check(betas, n_variables) : exit(1)
                
                params = Log_Params(seed,n_variables, error_variance, n_points, related_vars, n_scales, requested_odds, unuseful_vars, betas)
        else :
            
            print("Insert some information about the kind of generation you want to make:")
            
            type_of_regression = string_input("Type of regression ('lr' or 'log')", ["lr", "log"])
            chunk = bool_input("Do you want to generate datas by chunk")
            opt_level = string_input("Optimizzation level ('float16', 'float32' or 'float64')", ["float16", "float32", "float64"])
            verbose = bool_input("Do you want informations about memory usage and elapsed time")
            
            out_name = input("- Output filepath or name of the new folder in './output'? ")
                        
            if chunk == True :
                chunks = positive_input("Number of chunks")
                compression = bool_input("Do you want any kind of compression")
                if compression == True :
                    compr_type = string_input("What kind of compression ('bz', 'gz' or 'xz')", ['bz', 'gz', 'xz'])
            else:
                csv_files = bool_input("Do you want the generation of .csv files")
                if csv_files == True :
                    compression = bool_input("Do you want any kind of compression")
                    if compression == True :
                        compr_type = string_input("What kind of compression ('bz', 'gz' or 'xz')", ['bz', 'gz', 'xz'])
            
            print("Insert the mathematical parameters:")
            
            if type_of_regression == None or type_of_regression == "lr" :  #default is linear
                n_variables = positive_input("Number of variables")
                error_variance = integer_input("Error variance")
                n_points = positive_input("Number of points")
                related_vars = non_negative_input("Number of related variables")
                n_scales = positive_input("Number of scales")
                requested_mean = non_negative_input("requested mean")
                
                if bool_input("Do you want to specify betas (if not those will be randomized)") :
                    betas = betas_input(n_variables)
                else: 
                    unuseful_vars = non_negative_input("number of unuseful variables (0.0 betas)")
                    
                if bool_input("Do you want to specify a seed (if not it will be randomized)") :
                    seed = non_negative_input("seed")
                
                params = LR_Params(seed,n_variables, error_variance, n_points, related_vars, n_scales, requested_mean, unuseful_vars, betas)
                         
            if type_of_regression == "log" :
                n_variables = positive_input("Number of variables")
                error_variance = integer_input("Error variance")
                n_points = positive_input("Number of points")
                related_vars = non_negative_input("Number of related variables")
                n_scales = positive_input("Number of scales")
                requested_odds = non_negative_input("requested odds")
                
                if bool_input("Do you want to specify betas (if not those will be randomized)") :
                    betas = betas_input(n_variables)
                else: 
                    unuseful_vars = non_negative_input("number of unuseful variables (0.0 betas)")
                    
                if bool_input("Do you want to specify a seed (if not it will be randomized)") :
                    seed = non_negative_input("seed")
                
                params = Log_Params(seed,n_variables, error_variance, n_points, related_vars, n_scales, requested_odds, unuseful_vars, betas)
  
        if out_name :
            #CREATE OUTPUT FOLDER
            #if out_name is not an existing path, create folder with this name in output
            if os.path.exists(out_name) : 
                output_filepath = out_name
            else : 
                if "/" in out_name or  "\\" in out_name :
                    print ("Impossible to create a folder with '/' or '\' in 'out_name' ")
                    exit(4)
                
                output_filepath = os.path.join(os.getcwd(),"output",out_name)
                if not os.path.exists(output_filepath) :
                    try:
                        os.mkdir(output_filepath)
                    except OSError:
                        print ("Creation of the directory %s failed" % output_filepath)
                        exit(5)
        
        return Elements(params, type_of_regression, chunk, opt_level, verbose, output_filepath, chunks, compression, compr_type, csv_files)

    @staticmethod
    def manage_output(elements, results):
        """
        :param elements: instance of Elements
        :param results: instance of DGP_Results
        
        It provides to print on stdIO and, if needed, to write on files the results  
        """
        
        #print on stdio
        print("features:\n{}\n".format(results.features))
        print("response:\n{}\n".format(results.response))
        print("correlation matrix:\n{}\n".format(results.corr_matrix))
        print("covariance matrix:\n{}\n".format(results.cov_matrix))
        print("user: {}\n".format(results.gen_details.user))
        print("os name: {}".format(results.gen_details.platform_name))
            
        #create file in folder
        output_f = open(elements.output_filepath + "/results.txt","w+")
        
        #write in txt file
        output_f.write("features:\n{}\n\n".format(results.features))
        output_f.write("response:\n{}\n\n".format(results.response))
        output_f.write("correlation matrix:\n{}\n\n".format(results.corr_matrix))
        output_f.write("covariance matrix:\n{}\n\n".format(results.cov_matrix))
        output_f.write("user: {}\n\n".format(results.gen_details.user))
        output_f.write("os name: {}".format(results.gen_details.platform_name))
        
        #eventually write in csv files
        if(elements.csv_files == True):
            saver = Csv_Results_Saver(elements.output_filepath)            
            saver.save_all(elements.params, results, elements.compression, elements.compr_type,elements.verbose) 
        
        