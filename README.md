There are three different ways to access the software.
1) The launch of "generator.py" without passing arguments, being automatically guided in entering the input elements from the terminal.
   It is the easiest access method, recommended if it is the FIRST TIME you use this software;
2) The launch of "generator.py" passing as argument the name of the .json file from which to read the input elements.
   It requires a well-formed file in the "input" folder;
3) The invocation of the functions of "Generator" class, previously imported, without going through "generator.py".
   Examples of this kind of access are present in "notebooks" folder.
   
You can choose different generation options:
- "type_of_regression" ("lr", "log"), Linear or Logistic Regression
- "chunk" (boolean), requires the software to write on file the dataset in portions
- "opt_level" ("float16", "float32" and "float64"), optimization level
- "verbose" (boolean), requires statistics on generation time and memory
- "out_name", output filepath or name of the folder in "./output" to save the output files in
- "chunks" (positive integer), number of chunks
- "compression" (boolean), choice of compression for .csv files
- "compr_type" (".bz",".gz" and ".xz")
- "csv_files" (boolean), production of .csv files in generation without chunk

You can set different methematical parameters:
- "seed", seed of randomization
- "n_variables", number of X
- "error_variance"
- "n_points"
- "related_vars", number of related X 
- "n_scales"
- "requested_mean"
- "requested_odds"
- "unuseful_vars", number of 0.0 betas, if "betas" array is missing
- "betas", array of "n_variables" + 1 elements.

And finally you can obtain different results:
- X
- Y
- correlation matrix
- covariance matrix
- generation details
