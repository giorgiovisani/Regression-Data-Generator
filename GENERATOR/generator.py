import sys
from model.simple_access.functions import Generator
from model.simple_access.functions import IOfunctions

if len(sys.argv) > 2 or len(sys.argv) < 1 :
    print("Insert the name of the .json input file or not insert arguments")

#INPUT GENERATOR
if len(sys.argv) == 2 :
    elements = IOfunctions.get_input(sys.argv[1])
else :
    elements = IOfunctions.get_input()

#ELABORATION GENERATOR
results = Generator.by_input(elements)

#if you chose a "by chunk" generation:
#generator don't return results, just saves those on files stored in the folder you specified 

#OUTPUT GENERATOR
if(results) : # not "by chunk" generations
    IOfunctions.manage_output(elements, results)