import sys
import numpy as np
sys.path.append("../../GENERATOR/")
from model.utilities.correlation_matrix_generator import Correlation_Matrix_Generator

# 2 <= n_related_vars <= n_variables_1 <= n_variables_2 <= 10
seed = 11
n_related_vars = np.random.randint(2, 10)

#questa funzione servirebbe a testare qualsiasi tipo 
def get_matrix_sums_general(matrix, n_variables, index):
    ogg = []
    sum1 = 0
    sum2 = 0
    for i in range(n_variables):
       sum1 = sum1 + matrix[i][index]
    ogg.append(sum1)
    
    for j in range(n_variables):
       sum2 = sum2 + matrix[index][j]
    ogg.append(sum2)
    
    return ogg
    
def get_line_sum(matrix, n_variables, index):
    sum = 0
    for i in range(n_variables):
       sum = sum + matrix[i][index]
    return sum
    
def get_all_lines_sum(matrix, n_variables):
    lines = []
    for index in range(n_variables):
        lines.append(get_line_sum(matrix, n_variables, index))
    return lines
    
def check_symmetrical_matrix(matrix, n_variables):
    sum1 = 0
    sum2 = 0
    for index in range(n_variables):
        for i in range(n_variables):
           sum1 = sum1 + matrix[i][index]
        
        for j in range(n_variables):
           sum2 = sum2 + matrix[index][j]
           
        allowed_error = 0.0001
        if abs(sum1 - sum2) > allowed_error:
            return False
    return True
    

#matrix 1    
n_variables_1 = np.random.randint(n_related_vars, 10)
matrix1 = Correlation_Matrix_Generator.gen_corr_matrix(seed,n_variables_1,n_related_vars)
if check_symmetrical_matrix(matrix1, n_variables_1):
    print("matrix 1 is symmetrical: ")
    print(matrix1)
else:
    exit(1)
    
print("\n\n\n")

#matrix 2
n_variables_2 = np.random.randint(n_variables_1, 10)
matrix2 = Correlation_Matrix_Generator.gen_corr_matrix(seed,n_variables_2,n_related_vars)
if check_symmetrical_matrix(matrix2, n_variables_2):
    print("matrix 2 is symmetrical: ")
    print(matrix2)
else:
    exit(2)

print("\n\n\n")

lines1 = get_all_lines_sum(matrix1, n_variables_1)
lines2 = get_all_lines_sum(matrix2, n_variables_2)
   
counter = 0
for line1 in lines1 :
    if line1 != 1 and line1 in lines2 :
        counter += 1
            
if counter == n_related_vars :
    print("test: ok")
else:
    print("test: failed")