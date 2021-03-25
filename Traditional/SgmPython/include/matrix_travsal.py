import numpy as np

# 对角线遍历一个矩阵
#           1*****2
#           *    *     *
#           3*****4

def travesal_matrix_diag14(mat):
    height,width = mat.shape[:2]
    curr_col = 0 # 第几列， y 坐标
    curr_row = 0 # 第几行， x坐标
    for i in range(width):
        curr_col = width -1 -i
        curr_row =0
        while((curr_row<height) and (curr_col<width)):
            print(mat[curr_row][curr_col])
            curr_col = curr_col  + 1
            curr_row = curr_row +1
    
    for j in range(1,height):
        curr_col = 0
        curr_row = j
        while((curr_row<height) and (curr_col<width)):
            print(mat[curr_row][curr_col])
            curr_col = curr_col +1
            curr_row = curr_row +1
    
def travesal_matrix_diag41(mat):
    height, width = mat.shape[:2]
    curr_col = 0
    curr_row = 0
    for i in range(width):
        curr_row = height -1
        curr_col = i
        while((curr_col>=0) and(curr_row>=0)):
            print(mat[curr_row][curr_col])
            curr_col = curr_col -1
            curr_row = curr_row -1
    for  j  in range(1,height):
        curr_row = height -1 - j
        curr_col = width -1
        while((curr_col>=0) and(curr_row>=0)):
            print(mat[curr_row][curr_col])
            curr_col = curr_col -1
            curr_row = curr_row -1

def travesal_matrix_diag23(mat):
    height,width = mat.shape[:2]
    curr_col = 0 # 第几列， y 坐标
    curr_row = 0 # 第几行， x坐标
    for i in range(width):
        curr_col = i
        curr_row = 0
        while((curr_col>=0) and (curr_row<height)):
            print(mat[curr_row][curr_col])
            curr_col = curr_col -1
            curr_row = curr_row + 1
    for k in range(1,height):
        curr_col = width -1
        curr_row =k 
        while((curr_col>=0) and (curr_row<height)):
            print(mat[curr_row][curr_col])
            curr_col = curr_col -1
            curr_row = curr_row + 1
     
def travesal_matrix_diag32(mat):
    height,width = mat.shape[:2]
    cur_row = 0
    cur_col   = 0
    for i in range(height):
        cur_col = 0
        cur_row = i
        while((cur_col<width) and (cur_row>=0)):
            print(mat[cur_row][cur_col])
            cur_col = cur_col +1
            cur_row = cur_row -1

    for j in range(1,width):
        cur_col  = j
        cur_row = height -1
        while((cur_col<width) and (cur_row>=0)):
            print(mat[cur_row][cur_col])
            cur_col = cur_col +1
            cur_row = cur_row -1

if __name__=="__main__":
    a = np.array([[1,2,3,4],
                    [5,6,7,8],
                    [9,10,11,12]])
    a2 = np.array([[1,2,3,4],
                    [5,6,7,8],
                    [9,10,11,12]])
    b = a.transpose()
    travesal_matrix_diag14(a)
    print("-----------------------------------------------")
    travesal_matrix_diag41(a)
    print("-----------------------------------------------")
    travesal_matrix_diag23(a)
    print("-----------------------------------------------")
    travesal_matrix_diag32(a)
    print("-----------------------------------------------")
    print(a+a2)

    