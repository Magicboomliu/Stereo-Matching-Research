import numpy as np
import cv2
import math

def census_transform_5x5(img_data):
    '''
    img:data: numpy array
    Return the census of the Image
    '''
    if(img_data is None):
        return
    img_height = img_data.shape[0]
    img_width = img_data.shape[1]
    census_data = np.zeros((img_height,img_width))
    
    # Traversal the Pixel
    for i in range(2,img_height-2):
        for j in range(2,img_width-2):
            gray_center = img_data[i][j] # Get the Center GrayScale
            
            census_val =0
            for m in range(-2,3):
                for n in range(-2,3):
                    census_val = census_val<<1 #左边移动一位
                    gray_val = img_data[i+m][j+n]
                    if gray_center > gray_val:
                        census_val = census_val+1

            census_data[i][j] = census_val

    return census_data
     
def census_transform_9x7(img_data):
    '''
    img:data: numpy array
    Return the census of the Image
    '''
    if(img_data is None):
        return
    img_height = img_data.shape[0]
    img_width = img_data.shape[1]
    census_data = np.zeros((img_height,img_width))
    
    # Traversal the Pixel
    for i in range(4,img_height-4):
        for j in range(3,img_width-3):
            gray_center = img_data[i][j] # Get the Center GrayScale
            
            census_val =0
            for m in range(-4,5):
                for n in range(-3,4):
                    census_val = census_val<<1 #左边移动一位
                    gray_val = img_data[i+m][j+n]
                    if gray_center > gray_val:
                        census_val = census_val+1

            census_data[i][j] = census_val

    return census_data
     

def Hamming(census_val1, census_val2):
    hamming_dist  = 0
    val = census_val1 ^ census_val2
    while(val):
        hamming_dist = hamming_dist +1
        val = val & (val -1)
    return hamming_dist

# 对原始的图片在左右（Aggregation By Row）两个方向上进行聚合
def CostAggregationLeftRight(img_data, min_disparity,max_disparity,p1,p2_init,cost_init, is_forward):
    # Return Cost Aggregation OutCome from left and Right
    img_data = img_data.astype(np.uint8)
    height,width = img_data.shape[:2]
    if(img_data is None):
        print("Receive a NULL Data Input for CostAggregation func")
        return
    # 视差的范围
    disp_range = max_disparity - min_disparity
    cost_aggregation = np.zeros((height,width,disp_range)) # 初始化一个聚合矩阵， 全部为0
    # 现在开始一行一行聚合
    for i in range(height):
        # 从左边往右边进行聚合操作
        if(is_forward):
            # 第一个元素的cost = aggregation cost, 因为第一个元素最左边没有其他元素了
            cost_aggregation[i][0] = cost_init[i][0]
            last_aggregation = cost_aggregation[i][0]
            gray_cur = img_data[i][0]
            gray_last = img_data[i][0]
            min_cost_d_last_path = findMinIn1darray(last_aggregation) # 找到上一个路径的最小值
 
            # 每一行逐个元素进行更新, 从左边往右边聚合
            for j in range(width):
                if j==0:
                    continue
                gray_cur = img_data[i][j]
                p2_init_new = int(p2_init /(abs(int(gray_cur)-int(gray_last))+1))
               #对disparity进行遍历
                for  d_idx in range(disp_range):  # 更新cost, 更新gray_last,更新last_aggregation,更新 min_cost_d_last_path
                    cost = cost_init[i][j][d_idx]
                    if(d_idx+2<disp_range):
                        cost_l1 = last_aggregation[d_idx+1]
                        cost_l2 = last_aggregation[d_idx] + p1
                        cost_l3 = last_aggregation[d_idx + 2] + p1
                        cost_l4 = min_cost_d_last_path + max(p1,p2_init_new)
                        cost_new = cost + min(min(cost_l1,cost_l2),min(cost_l3,cost_l4)) -min_cost_d_last_path
                        cost_aggregation[i][j][d_idx] = cost_new   # 更新cost
                    else:
                        cost_l1 = 150
                        cost_l2 = last_aggregation[d_idx] + p1
                        cost_l3 = 150
                        cost_l4 = min_cost_d_last_path + max(p1,p2_init_new)
                        cost_new = cost + min(min(cost_l1,cost_l2),min(cost_l3,cost_l4)) -min_cost_d_last_path
                        cost_aggregation[i][j][d_idx] = cost_new   # 更新cost

                last_aggregation = cost_aggregation[i][j]         #更新last_aggregation
                min_cost_d_last_path = findMinIn1darray(last_aggregation) # 更新 min_cost_d_last_path
                gray_last = gray_cur                                                    #更新gray_last
        #从右边往左边进行聚合操作
        else:
            # 第一个元素是最右边的元素
            cost_aggregation[i][width-1] = cost_init[i][width-1]
            last_aggregation = cost_aggregation[i][width-1]
            min_cost_d_last_path = findMinIn1darray(last_aggregation)
            gray_last = img_data[i][width-1]
            gray_cur = img_data[i][width-1]
             
             # 开始从右边往左边进行遍历更新
            for m in range(width):
                if m ==0:  # 跳过最右边的元素
                    continue
                gray_cur = img_data[i][width-1-m] # 记录当前的灰度值
                p2_init_new = int(p2_init /(abs(int(gray_cur)-int(gray_last))+1))
                for  r_d_idx in range(disp_range):
                    cost = cost_init[i][width-1-m][r_d_idx] # 初始化的COST
                    if(r_d_idx+2<disp_range):
                        cost_l1 = last_aggregation[r_d_idx +1]
                        cost_l2 = last_aggregation[r_d_idx] + p1
                        cost_l3 = last_aggregation[r_d_idx+2] + p1
                        cost_l4 = min_cost_d_last_path + max(p1,p2_init_new)
                        cost_new = cost + min(min(cost_l1,cost_l2),min(cost_l3,cost_l4)) -min_cost_d_last_path
                        cost_aggregation[i][width-1-m][r_d_idx] = cost_new   # 更新cost
                    else:
                        cost_l1 = 150
                        cost_l2 = last_aggregation[r_d_idx] + p1
                        cost_l3 = 150
                        cost_l4 = min_cost_d_last_path + max(p1,p2_init_new)
                        cost_new = cost + min(min(cost_l1,cost_l2),min(cost_l3,cost_l4)) -min_cost_d_last_path
                        cost_aggregation[i][width-1-m][r_d_idx] = cost_new   # 更新cost
                        

                last_aggregation = cost_aggregation[i][width-1-m]         #更新last_aggregation
                min_cost_d_last_path = findMinIn1darray(last_aggregation) # 更新 min_cost_d_last_path
                gray_last = gray_cur                                                    #更新gray_last      
    return cost_aggregation

# 对原始的图片在上下（Aggregation By Column）两个方向上进行聚合
def CostAggregationUpDown(img_data,min_disparity,max_disparity,p1,p2_init,cost_init,is_forward):
    # Return Cost Aggregation OutCome from left and Right
    img_data = img_data.astype(np.uint8)
    height,width = img_data.shape[:2]
    if(img_data is None):
        print("Receive a NULL Data Input for CostAggregation func")
        return
    # 视差的范围
    disp_range = max_disparity - min_disparity
    cost_aggregation = np.zeros((height,width,disp_range)) # 初始化一个聚合矩阵， 全部为0
    for i in range(width): # 按照列进行遍历
        if(is_forward):     #从上到下聚合
            # 第一个元素的cost = aggregation cost, 因为第一个元素最左边没有其他元素了
            cost_aggregation[0][i] = cost_init[0][i] # 每一列的第一个元素
            last_aggregation = cost_aggregation[0][i] # 每一列第一个元素的COST array 
            gray_cur = img_data[0][i]  # 获得当前的灰度
            gray_last = img_data[0][i]
            min_cost_d_last_path = findMinIn1darray(last_aggregation) # 找到上一个路径的最小值
            for j in range(1,height): # 按照一行一行遍历每个元素
                # 获得当前的灰度
                gray_cur = img_data[j][i] # 获得当前的灰度
                p2_init_new = int(p2_init /(abs(int(gray_cur)-int(gray_last))+1))
                for d_idx in range(disp_range):
                    cost = cost_init[j][i][d_idx]
                    if(d_idx+2<disp_range):
                        cost_l1 = last_aggregation[d_idx+1]
                        cost_l2 = last_aggregation[d_idx] + p1
                        cost_l3 = last_aggregation[d_idx + 2] + p1
                        cost_l4 = min_cost_d_last_path + max(p1,p2_init_new)
                        cost_new = cost + min(min(cost_l1,cost_l2),min(cost_l3,cost_l4)) -min_cost_d_last_path
                        cost_aggregation[j][i][d_idx] = cost_new   # 更新cost
                    else:
                        cost_l1 = 150
                        cost_l2 = last_aggregation[d_idx] + p1
                        cost_l3 = 150
                        cost_l4 = min_cost_d_last_path + max(p1,p2_init_new)
                        cost_new = cost + min(min(cost_l1,cost_l2),min(cost_l3,cost_l4)) -min_cost_d_last_path
                        cost_aggregation[j][i][d_idx] = cost_new   # 更新cost
                
                gray_last = gray_cur
                last_aggregation = cost_aggregation[j][i]
                min_cost_d_last_path = findMinIn1darray(last_aggregation)
        else:
            cost_aggregation[height-1][i] = cost_init[height-1][i]
            last_aggregation = cost_aggregation[height-1][i] # 每一列最后一个元素的COST array
            gray_cur = img_data[height-1][i]  # 获得当前的灰度
            gray_last = img_data[height-1][i] # 获得当前路径上一个像素的灰度
            min_cost_d_last_path = findMinIn1darray(last_aggregation) # 找到上一个路径的最小值
            for k in range(1,height):
                gray_cur = img_data[height-1-k][i]
                p2_init_new = int(p2_init /(abs(int(gray_cur)-int(gray_last))+1))
                for dd_idx in range(disp_range):
                    cost = cost_init[height-1-k][i][dd_idx]
                    if(dd_idx+2<disp_range):
                        cost_l1 = last_aggregation[dd_idx+1]
                        cost_l2 = last_aggregation[dd_idx] + p1
                        cost_l3 = last_aggregation[dd_idx + 2] + p1
                        cost_l4 = min_cost_d_last_path + max(p1,p2_init_new)
                        cost_new = cost + min(min(cost_l1,cost_l2),min(cost_l3,cost_l4)) -min_cost_d_last_path
                        cost_aggregation[height-1-k][i][dd_idx] = cost_new   # 更新cost
                    else:
                        cost_l1 = 150
                        cost_l2 = last_aggregation[dd_idx] + p1
                        cost_l3 = 150
                        cost_l4 = min_cost_d_last_path + max(p1,p2_init_new)
                        cost_new = cost + min(min(cost_l1,cost_l2),min(cost_l3,cost_l4)) -min_cost_d_last_path
                        cost_aggregation[height-1-k][i][dd_idx] = cost_new   # 更新cost
                
                gray_last = gray_cur
                last_aggregation = cost_aggregation[height-1-k][i]
                min_cost_d_last_path = findMinIn1darray(last_aggregation)
                    
    return cost_aggregation

# 对角线遍历一个矩阵 ， 按照对角线的方向进行聚合
#           1*****2
#           *    *     *
#           3*****4

# 从1-4两个方向对原始视差图进行聚合
def CostDiagonal14(img_data,min_disparity,max_disparity,p1,p2_init,cost_init):
        # Return Cost Aggregation OutCome from left and Right
    img_data = img_data.astype(np.uint8)
    height,width = img_data.shape[:2]
    if(img_data is None):
        print("Receive a NULL Data Input for CostAggregation func")
        return
    # 视差的范围
    disp_range = max_disparity - min_disparity
    cost_aggregation = np.zeros((height,width,disp_range)) # 初始化一个聚合矩阵， 全部为0
    curr_col = 0 # 第几列， y 坐标
    curr_row = 0 # 第几行， x坐标
    for i in range(width):
        curr_col = width -1 -i
        curr_row =0
        cost_aggregation[curr_row][curr_col] = cost_init[curr_row][curr_col] # 保持原始的Cost
        last_aggregation = cost_aggregation[curr_row][curr_col] # 保留上一个的 cost，记录为 last aggregation
        gray_cur = img_data[curr_row][curr_col]
        gray_last = img_data[curr_row][curr_col]  # 记录当前灰度和路径中上一个元素的灰度
        min_cost_d_last_path = findMinIn1darray(last_aggregation) # 找到上一个路径的最小值
        
        while((curr_row<height) and (curr_col<width)):
            curr_col = curr_col  + 1
            curr_row = curr_row +1
            if((curr_row==height) or (curr_col ==width)):
                break
            gray_cur = img_data[curr_row][curr_col] # 获得当前的灰度
            p2_init_new = int(p2_init /(abs(int(gray_cur)-int(gray_last))+1))
            for d_idx in range(disp_range):
                cost= cost_init[curr_row][curr_col][d_idx]  # 原始的COST
                if(d_idx+2<disp_range):
                    cost_l1 = last_aggregation[d_idx+1]
                    cost_l2 = last_aggregation[d_idx] + p1
                    cost_l3 = last_aggregation[d_idx + 2] + p1
                    cost_l4 = min_cost_d_last_path + max(p1,p2_init_new)
                    cost_new = cost + min(min(cost_l1,cost_l2),min(cost_l3,cost_l4)) -min_cost_d_last_path
                    cost_aggregation[curr_row][curr_col][d_idx] = cost_new   # 更新cost
                else:
                    cost_l1 = 150
                    cost_l2 = last_aggregation[d_idx] + p1
                    cost_l3 = 150
                    cost_l4 = min_cost_d_last_path + max(p1,p2_init_new)
                    cost_new = cost + min(min(cost_l1,cost_l2),min(cost_l3,cost_l4)) -min_cost_d_last_path
                    cost_aggregation[curr_row][curr_col][d_idx] = cost_new   # 更新cost
            
            last_aggregation = cost_aggregation[curr_row][curr_col]
            gray_last = gray_cur
            min_cost_d_last_path = findMinIn1darray(last_aggregation)
            
    for j in range(1,height):
        curr_col = 0
        curr_row = j
        
        cost_aggregation[curr_row][curr_col] = cost_init[curr_row][curr_col] # 保持原始的Cost
        last_aggregation = cost_aggregation[curr_row][curr_col] # 保留上一个的 cost，记录为 last aggregation
        gray_cur = img_data[curr_row][curr_col]
        gray_last = img_data[curr_row][curr_col]  # 记录当前灰度和路径中上一个元素的灰度
        min_cost_d_last_path = findMinIn1darray(last_aggregation) # 找到上一个路径的最小值
    
        while((curr_row<height) and (curr_col<width)):
            curr_col = curr_col +1
            curr_row = curr_row +1
            if((curr_row==height) or (curr_col ==width)):
                break
            gray_cur = img_data[curr_row][curr_col] # 获得当前的灰度
            p2_init_new = int(p2_init /(abs(int(gray_cur)-int(gray_last))+1))
            for dd_idx in range(disp_range):
                cost = cost_init[curr_row][curr_col][dd_idx]
                if(dd_idx+2<disp_range):
                    cost_l1 = last_aggregation[dd_idx+1]
                    cost_l2 = last_aggregation[dd_idx] + p1
                    cost_l3 = last_aggregation[dd_idx + 2] + p1
                    cost_l4 = min_cost_d_last_path + max(p1,p2_init_new)
                    cost_new = cost + min(min(cost_l1,cost_l2),min(cost_l3,cost_l4)) -min_cost_d_last_path
                    cost_aggregation[curr_row][curr_col][dd_idx] = cost_new   # 更新cost
                else:
                    cost_l1 = 150
                    cost_l2 = last_aggregation[dd_idx] + p1
                    cost_l3 = 150
                    cost_l4 = min_cost_d_last_path + max(p1,p2_init_new)
                    cost_new = cost + min(min(cost_l1,cost_l2),min(cost_l3,cost_l4)) -min_cost_d_last_path
                    cost_aggregation[curr_row][curr_col][dd_idx] = cost_new   # 更新cost
            
            last_aggregation = cost_aggregation[curr_row][curr_col]
            gray_cur = gray_last
            min_cost_d_last_path = findMinIn1darray(last_aggregation)

    return cost_aggregation
            
# 从1-4两个方向对原始视差图进行聚合
def CostDiagonal41(img_data,min_disparity,max_disparity,p1,p2_init,cost_init):
        # Return Cost Aggregation OutCome from left and Right
    img_data = img_data.astype(np.uint8)
    height,width = img_data.shape[:2]
    if(img_data is None):
        print("Receive a NULL Data Input for CostAggregation func")
        return
    # 视差的范围
    disp_range = max_disparity - min_disparity
    cost_aggregation = np.zeros((height,width,disp_range)) # 初始化一个聚合矩阵， 全部为0
    curr_col = 0 # 第几列， y 坐标
    curr_row = 0 # 第几行， x坐标
    for i in range(width):
        curr_row = height -1
        curr_col = i
        cost_aggregation[curr_row][curr_col] = cost_init[curr_row][curr_col] # 保持原始的Cost
        last_aggregation = cost_aggregation[curr_row][curr_col] # 保留上一个的 cost，记录为 last aggregation
        gray_cur = img_data[curr_row][curr_col]
        gray_last = img_data[curr_row][curr_col]  # 记录当前灰度和路径中上一个元素的灰度
        min_cost_d_last_path = findMinIn1darray(last_aggregation) # 找到上一个路径的最小值
        
        while((curr_col>=0) and(curr_row>=0)):
            curr_col = curr_col -1
            curr_row = curr_row -1
            if((curr_row==0) or (curr_col ==0)):
                break
            gray_cur = img_data[curr_row][curr_col] # 获得当前的灰度
            p2_init_new = int(p2_init /(abs(int(gray_cur)-int(gray_last))+1))
            for d_idx in range(disp_range):
                cost= cost_init[curr_row][curr_col][d_idx]  # 原始的COST
                if(d_idx+2<disp_range):
                    cost_l1 = last_aggregation[d_idx+1]
                    cost_l2 = last_aggregation[d_idx] + p1
                    cost_l3 = last_aggregation[d_idx + 2] + p1
                    cost_l4 = min_cost_d_last_path + max(p1,p2_init_new)
                    cost_new = cost + min(min(cost_l1,cost_l2),min(cost_l3,cost_l4)) -min_cost_d_last_path
                    cost_aggregation[curr_row][curr_col][d_idx] = cost_new   # 更新cost
                else:
                    cost_l1 = 150
                    cost_l2 = last_aggregation[d_idx] + p1
                    cost_l3 = 150
                    cost_l4 = min_cost_d_last_path + max(p1,p2_init_new)
                    cost_new = cost + min(min(cost_l1,cost_l2),min(cost_l3,cost_l4)) -min_cost_d_last_path
                    cost_aggregation[curr_row][curr_col][d_idx] = cost_new   # 更新cost
            
            last_aggregation = cost_aggregation[curr_row][curr_col]
            gray_last = gray_cur
            min_cost_d_last_path = findMinIn1darray(last_aggregation)
            
    for j in range(1,height):
        curr_row = height -1 - j
        curr_col = width -1
        
        cost_aggregation[curr_row][curr_col] = cost_init[curr_row][curr_col] # 保持原始的Cost
        last_aggregation = cost_aggregation[curr_row][curr_col] # 保留上一个的 cost，记录为 last aggregation
        gray_cur = img_data[curr_row][curr_col]
        gray_last = img_data[curr_row][curr_col]  # 记录当前灰度和路径中上一个元素的灰度
        min_cost_d_last_path = findMinIn1darray(last_aggregation) # 找到上一个路径的最小值
    
        while((curr_col>=0) and(curr_row>=0)):
            curr_col = curr_col -1
            curr_row = curr_row -1
            if((curr_row==0) or (curr_col ==0)):
                break
            gray_cur = img_data[curr_row][curr_col] # 获得当前的灰度
            p2_init_new = int(p2_init /(abs(int(gray_cur)-int(gray_last))+1))
            for dd_idx in range(disp_range):
                cost = cost_init[curr_row][curr_col][dd_idx]
                if(dd_idx+2<disp_range):
                    cost_l1 = last_aggregation[dd_idx+1]
                    cost_l2 = last_aggregation[dd_idx] + p1
                    cost_l3 = last_aggregation[dd_idx + 2] + p1
                    cost_l4 = min_cost_d_last_path + max(p1,p2_init_new)
                    cost_new = cost + min(min(cost_l1,cost_l2),min(cost_l3,cost_l4)) -min_cost_d_last_path
                    cost_aggregation[curr_row][curr_col][dd_idx] = cost_new   # 更新cost
                else:
                    cost_l1 = 150
                    cost_l2 = last_aggregation[dd_idx] + p1
                    cost_l3 = 150
                    cost_l4 = min_cost_d_last_path + max(p1,p2_init_new)
                    cost_new = cost + min(min(cost_l1,cost_l2),min(cost_l3,cost_l4)) -min_cost_d_last_path
                    cost_aggregation[curr_row][curr_col][dd_idx] = cost_new   # 更新cost
            
            last_aggregation = cost_aggregation[curr_row][curr_col]
            gray_cur = gray_last
            min_cost_d_last_path = findMinIn1darray(last_aggregation)

    return cost_aggregation
            
# 从2-3两个方向对原始视差图进行聚合
def CostDiagonal23(img_data,min_disparity,max_disparity,p1,p2_init,cost_init):
        # Return Cost Aggregation OutCome from left and Right
    img_data = img_data.astype(np.uint8)
    height,width = img_data.shape[:2]
    if(img_data is None):
        print("Receive a NULL Data Input for CostAggregation func")
        return
    # 视差的范围
    disp_range = max_disparity - min_disparity
    cost_aggregation = np.zeros((height,width,disp_range)) # 初始化一个聚合矩阵， 全部为0
    curr_col = 0 # 第几列， y 坐标
    curr_row = 0 # 第几行， x坐标
    for i in range(width):
        curr_col = i
        curr_row = 0
        cost_aggregation[curr_row][curr_col] = cost_init[curr_row][curr_col] # 保持原始的Cost
        last_aggregation = cost_aggregation[curr_row][curr_col] # 保留上一个的 cost，记录为 last aggregation
        gray_cur = img_data[curr_row][curr_col]
        gray_last = img_data[curr_row][curr_col]  # 记录当前灰度和路径中上一个元素的灰度
        min_cost_d_last_path = findMinIn1darray(last_aggregation) # 找到上一个路径的最小值
        
        while((curr_col>=0) and (curr_row<height)):
            curr_col = curr_col -1
            curr_row = curr_row + 1
            if((curr_row==height) or (curr_col <0)):
                break
            gray_cur = img_data[curr_row][curr_col] # 获得当前的灰度
            p2_init_new = int(p2_init /(abs(int(gray_cur)-int(gray_last))+1))
            for d_idx in range(disp_range):
                cost= cost_init[curr_row][curr_col][d_idx]  # 原始的COST
                if(d_idx+2<disp_range):
                    cost_l1 = last_aggregation[d_idx+1]
                    cost_l2 = last_aggregation[d_idx] + p1
                    cost_l3 = last_aggregation[d_idx + 2] + p1
                    cost_l4 = min_cost_d_last_path + max(p1,p2_init_new)
                    cost_new = cost + min(min(cost_l1,cost_l2),min(cost_l3,cost_l4)) -min_cost_d_last_path
                    cost_aggregation[curr_row][curr_col][d_idx] = cost_new   # 更新cost
                else:
                    cost_l1 = 150
                    cost_l2 = last_aggregation[d_idx] + p1
                    cost_l3 = 150
                    cost_l4 = min_cost_d_last_path + max(p1,p2_init_new)
                    cost_new = cost + min(min(cost_l1,cost_l2),min(cost_l3,cost_l4)) -min_cost_d_last_path
                    cost_aggregation[curr_row][curr_col][d_idx] = cost_new   # 更新cost
            
            last_aggregation = cost_aggregation[curr_row][curr_col]
            gray_last = gray_cur
            min_cost_d_last_path = findMinIn1darray(last_aggregation)
            
    for j in range(1,height):
        curr_col = width -1
        curr_row =j 
        
        cost_aggregation[curr_row][curr_col] = cost_init[curr_row][curr_col] # 保持原始的Cost
        last_aggregation = cost_aggregation[curr_row][curr_col] # 保留上一个的 cost，记录为 last aggregation
        gray_cur = img_data[curr_row][curr_col]
        gray_last = img_data[curr_row][curr_col]  # 记录当前灰度和路径中上一个元素的灰度
        min_cost_d_last_path = findMinIn1darray(last_aggregation) # 找到上一个路径的最小值
    
        while((curr_col>=0) and (curr_row<height)):
            curr_col = curr_col -1
            curr_row = curr_row + 1
            if((curr_row==height) or (curr_col <0)):
                break
            gray_cur = img_data[curr_row][curr_col] # 获得当前的灰度
            p2_init_new = int(p2_init /(abs(int(gray_cur)-int(gray_last))+1))
            for dd_idx in range(disp_range):
                cost = cost_init[curr_row][curr_col][dd_idx]
                if(dd_idx+2<disp_range):
                    cost_l1 = last_aggregation[dd_idx+1]
                    cost_l2 = last_aggregation[dd_idx] + p1
                    cost_l3 = last_aggregation[dd_idx + 2] + p1
                    cost_l4 = min_cost_d_last_path + max(p1,p2_init_new)
                    cost_new = cost + min(min(cost_l1,cost_l2),min(cost_l3,cost_l4)) -min_cost_d_last_path
                    cost_aggregation[curr_row][curr_col][dd_idx] = cost_new   # 更新cost
                else:
                    cost_l1 = 150
                    cost_l2 = last_aggregation[dd_idx] + p1
                    cost_l3 = 150
                    cost_l4 = min_cost_d_last_path + max(p1,p2_init_new)
                    cost_new = cost + min(min(cost_l1,cost_l2),min(cost_l3,cost_l4)) -min_cost_d_last_path
                    cost_aggregation[curr_row][curr_col][dd_idx] = cost_new   # 更新cost
            
            last_aggregation = cost_aggregation[curr_row][curr_col]
            gray_cur = gray_last
            min_cost_d_last_path = findMinIn1darray(last_aggregation)

    return cost_aggregation

# 从 2-3 两个方向对原始视察图进行聚合
def CostDiagonal32(img_data,min_disparity,max_disparity,p1,p2_init,cost_init):
        # Return Cost Aggregation OutCome from left and Right
    img_data = img_data.astype(np.uint8)
    height,width = img_data.shape[:2]
    if(img_data is None):
        print("Receive a NULL Data Input for CostAggregation func")
        return
    # 视差的范围
    disp_range = max_disparity - min_disparity
    cost_aggregation = np.zeros((height,width,disp_range)) # 初始化一个聚合矩阵， 全部为0
    curr_col = 0 # 第几列， y 坐标
    curr_row = 0 # 第几行， x坐标
    for i in range(height):
        curr_col = 0
        curr_row =i
        cost_aggregation[curr_row][curr_col] = cost_init[curr_row][curr_col] # 保持原始的Cost
        last_aggregation = cost_aggregation[curr_row][curr_col] # 保留上一个的 cost，记录为 last aggregation
        gray_cur = img_data[curr_row][curr_col]
        gray_last = img_data[curr_row][curr_col]  # 记录当前灰度和路径中上一个元素的灰度
        min_cost_d_last_path = findMinIn1darray(last_aggregation) # 找到上一个路径的最小值
        
        while((curr_row>=0) and (curr_col<width)):
            curr_col = curr_col  + 1
            curr_row = curr_row -1
            if((curr_row<0) or (curr_col ==width)):
                break
            gray_cur = img_data[curr_row][curr_col] # 获得当前的灰度
            p2_init_new = int(p2_init /(abs(int(gray_cur)-int(gray_last))+1))
            for d_idx in range(disp_range):
                cost= cost_init[curr_row][curr_col][d_idx]  # 原始的COST
                if(d_idx+2<disp_range):
                    cost_l1 = last_aggregation[d_idx+1]
                    cost_l2 = last_aggregation[d_idx] + p1
                    cost_l3 = last_aggregation[d_idx + 2] + p1
                    cost_l4 = min_cost_d_last_path + max(p1,p2_init_new)
                    cost_new = cost + min(min(cost_l1,cost_l2),min(cost_l3,cost_l4)) -min_cost_d_last_path
                    cost_aggregation[curr_row][curr_col][d_idx] = cost_new   # 更新cost
                else:
                    cost_l1 = 150
                    cost_l2 = last_aggregation[d_idx] + p1
                    cost_l3 = 150
                    cost_l4 = min_cost_d_last_path + max(p1,p2_init_new)
                    cost_new = cost + min(min(cost_l1,cost_l2),min(cost_l3,cost_l4)) -min_cost_d_last_path
                    cost_aggregation[curr_row][curr_col][d_idx] = cost_new   # 更新cost
            
            last_aggregation = cost_aggregation[curr_row][curr_col]
            gray_last = gray_cur
            min_cost_d_last_path = findMinIn1darray(last_aggregation)
            
    for j in range(1,width):
        curr_col = j
        curr_row = height -1 
        
        cost_aggregation[curr_row][curr_col] = cost_init[curr_row][curr_col] # 保持原始的Cost
        last_aggregation = cost_aggregation[curr_row][curr_col] # 保留上一个的 cost，记录为 last aggregation
        gray_cur = img_data[curr_row][curr_col]
        gray_last = img_data[curr_row][curr_col]  # 记录当前灰度和路径中上一个元素的灰度
        min_cost_d_last_path = findMinIn1darray(last_aggregation) # 找到上一个路径的最小值
    
        while((curr_row>=0) and (curr_col<width)):
            curr_col = curr_col +1
            curr_row = curr_row -1
            if((curr_row<0) or (curr_col ==width)):
                break
            gray_cur = img_data[curr_row][curr_col] # 获得当前的灰度
            p2_init_new = int(p2_init /(abs(int(gray_cur)-int(gray_last))+1))
            for dd_idx in range(disp_range):
                cost = cost_init[curr_row][curr_col][dd_idx]
                if(dd_idx+2<disp_range):
                    cost_l1 = last_aggregation[dd_idx+1]
                    cost_l2 = last_aggregation[dd_idx] + p1
                    cost_l3 = last_aggregation[dd_idx + 2] + p1
                    cost_l4 = min_cost_d_last_path + max(p1,p2_init_new)
                    cost_new = cost + min(min(cost_l1,cost_l2),min(cost_l3,cost_l4)) -min_cost_d_last_path
                    cost_aggregation[curr_row][curr_col][dd_idx] = cost_new   # 更新cost
                else:
                    cost_l1 = 150
                    cost_l2 = last_aggregation[dd_idx] + p1
                    cost_l3 = 150
                    cost_l4 = min_cost_d_last_path + max(p1,p2_init_new)
                    cost_new = cost + min(min(cost_l1,cost_l2),min(cost_l3,cost_l4)) -min_cost_d_last_path
                    cost_aggregation[curr_row][curr_col][dd_idx] = cost_new   # 更新cost
            
            last_aggregation = cost_aggregation[curr_row][curr_col]
            gray_cur = gray_last
            min_cost_d_last_path = findMinIn1darray(last_aggregation)

    return cost_aggregation

def CheckNumPaths(curr_path,num_paths):
    if (curr_path<num_paths):
        return False
    else:
        return True


def GetSecondSmallest(mat,smallest):
    second_min =mat[0]
    for i in range(len(mat)):
        if mat[i]==smallest:
            continue
        if (mat[i]<second_min):
            second_min = mat[i]
    return second_min
        


def findMinIn1darray(oneDarray):
    size = oneDarray.shape[0]
    min_value = oneDarray[0]
    for i in range(size):
        if (oneDarray[i]<min_value):
            min_value = oneDarray[i]
    return min_value

