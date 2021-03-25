from include import sgmutils
from enum import Enum
import numpy as np
import cv2
MAX = 1000

class Census_Size(Enum):
    # 为序列值指定value值
    Census5x5= 1
    Census9x7 = -1

class SgmOptions:
    def __init__(self,num_paths=8,census_type = Census_Size.Census5x5 ,min_disparity=0, max_disparity =64,p1=10,p2_init=150,min_speckle_area=50,unique_ratio =0.99):
        self.num_paths_ =num_paths
        self.min_disparity_= min_disparity
        self.max_disparity_ =  max_disparity
        self.p1_ =p1
        self.p2_init_ = p2_init
        self.census_type_ = census_type
        self.min_speckle_area_ = min_speckle_area
        self.unique_ratio_ = unique_ratio

class SemiGlobalMatching:
    def __init__(self, sgmOptions,img_width,img_height, left_image_data,right_image_data,is_initalized= False, whether_aggregation= False,whether_LRC= False):
        self.sgmOptions_ = sgmOptions
        self.img_width_ = img_width
        self.img_height_ = img_height
        self.right_image_data_ = right_image_data
        self.left_image_data_ = left_image_data
        self.is_initalized_ = is_initalized
        self.census_left_ = np.zeros((img_height,img_width))
        self.census_right_ = np.zeros((img_height,img_width))
        self.cost_init_ = np.zeros((img_height,img_width,(sgmOptions.max_disparity_-sgmOptions.min_disparity_)))
        self.disparity_left = np.zeros((img_height,img_width))
        self.cost_aggregation_ = np.zeros((img_height,img_width,(sgmOptions.max_disparity_-sgmOptions.min_disparity_)))
        self.whether_aggregation_ = whether_aggregation
        self.whether_LRC_ = whether_LRC
        
    def Initalize(self): 
        if((self.img_width_ ==0) or (self.img_height_==0)):
            print("Load Image Error ! Please Check the Image Path.")
            self.is_initalized_= False
        dsp_range = self.sgmOptions_.max_disparity_ - self.sgmOptions_.min_disparity_
        if(dsp_range<0):
            print("Disparity Range shoud larger than zero.")
            self.is_initalized_= False
        else:
            self.is_initalized_ = True
            print("Initalizing is Done.")
        return self.is_initalized_

    def Match(self): # return disp_left
        # First Check
        if(self.is_initalized_):
            self.is_initalized_ = False
        if((self.left_image_data_ is  None) or (self.right_image_data_ is None)):
            return
        
        # Census Computation
        self.CensusTransform() # Compute the Census 

        self.ComputeCost() # Output the Census Cost first: Hamming Distance
        print("Cost Compute is DONE")
        

        if(self.whether_aggregation_):
            self.CostAggregation() # Output The aggregation cost
            self.cost_init_ = self.cost_aggregation_
            print("Cost Aggregation is DONE")


        self.ComputeDisparity()  # Output The disparty
        print("Disparty Compute is DONE")


        if(self.whether_LRC_):
            self.LRCheck()  # LRCHECK
            print("LR Check is DONE")


        self.SubPixelize() # 子像素化，提高精度

        
        # self.RemoveSpeckles()
        # 中值滤波
        self.disparity_left = self.disparity_left.astype(np.uint8)    
        self.disparity_left  = cv2.medianBlur(self.disparity_left,3)
        
    def Reset(self,sgmOptions,img_width,img_height, left_image_data,right_image_data,is_initalized= False, whether_aggregation= False,whether_LRC= False):
        self.sgmOptions_ = sgmOptions
        self.img_width_ = img_width
        self.img_height_ = img_height
        self.right_image_data_ = right_image_data
        self.left_image_data_ = left_image_data
        self.is_initalized_ = is_initalized
        self.census_left_ = np.zeros((img_height,img_width))
        self.census_right_ = np.zeros((img_height,img_width))
        self.cost_init_ = np.zeros((img_height,img_width,(sgmOptions.max_disparity_-sgmOptions.min_disparity_)))
        self.disparity_left = np.zeros((img_height,img_width))
        self.cost_aggregation_ = np.zeros((img_height,img_width,(sgmOptions.max_disparity_-sgmOptions.min_disparity_)))
        self.whether_aggregation_ = whether_aggregation
        self.whether_LRC_ = whether_LRC
        print("sgm parameter has been reset")
        
    def CensusTransform(self): # return the cenus
        census_type = self.sgmOptions_.census_type_
        if (census_type.value):
            self.census_left_= sgmutils.census_transform_5x5(self.left_image_data_)
            self.census_right_ =sgmutils.census_transform_5x5(self.right_image_data_)
        else:
            self.census_left_= sgmutils.census_transform_9x7(self.left_image_data_)
            self.census_right_ =sgmutils.census_transform_9x7(self.right_image_data_)
           
    def ComputeCost(self): # return the cost
        for  i in range(self.img_height_):
            for j in range(self.img_width_):
                census_cur_left = self.census_left_[i][j]
                # Compute the HammingCost
                for d in range(self.sgmOptions_.min_disparity_,self.sgmOptions_.max_disparity_):
                    if((j-d<0) or ((j-d)>=self.img_width_)):
                        self.cost_init_[i][j][d-self.sgmOptions_.min_disparity_] = MAX//2
                        continue
                    census_cur_right = self.census_right_[i][j-d]
                    cost_d = sgmutils.Hamming(int(census_cur_left),int(census_cur_right))
                    self.cost_init_[i][j][d-self.sgmOptions_.min_disparity_] = cost_d

    def CostAggregation(self):
        print("Begin CostAggregation ...")
        aggregated_path_nums  = 0

        if(sgmutils.CheckNumPaths(aggregated_path_nums,self.sgmOptions_.num_paths_)):
            return 

        # 左---> 右 边聚合
        cost_aggregation_1 =  sgmutils.CostAggregationLeftRight(self.left_image_data_,self.sgmOptions_.min_disparity_,self.sgmOptions_.max_disparity_,
        self.sgmOptions_.p1_,self.sgmOptions_.p2_init_,self.cost_init_,True)
        print("CostAggregation Finished 1/{}".format(self.sgmOptions_.num_paths_))

        aggregated_path_nums = aggregated_path_nums +1
        if(sgmutils.CheckNumPaths(aggregated_path_nums,self.sgmOptions_.num_paths_)):
            self.cost_aggregation_ = cost_aggregation_1
            return

        # 右 ----> 左 边聚合
        cost_aggregation_2 = sgmutils.CostAggregationLeftRight(self.left_image_data_,self.sgmOptions_.min_disparity_,self.sgmOptions_.max_disparity_,
        self.sgmOptions_.p1_,self.sgmOptions_.p2_init_,self.cost_init_,False)
        print("CostAggregation Finished 2/{}".format(self.sgmOptions_.num_paths_))

        aggregated_path_nums = aggregated_path_nums +1
        if(sgmutils.CheckNumPaths(aggregated_path_nums,self.sgmOptions_.num_paths_)):
            self.cost_aggregation_ = cost_aggregation_1 + cost_aggregation_2
            return 

        # 上 -----> 下 聚合
        cost_aggregation_3 = sgmutils.CostAggregationUpDown(self.left_image_data_,self.sgmOptions_.min_disparity_,self.sgmOptions_.max_disparity_,
        self.sgmOptions_.p1_,self.sgmOptions_.p2_init_,self.cost_init_,True)
        print("CostAggregation Finished 3/{}".format(self.sgmOptions_.num_paths_))
        
        aggregated_path_nums = aggregated_path_nums +1
        if(sgmutils.CheckNumPaths(aggregated_path_nums,self.sgmOptions_.num_paths_)):
            self.cost_aggregation_ = cost_aggregation_1 + cost_aggregation_2 +cost_aggregation_3
            return 

        # 下 ----> 上 聚合
        cost_aggregation_4 = sgmutils.CostAggregationUpDown(self.left_image_data_,self.sgmOptions_.min_disparity_,self.sgmOptions_.max_disparity_,
        self.sgmOptions_.p1_,self.sgmOptions_.p2_init_,self.cost_init_,False)
        print("CostAggregation Finished 4/{}".format(self.sgmOptions_.num_paths_))

        aggregated_path_nums = aggregated_path_nums +1
        if(sgmutils.CheckNumPaths(aggregated_path_nums,self.sgmOptions_.num_paths_)):
            self.cost_aggregation_ = cost_aggregation_1 + cost_aggregation_2 +cost_aggregation_3 +cost_aggregation_4            
            return 
        
        #从14对角线方向聚合
        cost_aggregation_5 = sgmutils.CostDiagonal14(self.left_image_data_,self.sgmOptions_.min_disparity_,self.sgmOptions_.max_disparity_,
        self.sgmOptions_.p1_,self.sgmOptions_.p2_init_,self.cost_init_)
        print("CostAggregation Finished 5/{}".format(self.sgmOptions_.num_paths_))
        
        aggregated_path_nums = aggregated_path_nums +1
        if(sgmutils.CheckNumPaths(aggregated_path_nums,self.sgmOptions_.num_paths_)):
            self.cost_aggregation_ = cost_aggregation_1 + cost_aggregation_2 +cost_aggregation_3 +cost_aggregation_4+cost_aggregation_5    
            return     
        
        
        #从41对角线的方向聚合
        cost_aggregation_6 = sgmutils.CostDiagonal41(self.left_image_data_,self.sgmOptions_.min_disparity_,self.sgmOptions_.max_disparity_,
        self.sgmOptions_.p1_,self.sgmOptions_.p2_init_,self.cost_init_)
        print("CostAggregation Finished 6/{}".format(self.sgmOptions_.num_paths_))
        
        aggregated_path_nums = aggregated_path_nums +1
        if(sgmutils.CheckNumPaths(aggregated_path_nums,self.sgmOptions_.num_paths_)):
            self.cost_aggregation_ = cost_aggregation_1 + cost_aggregation_2 +cost_aggregation_3 +cost_aggregation_4+cost_aggregation_5 +cost_aggregation_6
            return 
        
        
         # 从23两个方向进行聚合
        cost_aggregation_7 = sgmutils.CostDiagonal23(self.left_image_data_,self.sgmOptions_.min_disparity_,self.sgmOptions_.max_disparity_,
        self.sgmOptions_.p1_,self.sgmOptions_.p2_init_,self.cost_init_)
        print("CostAggregation Finished 7/{}".format(self.sgmOptions_.num_paths_))
        
        aggregated_path_nums = aggregated_path_nums +1
        if(sgmutils.CheckNumPaths(aggregated_path_nums,self.sgmOptions_.num_paths_)):
            self.cost_aggregation_ = cost_aggregation_1 + cost_aggregation_2 +cost_aggregation_3 +cost_aggregation_4+cost_aggregation_5 +cost_aggregation_6+cost_aggregation_7 
            return 
        
        # 从 32两个方向进行聚合
        cost_aggregation_8 = sgmutils.CostDiagonal32(self.left_image_data_,self.sgmOptions_.min_disparity_,self.sgmOptions_.max_disparity_,
        self.sgmOptions_.p1_,self.sgmOptions_.p2_init_,self.cost_init_)
        print("CostAggregation Finished 8/{}".format(self.sgmOptions_.num_paths_))

        aggregated_path_nums = aggregated_path_nums +1
        if(sgmutils.CheckNumPaths(aggregated_path_nums,self.sgmOptions_.num_paths_)):
            self.cost_aggregation_ = cost_aggregation_1 + cost_aggregation_2 + cost_aggregation_3 + cost_aggregation_4 + cost_aggregation_5 \
        + cost_aggregation_6 +cost_aggregation_7+ cost_aggregation_8

    def ComputeDisparity(self):
        for i in range(self.img_height_):
            for j in range(self.img_width_):
                best_disparity = 0
                max_cost =0
                min_cost = MAX
                
                # WTA:  Get the Least(Smallest) ONE 
                for d in range(self.sgmOptions_.min_disparity_,self.sgmOptions_.max_disparity_):
                    cost_now = self.cost_init_[i][j][d-self.sgmOptions_.min_disparity_]
                    if (cost_now<min_cost):
                        min_cost = cost_now
                        best_disparity = d
         
                    max_cost = max(max_cost,cost_now)
                # Get the second smallest cost
                second_min_cost = sgmutils.GetSecondSmallest(self.cost_init_[i][j],min_cost)
                unique_ratio = (min_cost *1.0 )/(second_min_cost*1.0+0.0001)
                if (unique_ratio>self.sgmOptions_.unique_ratio_):
                    self.disparity_left[i][j]=0  # Invalid
                # Max Last CHECK, Show the 3 Year
                if (max_cost!= min_cost):
                    self.disparity_left[i][j] = best_disparity
                else:
                    self.disparity_left[i][j]=0
    
    def LRCheck(self):
        # 右边矩阵代价DSI矩阵进行初始化
        height,width = self.right_image_data_.shape[:2]
        disp_range =  self.sgmOptions_.max_disparity_ - self.sgmOptions_.min_disparity_
        cost_init_right = np.zeros((height,width,disp_range))
        for i in range(height):
            for j in range(width):
                for d_idx in range(disp_range):
                    if(j+d_idx+self.sgmOptions_.min_disparity_<width):
                        cost_init_right[i][j][d_idx] = self.cost_aggregation_[i][j+d_idx+self.sgmOptions_.min_disparity_][d_idx]
       # 计算出右边的disparity ---> PASS
        temp_left  = self.disparity_left 
        self.cost_init_ = cost_init_right
        self.ComputeDisparity()
        temp_right = self.disparity_left
        for i in range(height):
            for j in range(width):
                if(j - int(temp_left[i][j]) >=0 ):
                    d_left  = temp_left[i][j]
                    d_left = int(d_left)
                    d_right = temp_right[i][j-d_left]
                    d_right =int(d_right)
                    if d_left!=d_right:
                        temp_left[i][j] = 0
        self.disparity_left = temp_left

    def SubPixelize(self):
                # get the left Side Cost
            for i in range(self.img_height_):
                for j in range(self.img_width_):
                    best_disparity = self.disparity_left[i][j]
                    best_disparity = int(best_disparity)
                    left_disparity_id  = best_disparity -1 -self.sgmOptions_.min_disparity_
                    right_disparity_id = best_disparity + 1 -self.sgmOptions_.min_disparity_
                    if(right_disparity_id>=self.sgmOptions_.max_disparity_):
                        right_disparity_id = best_disparity -self.sgmOptions_.min_disparity_
                    if(left_disparity_id<=0):
                        left_disparity_id = best_disparity -self.sgmOptions_.min_disparity_
                    left_disparity_cost  = self.cost_aggregation_[i][j][left_disparity_id]    # c1
                    right_disparity_cost  = self.cost_aggregation_[i][j][right_disparity_id] # c2
                    cur_best_disparity_cost = self.cost_aggregation_[i][j][best_disparity-self.sgmOptions_.min_disparity_]  # c0
                    best_disparity_updated = best_disparity *1.0 + (left_disparity_cost-right_disparity_cost)/(left_disparity_cost*1.0+right_disparity_cost*1.0-2.0*cur_best_disparity_cost +0.00001)               
                    self.disparity_left[i][j] = best_disparity_updated
                    
    def RemoveSpeckles(self):
        visited = np.zeros((self.img_height_,self.img_width_))
        visited = visited.astype(np.uint8)
        for i in range(self.img_height_):
            for j in range(self.img_width_):
                if((int(visited[i][j])==1)):
                    continue  # 跳过访问过的元素和无效的元素
                speckle_areas = [] # 目前设置为一个连通的集合，数量设置为连通的个数
                speckle_areas.append([i,j]) #添加当前的元素
                disp_base = self.disparity_left[i][j] # GET 当前的深度
                visited[i][j] = 1   # 标记访问
                curr_p  = 0
                next_p  = 0
                while(next_p<len(speckle_areas)): 
                    next_p = len(speckle_areas)
                    for k in range(curr_p,next_p):
                        pixel = speckle_areas[k]
                        row = pixel[0]
                        col = pixel[1]
                        disp_base = self.disparity_left[row][col] # 输出当前的深度

                        disp_base = int (disp_base)
                        for m in range(-1,2):
                            for n in range(-1,2):
                                if((m==0) and (n==0)):
                                    continue
                                rowc = row + m
                                colc = col +n
                                if(  (rowc>=0) and (rowc<self.img_height_) and (colc>=0) and (colc<self.img_width_) ):

                                    diff  = int(abs(self.disparity_left[rowc][colc] - disp_base))

                                    if( (diff<=8)    and (int(visited[rowc][colc]) ==0) ) :
                                        speckle_areas.append([rowc,colc])
                                        visited[rowc][colc] = 1
                    curr_p  = next_p
                if(len(speckle_areas)<5):
                    for pos in speckle_areas:
                        self.disparity_left[pos[0]][pos[1]] = 0