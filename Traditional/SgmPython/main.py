import cv2
import matplotlib.pyplot as plt
import os
from include.SemiGlobalMatching import SemiGlobalMatching,SgmOptions
from include.SemiGlobalMatching import Census_Size
from include import sgmutils
import  numpy as np


if __name__=="__main__":
    # Image Paths
    LeftImageFilePath ="TestData/cone/im2.png"
    RightImageFilePath ="TestData/cone/im6.png"
    LeftDepthPath ="TestData/cone/disp2.png"
    RightDepthPath ="TestData/cone/disp6.png"
    
    # Load Image
    img_left_bgr = cv2.imread(LeftImageFilePath,1)
    img_left_gray = cv2.imread(LeftImageFilePath,0)
    img_right_gray = cv2.imread(RightImageFilePath,0)
    img_left_depth = cv2.imread(LeftDepthPath,-1)

    if(img_left_gray.size!=img_right_gray.size):
        print("Error : Input SIze should be Same")
    
    # Convert to Numpy 
    img_left_data = np.array(img_left_gray)
    img_right_data = np.array(img_right_gray)
    img_height, img_width = img_left_gray.shape[:2]

    sgmOptions = SgmOptions(num_paths=8,census_type=Census_Size.Census5x5,min_disparity=0,max_disparity=64,p1=10,p2_init=150,min_speckle_area=50,unique_ratio=0.99)
    sgmSolver = SemiGlobalMatching(sgmOptions=sgmOptions,img_height=img_height,img_width=img_width,left_image_data=img_left_data,right_image_data=img_right_data,is_initalized=False,whether_aggregation=False,whether_LRC=False)
    sgmSolver.Initalize()
    sgmSolver.Match()

    # # Just FO TEST
    # sgmSolver2 = SemiGlobalMatching(sgmOptions=sgmOptions,img_height=img_height,img_width=img_width,left_image_data=img_left_data,right_image_data=img_right_data,is_initalized=False,whether_aggregation=True,whether_LRC=False)
    # sgmSolver2.Initalize()
    # sgmSolver2.Match()
    
    
    sgmSolver.disparity_left= sgmSolver.disparity_left.astype(np.uint8)
    # sgmSolver2.disparity_left= sgmSolver2.disparity_left.astype(np.uint8)

    cv2.imshow("Orginal LEFT", img_left_bgr)
    cv2.imshow("disparity Brutal LEFT",sgmSolver.disparity_left)
    # cv2.imshow("disparity Aggregation LEFT",sgmSolver2.disparity_left)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
 


    # census_left  = sgmutils.census_transform_9x7(img_left_data)
    # census_right = sgmutils.census_transform_5x5(img_right_data)
    # plt.imshow(census_left)
    # plt.imshow(census_right)
    # plt.show()
    






    # cv2.imshow("Orginal Left Image", img_left_bgr)
    # cv2.imshow("Left Image Depth: Ground Truth",img_left_depth)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
