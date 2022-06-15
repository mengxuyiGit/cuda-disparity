import time
import argparse
import numpy as np
import torch
from ipdb import set_trace as st

import cv2 as cv
from PIL import Image 
# import matplotlib.pyplot as plt

torch.manual_seed(0)
ntest = 1

def load_rectified_gray_image():
    # Read both images and convert to grayscale
    img1 = cv.imread(args.left_image, cv.IMREAD_GRAYSCALE) # (768, 1024) np array
    img2 = cv.imread(args.right_image, cv.IMREAD_GRAYSCALE)
    assert img1.shape == img2.shape
    h,w = img1.shape
    d_range = 60
    d_min = args.min_disparity
    print("Disparity value ranges from {} to {}".format(d_min, d_min+d_range))
   
    img1_tensor = torch.from_numpy(img1).to(dtype=torch.float32, device="cuda:0").contiguous()
    img2_tensor = torch.from_numpy(img2).to(dtype=torch.float32, device="cuda:0").contiguous()

    disp_map = torch.zeros((h,w), dtype=torch.float32, device="cuda:0" )
    cost_volume = torch.zeros((h,w,d_range),dtype=torch.float32, device="cuda:0") # cost can remain as float
   
    img_times = list()
    for _ in range(ntest):
        torch.cuda.synchronize(device="cuda:0")
        start_time = time.time()
        sub3.torch_launch_sub3(cost_volume, img1_tensor, img2_tensor, disp_map, h, w, d_range, d_min)
        torch.cuda.synchronize(device="cuda:0")
        end_time = time.time()
        # ------time ends-------
        img_times.append((end_time-start_time)*1e6)
    print("disp calculation time:  {:.3f}us".format(np.mean(img_times)))
    print("cost_volume:", cost_volume.dtype)
    print("disp map shape of 3 gray images:", disp_map.shape)
    
    # save disparity as gray scale image
    disp_map_arr = disp_map.cpu().detach().numpy()
    disp_map_arr = (disp_map_arr/d_range)*255
    disp_img = Image.fromarray(disp_map_arr.astype(np.uint8), 'L')
    disp_img_path = "results/disp_img2_dmin_{}.jpg".format(d_min)
    disp_img.save(disp_img_path)
    print("disparity image saved at: {}".format(disp_img_path))
    

    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--min_disparity', type=int, default=0)
    parser.add_argument('--left_image', type=str, default='rectified_1.png')
    parser.add_argument('--right_image', type=str, default='rectified_2.png')
    
    args = parser.parse_args()

    import sub3
    load_rectified_gray_image()
