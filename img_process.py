import cv2  
import numpy as np
import time
  
# 读取图片  
image_rgb = cv2.imread('001791.png')  
  
# 确保图片是RGB格式（OpenCV默认读取为BGR，需要转换）  
# image_rgb = cv2.cvtColor(image)  

# 定义要保留的颜色  
target_color = np.array([142, 0, 0])  
  
# 计算颜色阈值（考虑一定的容差，因为颜色可能不完全匹配）  
threshold = 10  # 容差范围，可以根据需要调整  
lower_bound = target_color - threshold  
upper_bound = target_color + threshold  

time1 = time.time()
# 创建一个与原图大小相同的全黑图像  
masked_image = np.zeros_like(image_rgb)  

# 使用颜色范围创建掩码  
mask = cv2.inRange(image_rgb, lower_bound.reshape(1, 1, 3), upper_bound.reshape(1, 1, 3))  
  
# 将掩码应用到原图上，只保留匹配颜色的像素  
masked_image[mask == 255] = image_rgb[mask == 255]
time2 = time.time()
print(1/(time2-time1)) 
  
# 将处理后的图像保存或显示  
cv2.imwrite('processed_image.png', masked_image)  
# 或者使用cv2.imshow('Processed Image', final_image)来显示图像，然后按任意键关闭窗口。