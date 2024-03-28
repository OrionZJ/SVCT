from utils import *
import utils
import SimpleITK as sitk

import numpy as np

# 主程序
src = 'data/1-097.dcm'
my_dcm = read_dcm_file(src)

info = extract_dcm_info(my_dcm)
ct_image = compute_ct_image(info)

display_image(ct_image)

ct_image_adjusted = adjust_window(ct_image, info['WindowWidth'], info['WindowCenter'])
display_image(ct_image_adjusted)

# 创建一个二维图像

sv_img_path = "data/1-097.dcm"
sv_img = sitk.ReadImage(sv_img_path)
sv_img = sitk.GetArrayFromImage(sv_img )
sv_img = sv_img[0]
sv_sino = utils.img2sino(ct_image_adjusted, 367, 720, visualize=True, circle=True)
sv_recon_img = utils.sino2img(sv_sino, sv_img, visualize=True, circle=True)

sv_sino = sitk.GetImageFromArray(sv_sino)
save_path = "90_scikit_sino.nii"
sitk.WriteImage(sv_sino, save_path)

