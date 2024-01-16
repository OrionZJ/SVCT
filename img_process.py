import numpy as np
# import pydicom
import matplotlib.pyplot as plt
from skimage.transform import radon, iradon, rescale
import SimpleITK as sitk

"""
# 读取dcm文件  
src = 'manifest-1704460560034/LDCT-and-Projection-data/C004/12-23-2021-NA-NA-40816/1.000000-Full Dose Images-73627/1-048.dcm'  
my_dcm = pydicom.dcmread(src)  

info18 = my_dcm.WindowCenter
info19 = my_dcm.WindowWidth
info20 = my_dcm.RescaleIntercept
info21 = my_dcm.RescaleSlope
info22 = my_dcm.pixel_array

ct_image = info21 * info22 + info20

'''调窗'''
def adjustMethod1(data_resampled,w_width,w_center):
    val_min = w_center - (w_width / 2)
    val_max = w_center + (w_width / 2)
    data_adjusted = data_resampled.copy()
    data_adjusted[data_resampled < val_min] = val_min
    data_adjusted[data_resampled > val_max] = val_max
    return data_adjusted


plt.figure(figsize=(10, 10))
plt.axis('off')
plt.imshow(ct_image,'gray')
plt.show()

ct_image = adjustMethod1(ct_image, info19, info18)

plt.figure(figsize=(10, 10)) # 适配屏幕
plt.axis('off')
plt.imshow(ct_image,'gray')
plt.show()

"""


def img2sino(image, L, views, circle=False, visualize=False):
    image = rescale(image, scale=L / min(image.shape), mode='reflect', channel_axis=None)
    theta = np.linspace(0, 180, views, endpoint=False)
    sinogram = radon(image, theta=theta, circle=circle)
    if visualize:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4.5))
        ax1.set_title("Original")
        ax1.imshow(image, cmap=plt.cm.Greys_r)
        dx, dy = 0.5 * 180 / views, 0.5 / sinogram.shape[0]
        ax2.set_title("Radon transform\n(Sinogram)")
        ax2.set_xlabel("Projection angle (deg)")
        ax2.set_ylabel("Projection position (pixels)")
        ax2.imshow(sinogram, cmap=plt.cm.Greys_r,
                   extent=(-dx, 180 + dx, -dy, sinogram.shape[0] + dy),
                   aspect='auto')

        fig.tight_layout()
        plt.show()
    return sinogram


def sino2img(sinogram, gt_image=None, circle=False, visualize=False):
    theta = np.linspace(0, 180, sinogram.shape[1], endpoint=False)
    reconstruction_fbp = iradon(radon_image=sinogram, theta=theta, filter_name='ramp', circle=circle)
    if visualize:
        if gt_image is not None:
            gt_image = rescale(gt_image, scale=reconstruction_fbp.shape[0] / gt_image.shape[0], mode='reflect',
                               channel_axis=None)
            error = reconstruction_fbp - gt_image
            print(f'FBP rms reconstruction error: {np.sqrt(np.mean(error ** 2)):.3g}')
        imkwargs = dict(vmin=-0.2, vmax=0.2)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4.5),
                                       sharex=True, sharey=True)
        ax1.set_title("Reconstruction\nFiltered back projection")
        ax1.imshow(reconstruction_fbp, cmap=plt.cm.Greys_r)
        if gt_image is not None:
            ax2.set_title("Reconstruction error\nFiltered back projection")
            ax2.imshow(reconstruction_fbp - gt_image, cmap=plt.cm.Greys_r, **imkwargs)
        plt.show()
    return reconstruction_fbp


if __name__ == '__main__':
    src = "../SCOPE/data/gt_img.nii"
    # src = "E:/work/TCIA/manifest-1704284681625/LDCT-and-Projection-data/C001/1.2.840.113713.4.100.1.2.110644788119750551356682561800775/1.2.840.113713.4.100.1.2.719060684113555115309634524032775/1-011.dcm"
    image = sitk.GetArrayFromImage(sitk.ReadImage(src))
    # image = image[0]
    L = 367
    views = 180
    sinogram = img2sino(image, L, views, visualize=True)
    sino2img(sinogram, image, visualize=True)
