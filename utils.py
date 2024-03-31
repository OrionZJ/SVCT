import numpy as np
import pydicom
import matplotlib.pyplot as plt
from skimage.transform import radon, iradon, rescale, resize, rotate
import SimpleITK as sitk


def read_dcm_file(file_path):
    """读取DCM文件并返回pydicom对象"""
    return pydicom.dcmread(file_path)


def extract_dcm_info(dcm_obj):
    """从pydicom对象中提取所需信息"""
    info = {
        'WindowCenter': dcm_obj.WindowCenter,
        'WindowWidth': dcm_obj.WindowWidth,
        'RescaleIntercept': dcm_obj.RescaleIntercept,
        'RescaleSlope': dcm_obj.RescaleSlope,
        'pixel_array': dcm_obj.pixel_array
    }
    return info


def compute_ct_image(info):
    """根据提取的信息计算CT图像"""
    ct_image = info['RescaleSlope'] * info['pixel_array'] + info['RescaleIntercept']
    return ct_image


def setDicomWinWidthWinCenter(data_resampled, info):
    w_width = info['WindowWidth']
    w_center = info['WindowCenter']
    """设置CT图像的窗宽和窗位"""
    if isinstance(w_width, int):
        pass
    elif w_width.__class__.__name__ == 'MultiValue':
        w_width = (w_width[0] + w_width[1]) / 2

    if isinstance(w_center, int):
        pass
    elif w_center.__class__.__name__ == 'MultiValue':
        w_center = (w_center[0] + w_center[1]) / 2

    val_min = w_center - (w_width / 2)
    val_max = w_center + (w_width / 2)
    data_adjusted = data_resampled.copy()
    data_adjusted[data_resampled < val_min] = val_min
    data_adjusted[data_resampled > val_max] = val_max
    return data_adjusted


def display_image(image, figsize=(10, 10)):
    """显示图像"""
    plt.figure(figsize=figsize)
    plt.axis('off')
    plt.imshow(image, 'gray')
    plt.show()


def img2sino(image, pixels, views, circle=False, visualize=False):
    # 归一化到0到1范围
    # image = (image - image.min()) / (image.max() - image.min())
    theta = np.linspace(90, 270, views, endpoint=False)
    sinogram = radon(image, theta=theta, circle=circle)
    sinogram = resize(sinogram, (pixels, views))
    sinogram = np.rot90(sinogram, 1)
    if visualize:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4.5))
        ax1.set_title("Original")
        ax1.imshow(image, cmap=plt.cm.Greys_r)
        dy, dx = 0.5 * 180 / views, 0.5 / sinogram.shape[0]
        ax2.set_title("Radon transform\n(Sinogram)")
        ax2.set_xlabel("Projection position (pixels)")
        ax2.set_ylabel("Projection angle (deg)")
        ax2.imshow(sinogram, cmap=plt.cm.Greys_r,
                   extent=(-dx, 180 + dx, -dy, sinogram.shape[0] + dy),
                   aspect='auto')

        fig.tight_layout()
        plt.show()
    return sinogram


def sino2img(sinogram, gt_image=None, circle=False, visualize=False):
    sinogram = np.rot90(sinogram, -1)
    theta = np.linspace(90, 270, sinogram.shape[1], endpoint=False)
    reconstruction_fbp = iradon(radon_image=sinogram, theta=theta, filter_name='hann', circle=circle)
    if gt_image is not None:
        reconstruction_fbp = resize(reconstruction_fbp, (gt_image.shape[0], gt_image.shape[1]))
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
    """
    # 测试nii读取
    src = "../SCOPE/data/gt_img.nii"
    # src = "E:/work/TCIA/manifest-1704284681625/LDCT-and-Projection-data/C001/1.2.840.113713.4.100.1.2.110644788119750551356682561800775/1.2.840.113713.4.100.1.2.719060684113555115309634524032775/1-011.dcm"
    image = sitk.GetArrayFromImage(sitk.ReadImage(src))
    
    # image = image[0]
    L = 367
    views = 180
    sinogram = img2sino(image, L, views, visualize=True)
    sino2img(sinogram, image, visualize=True)


    """
    # 测试读取dcm文件
    path = "E:/work/CT Medical Images_datasets/CT Medical Images_datasets/CT Medical Images_dicom_dir_datasets/"
    id = "ID_0099_AGE_0061_CONTRAST_0_CT.dcm"
    src = path + id
    # src = "E:/work/TCIA/manifest-1704460560034/LDCT-and-Projection-data/C009/02-04-2022-NA-NA-40176/302.000000-Full Dose Images-52856/1-026.dcm"

    my_dcm = read_dcm_file(src)
    info = extract_dcm_info(my_dcm)
    ct_image = compute_ct_image(info)
    # ct_image = setDicomWinWidthWinCenter(ct_image, info)
    from scipy.ndimage import zoom
    zoom_factor = 256 / ct_image.shape[0], 256 / ct_image.shape[1]
    ct_image = zoom(ct_image, zoom_factor, order=3)
    # 归一化到0到1范围
    ct_image = (ct_image - ct_image.min()) / (ct_image.max() - ct_image.min())

    sitk.WriteImage(sitk.GetImageFromArray(ct_image), "./data/new_gt_img.nii")

    display_image(ct_image)

    sino = img2sino(ct_image, pixels=367, views=90, visualize=True)
    sino2img(sino, ct_image, visualize=True)
    sino = sitk.GetImageFromArray(sino)
    writepath = "./data/new_90_sino.nii"
    sitk.WriteImage(sino, writepath)

