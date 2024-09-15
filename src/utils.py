import os
import cv2 as cv
import matplotlib.pyplot as plt

def find_coupled_frame(index, step_size, args):
    """
    Find the last inpainted frame in the save directory with reference to the given index

    Args:
        index: the current frame index
        step_size: the step size
        args: the arguments passed to the script
    """
    files = os.listdir(args['save_dir'])
    last_path = sorted(files)[-1]

    return os.path.join(args['save_dir'], last_path)


def crop_rgb_frame(prev_frame, next_frame, inpainted_frame, args):
    """
    Crop the inpainted frame to the approximate the touched region

    Args:
        prev_frame: the previous frame
        next_frame: the next frame
        inpainted_frame: the inpainted frame
        args: the arguments passed to the script
    """
    RIGHT_SIZE = 50
    
    h, w, _ = inpainted_frame.shape
    inpainted_frame_crop = inpainted_frame[150:, 200:w-RIGHT_SIZE]
    plt.subplot(1, 2, 1)
    plt.imshow(inpainted_frame)
    plt.title('Inpainted Frame')
    plt.subplot(1, 2, 2)
    plt.imshow(inpainted_frame_crop)
    plt.title('Inpainted Frame Crop')
    plt.show()

    return inpainted_frame_crop

def save_samples(rgb_image, touch_image, args):
    """
    Save the RGB and touch images for comparison in the data folder

    Args:
        rgb_image: the RGB image
        touch_image: the touch image
        args: the arguments passed to the script
    """
    dir_name = args['image'].split('_')[-1]
    dir_name = "data/" + dir_name
    os.makedirs(dir_name, exist_ok=True)

    #create subdirs for touch and rgb images
    os.makedirs(f'{dir_name}/video_frame', exist_ok=True)
    os.makedirs(f'{dir_name}/gelsight_frame', exist_ok=True)
    video_dir = f'{dir_name}/video_frame'
    touch_dir = f'{dir_name}/gelsight_frame'
    
    index = len(os.listdir(f'{dir_name}/video_frame'))
    cv.imwrite(f'{video_dir}/rgb_image_{index:03d}.png', rgb_image)
    cv.imwrite(f'{touch_dir}/touch_image_{index:03d}.png', touch_image)
