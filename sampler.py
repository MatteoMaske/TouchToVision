import argparse
import cv2 as cv
import os, shutil
import numpy as np
import matplotlib.pyplot as plt

from src.monocularDepth import compute_monocular_depth
from src.utils import *

def parse_args():
	
    ap = argparse.ArgumentParser()
    # ap.add_argument("-s", "--starting_frame", type=int, default=140)
    ap.add_argument("--step_size", type=int, default=20)
    ap.add_argument("-i", "--image", type=str, default="rock",
                    choices=['grass', 'rock', 'tree', 'rock2'],
                    help="frames to inpaint")
    ap.add_argument("-a", "--method", type=str, default="telea",
        choices=["telea", "ns"],
        help="inpainting algorithm to use")
    ap.add_argument("-r", "--radius", type=int, default=3, help="inpainting radius")
    ap.add_argument("-k", "--scaling_factor", type=float, default=2.0, help="scaling factor for motion mask")
    ap.add_argument("--max-magnitude", type=int, default=75, help="maximum magnitude for the motion mask")
    ap.add_argument("--sampling-rate", type=int, default=10, help="sampling rate to save the inpaintings")
    ap.add_argument("--save-dir", type=str, default="inpaintings", help="directory to save the inpaintings")
    
    args = vars(ap.parse_args())

    if args['image'] == 'grass':
        args['image'] = '20220318_020426_grass' 
    elif args['image'] == 'rock':
        args['image'] = '20220319_002026_rock'
    elif args['image'] == 'tree':
        args['image'] =  '20220410_032506_tree'
    elif args['image'] == 'rock2':
        args['image'] = '20220319_002203_rock2'

    args['method'] = cv.INPAINT_TELEA if args['method'] == 'telea' else cv.INPAINT_NS

    return args

def find_best_step_parameters(frames_files_rgb, args):
    debug = False
    prev_frame = cv.imread(frames_files_rgb[0])
    max_magnitude, best_step_size = 0, 0

    for i in range(args['step_size'], 100, args['step_size']):
        next_frame = cv.imread(frames_files_rgb[i])
        H = compute_homography(prev_frame, next_frame)
        if H is None:
            continue
        tx, ty = H[0, 2], H[1, 2]
        magnitude = np.sqrt(tx**2 + ty**2)
        if magnitude > max_magnitude:
            max_magnitude = magnitude
            best_step_size = i

        # Just for debugging purposes
        if debug:
            warped_frame = warp_frame_sift(prev_frame, next_frame, H)
            cv.arrowedLine(warped_frame, (256,256) , (int(tx)+256, int(ty)+256), (0, 255, 0), 1)
            cv.putText(warped_frame, f"tx:{tx:.2f}, ty:{ty:.2f}, mag:{magnitude:.2f}, frame:{i}", (0, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv.imshow('warped frame', warped_frame)
            if cv.waitKey(0) & 0xFF == ord('q'):
                break

    print(f"Best step size: {best_step_size}, magnitude: {max_magnitude}")
    

    max_magnitude=0
    best_start=0
    max_frame_per_element = 300
    end_index = min(len(frames_files_rgb)-best_step_size, max_frame_per_element)

    for i in range(0, end_index, best_step_size):
        prev_frame = cv.imread(frames_files_rgb[i])
        next_frame = cv.imread(frames_files_rgb[i+best_step_size])
        H = compute_homography(prev_frame, next_frame)
        if H is None:
            continue
        tx, ty = H[0, 2], H[1, 2]
        magnitude = np.sqrt(tx**2 + ty**2)
        if magnitude > max_magnitude and magnitude < args['max_magnitude']:
            max_magnitude = magnitude
            best_start = i
            
    print(f"Current best start: {best_start}, magnitude: {max_magnitude}")

    return best_start, best_step_size

def find_max_pressure(frames_files_touch, start_frame=0, step_size=50):
    """
    Find the frame with the maximum pressure using the sum of the pixel values
    """
    rest_frame_touch = cv.imread(frames_files_touch[start_frame+step_size])
    min_frame_sum = np.sum(rest_frame_touch)
    max_pressure_index = start_frame+step_size

    start_index = start_frame + step_size
    end_index = start_frame + step_size + step_size//2

    # Touch frames processing ========================================
    for i in range(start_index, end_index):
        frame_touch = cv.imread(frames_files_touch[i])
        # cv.imshow(f'frame touch {i}', frame_touch)
        # cv.waitKey(0)
        
        if np.sum(frame_touch) < min_frame_sum:
            min_frame_sum = np.sum(frame_touch)
            max_pressure_index = i

    print(f"Max pressure frame: {max_pressure_index}, sum: {min_frame_sum}")
    return min_frame_sum, max_pressure_index


def compute_optical_flow(prev_frame, next_frame):
    prev_gray = cv.cvtColor(prev_frame, cv.COLOR_BGR2GRAY)
    next_gray = cv.cvtColor(next_frame, cv.COLOR_BGR2GRAY)
    
    flow = cv.calcOpticalFlowFarneback(prev_gray, next_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    return flow

def create_mask(magnitude, k):
    # Threshold the magnitude to create a binary mask
    mean_magnitude = np.mean(magnitude)
    std_magnitude = np.std(magnitude)

    # Set threshold to mean + k * std, k is a hyperparameter
    threshold = mean_magnitude + k * std_magnitude
    # print(f"Threshold: {threshold}")

    mask = np.zeros_like(magnitude, dtype=np.uint8)
    mask[magnitude > threshold] = 255
    return mask

def clean_mask(mask):
    #remove all the white pixels not in the bottom right corner

    cleaned_mask = mask.copy()
    h, w = cleaned_mask.shape
    cleaned_mask[:h//2, :] = 0
    cleaned_mask[:, :w//2] = 0

    return cleaned_mask

def create_motion_mask(prev_frame, next_frame, args):

    flow = compute_optical_flow(prev_frame, next_frame)
    magnitude = compute_magnitude(flow)

    motion_mask = create_mask(magnitude, k=args["scaling_factor"])
    cleaned_motion_mask = clean_mask(motion_mask)

    return cleaned_motion_mask

def warp_frame_of(prev_frame, flow):
    """
    Warps the previous frame using the optical flow field to align with the current frame
    Applicable only for subsequent frames

    Args:
        prev_frame: Previous frame
        flow: Optical flow field
    
    Returns:
        warped_frame: Warped frame
    """
    h, w = flow.shape[:2]
    grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))

    flow_map = np.float32(np.stack((grid_x + flow[..., 0], grid_y + flow[..., 1]), axis=-1))
    warped_frame = cv.remap(prev_frame, flow_map, None, cv.INTER_LINEAR)
    return warped_frame

def compute_homography(prev_frame, current_frame):
    # Convert to grayscale
    prev_gray = cv.cvtColor(prev_frame, cv.COLOR_BGR2GRAY)
    current_gray = cv.cvtColor(current_frame, cv.COLOR_BGR2GRAY)

    # Detect SIFT keypoints and descriptors
    sift = cv.SIFT_create()
    kp1, des1 = sift.detectAndCompute(prev_gray, None)
    kp2, des2 = sift.detectAndCompute(current_gray, None)

    # Match descriptors using FLANN-based matcher
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)

    # Store all the good matches as per Lowe's ratio test.
    good_matches = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)

    # Extract location of good matches
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    # check if we have enough points to compute the homography
    if len(src_pts) < 4 or len(dst_pts) < 4:
        print("Not enough points to compute homography")
        return None

    # Find homography matrix
    M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)

    return M

def warp_frame_sift(prev_frame, current_frame, M=None):
    
    # Compute the homography matrix
    if not M:
        M = compute_homography(prev_frame, current_frame)

    # Warp the previous frame to align with the current frame
    h, w = current_frame.shape[:2]
    # verify the homography matrix to be used for warping
    if M is None:
        print("Invalid homography matrix")
        return None
    aligned_frame_next = cv.warpPerspective(prev_frame, M, (w, h))

    return aligned_frame_next

def blend_frames(next_frame, prev_frame, mask):

    # find the upper left corner of the mask
    h, w = mask.shape
    min_x, min_y = h, w
    # Find the highest and leftmost point in the mask to create the double blended frame
    for i in range(h):
        for j in range(w):
            if mask[i, j] != 0:
                if i < min_x:
                    min_x = i
                if j < min_y:
                    min_y = j
    
    prev_frame_mask = prev_frame.copy()
    prev_frame_mask[min_x:, min_y:] = 0
    
    next_frame_double_blended = next_frame.copy()
    next_frame_double_blended[prev_frame_mask != 0] = prev_frame_mask[prev_frame_mask != 0]

    # Frame blending using entirely the previous frame
    next_frame[prev_frame != 0] = prev_frame[prev_frame != 0]

    return next_frame, next_frame_double_blended

def compute_magnitude(flow):
    magnitude, _ = cv.cartToPolar(flow[..., 0], flow[..., 1])
    return magnitude

def show_current_magnitude(magnitude):
    plt.imshow(magnitude)
    plt.show()

def remove_bg(image, fgbg):
    foreground_mask = fgbg.apply(image, learningRate=0.99)
    return foreground_mask

def auto_canny_median(image, sigma=0.33):
    # Compute the median of the pixel intensities
    median_val = np.median(image)
    
    # Set the lower and upper thresholds based on the median
    lower = int(max(0, (1.0 - sigma) * median_val))
    upper = int(min(255, (1.0 + sigma) * median_val))
    
    # Apply Canny Edge Detection
    edges = cv.Canny(image, lower, upper)
    return edges

def auto_canny(image, sigma=0.33):

    otsu_threshold, _ = cv.threshold(image, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

    lower = int((1-sigma) * otsu_threshold)
    upper = int((1+sigma) * otsu_threshold)

    edges = cv.Canny(image, lower, upper)
    return edges

def prepare_matching_samples(rgb_image, touch_image, depth_image, args):
    """
    Compute the similarity between the inpainted RGB image and the touch image
        - Perform high pass filter on the images
        - Compute the edges using Canny edge detector
        - Compute the similarity between the two images using SIFT descriptors
    """

    # Perform high pass filter on the images
    rgb_image = cv.cvtColor(rgb_image, cv.COLOR_BGR2GRAY)
    touch_image = cv.cvtColor(touch_image, cv.COLOR_BGR2GRAY)

    # cut the rgb image in the center
    # depth_image_crop = depth_image[200:, 300:]

    # Sharp the touch image using a high pass filter
    clahe = cv.createCLAHE(clipLimit=8, tileGridSize=(4,4))
    touch_image_enhanced = clahe.apply(touch_image)

    rgb_edges = auto_canny(rgb_image)
    touch_edges = auto_canny(touch_image_enhanced)

    save_samples(rgb_image, touch_image_enhanced, args)

def clean_touch_frame(frame, args):
    """
    Clean the touch frame by removing the background the black points inpainting them after
    identifying them using KNN segmentation
    """
    # Reshaping the image into a 2D array of pixels and 3 color values (RGB)
    pixel_vals = frame.reshape((-1,3))

    # Convert to float type
    pixel_vals = np.float32(pixel_vals)

    # kmeans search until 100 iterations are run or the required accuracy is epsilon = 0.85
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 0.85)
    k = 3
    retval, labels, centers = cv.kmeans(pixel_vals, k, None, criteria, 10, cv.KMEANS_RANDOM_CENTERS)

    # convert data into 8-bit values
    centers = np.uint8(centers)
    segmented_data = centers[labels.flatten()]

    # reshape data into the original image dimensions
    segmented_image = segmented_data.reshape((frame.shape))
    h, w, _ = segmented_image.shape
    mask = np.zeros_like(segmented_image, dtype=np.uint8)

    # find the black points
    darkest_centroid = np.min(centers,axis=0)
    mask[segmented_image == darkest_centroid] = 255
    mask = mask[:, :, 0]

    # Perform morphological operations to refine the mask
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE,(5,5))

    # Dilate the mask to ensure the dots are fully covered
    refined_mask = cv.dilate(mask, kernel, iterations=2)

    refined_output = cv.inpaint(frame, refined_mask, args["radius"], args["method"])

    return refined_output

def main(args):

    print("Press 'n' to go to the next frame, 's' to save the frame, press 'q' to quit")

    fgbg = cv.createBackgroundSubtractorMOG2(history=100, varThreshold=16, detectShadows=False)

    frames_dir_rgb = os.path.join(args['image'], 'video_frame')
    frames_files_rgb = os.listdir(frames_dir_rgb)
    frames_files_rgb = [os.path.join(frames_dir_rgb, frame) for frame in frames_files_rgb]

    frames_dir_touch = os.path.join(args['image'], 'gelsight_frame')
    frames_files_touch = os.listdir(frames_dir_touch)
    frames_files_touch = [os.path.join(frames_dir_touch, frame) for frame in frames_files_touch]

    start, step_size = find_best_step_parameters(frames_files_rgb, args)

    _, max_pressure_index = find_max_pressure(frames_files_touch, start, step_size)

    for i in range(start, len(frames_files_rgb)-step_size):

        prev_frame = cv.imread(frames_files_rgb[i])
        next_frame = cv.imread(frames_files_rgb[i+1])

        prev_after_motion = cv.imread(frames_files_rgb[i+step_size-5])
        after_motion = cv.imread(frames_files_rgb[i+step_size])
        after_motion_touch = cv.imread(frames_files_touch[i+step_size])

        motion_mask = create_motion_mask(prev_frame, next_frame, args)

        bg_mask = remove_bg(next_frame, fgbg)
        bitwise_mask = cv.bitwise_and(motion_mask, bg_mask)

        warped_frame = warp_frame_sift(prev_frame, after_motion)
        if warped_frame is None:
            _, max_pressure_index = find_max_pressure(frames_files_touch, i+1, step_size)
            continue

        inpainted_frame, inpainted_frame_double = blend_frames(after_motion, warped_frame, motion_mask)

        # cv.imshow('motion mask', motion_mask)
        cv.imshow('inpainted frame', inpainted_frame)

        prev_frame_1 = cv.cvtColor(prev_frame, cv.COLOR_BGR2RGB)
        after_motion_1 = cv.cvtColor(after_motion, cv.COLOR_BGR2RGB)
        motion_mask_1 = cv.cvtColor(motion_mask, cv.COLOR_BGR2RGB)
        inpainted_frame_1 = cv.cvtColor(inpainted_frame, cv.COLOR_BGR2RGB)

        if i % args['sampling_rate'] == 0:
            os.makedirs(args["save_dir"], exist_ok=True)
            cv.imwrite(f'{args["save_dir"]}/inpainted{i}-{i+step_size}.png', inpainted_frame)

        if i+step_size == max_pressure_index:
            print(f"Pressure frame reached: {max_pressure_index}")
            cv.imwrite(f'inpainted_max_pressure.png', inpainted_frame)
            after_motion_touch = clean_touch_frame(after_motion_touch, args)
            inpainted_frame = crop_rgb_frame(prev_after_motion, after_motion, inpainted_frame, args)
            inpainted_couple = find_coupled_frame(i, step_size, args)
            # inpainted_depth = compute_monocular_depth( "inpainted_max_pressure.png",inpainted_couple)
            prepare_matching_samples(inpainted_frame, after_motion_touch, None, args)
            _, max_pressure_index = find_max_pressure(frames_files_touch, i+1, step_size)

        # perform inpainting using OpenCV
        image = next_frame
        mask = motion_mask
        output = cv.inpaint(image, mask, args["radius"], args["method"])

        # cv.imshow("Output", output)

        # press 'n' to go to the next frame, press 's' to save the frame, press 'q' to quit
        if cv.waitKey(0) & 0xFF == ord('q'):
            shutil.rmtree(args["save_dir"])
            break
        
        if cv.waitKey(0) & 0xFF == ord('n'):
            continue

        if cv.waitKey(0) & 0xFF == ord('s'):
            cv.imwrite(f'inpainted{i}.png', inpainted_frame)
            plt.subplot(2, 2, 1)
            plt.axis('off')
            plt.imshow(prev_frame_1)
            plt.title('Previous Frame')
            plt.subplot(2, 2, 2)
            plt.axis('off')
            plt.imshow(after_motion_1)
            plt.title('After Motion Frame')
            plt.subplot(2, 2, 3)
            plt.axis('off')
            plt.imshow(motion_mask_1)
            plt.title('Motion Mask')
            plt.subplot(2, 2, 4)
            plt.axis('off')
            plt.imshow(inpainted_frame_1)
            plt.title('Inpainted Frame')
            plt.show()

    cv.destroyAllWindows()


if __name__ == "__main__":
    args = parse_args()
    main(args)