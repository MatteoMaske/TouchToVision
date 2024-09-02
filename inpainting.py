import argparse
import cv2 as cv
import os
import numpy as np
import matplotlib.pyplot as plt

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
    for i in range(0, len(frames_files_rgb)-best_step_size, best_step_size):
        prev_frame = cv.imread(frames_files_rgb[i])
        next_frame = cv.imread(frames_files_rgb[i+best_step_size])
        H = compute_homography(prev_frame, next_frame)
        if H is None:
            continue
        tx, ty = H[0, 2], H[1, 2]
        magnitude = np.sqrt(tx**2 + ty**2)
        if magnitude > max_magnitude:
            max_magnitude = magnitude
            best_start = i
            
    print(f"Current best start: {best_start}, magnitude: {max_magnitude}")

    # best settings found
    prev_frame = cv.imread(frames_files_rgb[best_start])
    next_frame = cv.imread(frames_files_rgb[best_start+best_step_size])
    warped_frame = warp_frame_sift(prev_frame, next_frame)
    cv.arrowedLine(warped_frame, (256,256) , (int(tx)+256, int(ty)+256), (0, 255, 0), 1)
    cv.putText(warped_frame, f"tx:{tx:.2f}, ty:{ty:.2f}, mag:{magnitude:.2f}, frame:{i}", (0, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv.imshow('warped frame', warped_frame)

    return best_start, best_step_size

def segment_frame(frame):
    """
        Segments the frame to separate the background from the foreground using KNN segmentation
    """

    # Convert to grayscale
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    # Find Canny edges 
    _, thresholded = cv.threshold(gray, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

    # # Apply a binary threshold to ensure it's binary (if not already)
    # _, binary_image = cv.threshold(gray, 128, 255, cv.THRESH_BINARY)

    # Find contours
    contours, _ = cv.findContours(thresholded, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    # Draw all contours on a copy of the original image for visualization
    contoured_image = frame.copy()
    cv.drawContours(contoured_image, contours, -1, (0, 255, 0), 2)  # Draw all contours in green

    # Identify and highlight the largest contour (assuming it's the hand)
    largest_contour = max(contours, key=cv.contourArea)
    cv.drawContours(contoured_image, [largest_contour], -1, (255, 0, 0), 2)  # Draw the largest contour in blue

    # Display the images
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.title('Binary Image')
    plt.imshow(thresholded, cmap='gray')
    plt.subplot(1, 2, 2)
    plt.title('Contours on Image')
    plt.imshow(cv.cvtColor(contoured_image, cv.COLOR_BGR2RGB))
    plt.show()

    # Save the contoured image for review
    output_contour_image_path = 'contoured_image.png'
    cv.imwrite(output_contour_image_path, contoured_image)

    return contoured_image

def compute_optical_flow(prev_frame, next_frame):
    prev_gray = cv.cvtColor(prev_frame, cv.COLOR_BGR2GRAY)
    next_gray = cv.cvtColor(next_frame, cv.COLOR_BGR2GRAY)
    
    flow = cv.calcOpticalFlowFarneback(prev_gray, next_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    return flow

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

def create_motion_mask(magnitude, k):
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
    kernel = np.ones((3,3),np.uint8)
    # cleaned_mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel)
    # cleaned_mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel)
    cleaned_mask = mask

    #remove all the white pixels not in the bottom right corner
    h, w = cleaned_mask.shape
    cleaned_mask[:h//2, :] = 0
    cleaned_mask[:, :w//2] = 0

    return cleaned_mask

def show_current_magnitude(magnitude):
    plt.imshow(magnitude)
    plt.show()

def remove_bg(image, fgbg):
    foreground_mask = fgbg.apply(image, learningRate=0.99)
    return foreground_mask

def main(args):

    print("Press 'n' to go to the next frame, press 'p' to show the magnitude, press 'q' to quit")

    fgbg = cv.createBackgroundSubtractorMOG2(history=100, varThreshold=16, detectShadows=False)

    frames_dir_rgb = os.path.join(args['image'], 'video_frame')
    frames_files_rgb = os.listdir(frames_dir_rgb)
    frames_files_rgb = [os.path.join(frames_dir_rgb, frame) for frame in frames_files_rgb]

    frames_dir_touch = os.path.join(args['image'], 'gelsight_frame')
    frames_files_touch = os.listdir(frames_dir_touch)
    frames_files_touch = [os.path.join(frames_dir_touch, frame) for frame in frames_files_touch]

    start, step_size = find_best_step_parameters(frames_files_rgb, args)

    for i in range(start, len(frames_files_rgb)):
        # RGB frames processing ========================================

        prev_frame = cv.imread(frames_files_rgb[i])
        next_frame = cv.imread(frames_files_rgb[i+1])
        after_motion = cv.imread(frames_files_rgb[i+step_size])
        cv.imshow('prev frame', prev_frame)
        cv.imshow('after motion', after_motion)

        flow = compute_optical_flow(prev_frame, next_frame)
        magnitude = compute_magnitude(flow)

        # seg_frame = segment_frame(prev_frame)

        warped_frame = warp_frame_sift(prev_frame, after_motion)

        motion_mask = create_motion_mask(magnitude, k=args["scaling_factor"])
        cleaned_motion_mask = clean_mask(motion_mask)

        bg_mask = remove_bg(next_frame, fgbg)
        bitwise_mask = cv.bitwise_and(cleaned_motion_mask, bg_mask)

        inpainted_frame, inpainted_frame_double = blend_frames(after_motion, warped_frame, cleaned_motion_mask)

        cv.imshow('inpainted frame', inpainted_frame)
        # cv.imshow('inpainted frame double', inpainted_frame_double)
        cv.imshow('clean motion mask', cleaned_motion_mask)
        # cv.imshow('bitwise mask', bitwise_mask)

        # perform inpainting using OpenCV
        image = next_frame
        mask = cleaned_motion_mask
        output = cv.inpaint(image, mask, args["radius"], args["method"])

        # cv.imshow("Output", output)

        # Touch frames processing ========================================
        frame_touch = cv.imread(frames_files_touch[i])
        cv.imshow('frame touch', frame_touch)
        
        min_pressure = np.inf
        min_pressure_idx = 0
        if np.sum(frame_touch) < min_pressure:
            min_pressure = np.sum(frame_touch)
            min_pressure_idx = i
        print(f"Min pressure: {min_pressure}, at frame {min_pressure_idx}")

        # press 'n' to go to the next frame, press 'q' to quit
        if cv.waitKey(0) & 0xFF == ord('q'):
            break
        
        if cv.waitKey(0) & 0xFF == ord('n'):
            continue
        
        if cv.waitKey(0) & 0xFF == ord('p'):
            show_current_magnitude(magnitude)

    cv.destroyAllWindows()


if __name__ == "__main__":
    args = parse_args()
    main(args)