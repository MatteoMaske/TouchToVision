import argparse
import cv2 as cv
import os
import numpy as np
import matplotlib.pyplot as plt

def parse_args():
	
    ap = argparse.ArgumentParser()
    ap.add_argument("-s", "--starting_frame", type=int, default=140)
    ap.add_argument("--step_size", type=int, default=50)
    ap.add_argument("-i", "--image", type=str, default="rock",
                    choices=['grass', 'rock'],
                    help="frames to inpaint")
    ap.add_argument("-a", "--method", type=str, default="telea",
        choices=["telea", "ns"],
        help="inpainting algorithm to use")
    ap.add_argument("-r", "--radius", type=int, default=3, help="inpainting radius")
    ap.add_argument("-k", "--scaling_factor", type=float, default=2.0, help="scaling factor for motion mask")
    
    args = vars(ap.parse_args())

    args['image'] = '20220318_020426_grass' if args['image'] == 'grass' else '20220319_002026_rock'
    args['method'] = cv.INPAINT_TELEA if args['method'] == 'telea' else cv.INPAINT_NS

    return args

def compute_optical_flow(prev_frame, next_frame):
    prev_gray = cv.cvtColor(prev_frame, cv.COLOR_BGR2GRAY)
    next_gray = cv.cvtColor(next_frame, cv.COLOR_BGR2GRAY)
    
    flow = cv.calcOpticalFlowFarneback(prev_gray, next_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    return flow

def warp_frame_of(prev_frame, flow):
    h, w = flow.shape[:2]
    grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))

    flow_map = np.float32(np.stack((grid_x + flow[..., 0], grid_y + flow[..., 1]), axis=-1))
    warped_frame = cv.remap(prev_frame, flow_map, None, cv.INTER_LINEAR)
    return warped_frame

def warp_frame_sift(prev_frame, current_frame):
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

    # Find homography matrix
    M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)

    # Warp the previous frame to align with the current frame
    h, w = current_frame.shape[:2]
    aligned_frame_next = cv.warpPerspective(prev_frame, M, (w, h))

    return aligned_frame_next

def blend_frames(next_frame, prev_frame, mask):

    # find the upper left corner of the mask
    h, w = mask.shape
    min_x, min_y = h, w
    # Find the highest and leftmost point in the mask
    for i in range(h):
        for j in range(w):
            if mask[i, j] != 0:
                if i < min_x:
                    min_x = i
                if j < min_y:
                    min_y = j
    cv.imshow('prev frame pre', prev_frame)
    prev_frame[min_x:, min_y:] = 0
    cv.imshow('prev frame post', prev_frame)

    next_frame[prev_frame != 0] = prev_frame[prev_frame != 0]
    return next_frame

def compute_magnitude(flow):
    magnitude, _ = cv.cartToPolar(flow[..., 0], flow[..., 1])
    return magnitude

def create_motion_mask(magnitude, k):
    # Threshold the magnitude to create a binary mask
    mean_magnitude = np.mean(magnitude)
    std_magnitude = np.std(magnitude)

    # Set threshold to mean + k * std, k is a hyperparameter
    threshold = mean_magnitude + k * std_magnitude
    print(f"Threshold: {threshold}")

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
    foreground_mask = fgbg.apply(image)
    return foreground_mask


def main():
    args = parse_args()

    print("Press 'n' to go to the next frame, press 'p' to show the magnitude, press 'q' to quit")

    fgbg = cv.createBackgroundSubtractorMOG2(history=100, detectShadows=False)

    frames_dir = os.path.join(args['image'], 'video_frame')
    frames_files = os.listdir(frames_dir)
    frames_files = [os.path.join(frames_dir, frame) for frame in frames_files]

    for i in range(args['starting_frame'], len(frames_files)-args['step_size']):
        prev_frame = cv.imread(frames_files[i])
        next_frame = cv.imread(frames_files[i+1])
        after_motion = cv.imread(frames_files[i+args['step_size']])
        # cv.imshow('prev frame', prev_frame)
        # cv.imshow('next frame', next_frame)
        # cv.imshow('after motion', after_motion)

        flow = compute_optical_flow(prev_frame, next_frame)
        magnitude = compute_magnitude(flow)

        warped_frame = warp_frame_sift(prev_frame, after_motion)
        # cv.imshow('warped frame', warped_frame)

        motion_mask = create_motion_mask(magnitude, k=args["scaling_factor"])
        cleaned_motion_mask = clean_mask(motion_mask)

        bg_mask = remove_bg(next_frame, fgbg)
        bitwise_mask = cv.bitwise_and(cleaned_motion_mask, bg_mask)

        inpainted_frame = blend_frames(after_motion, warped_frame, cleaned_motion_mask)
        cv.imshow('inpainted frame', inpainted_frame)

        cv.imshow('clean motion mask', cleaned_motion_mask)
        # cv.imshow('bg mask', bg_mask)
        # cv.imshow('bitwise mask', bitwise_mask)

        # perform inpainting using OpenCV
        image = next_frame
        mask = cleaned_motion_mask
        output = cv.inpaint(image, mask, args["radius"], args["method"] )

        # cv.imshow("Output", output)

        # press 'n' to go to the next frame, press 'q' to quit
        if cv.waitKey(0) & 0xFF == ord('q'):
            break
        
        if cv.waitKey(0) & 0xFF == ord('n'):
            continue
        
        if cv.waitKey(0) & 0xFF == ord('p'):
            show_current_magnitude(magnitude)

    cv.destroyAllWindows()


if __name__ == "__main__":
    main()