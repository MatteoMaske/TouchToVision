import cv2
import matplotlib.pyplot as plt
import numpy as np
import os

def find_matching_keypoints(img1, img2):
    # Detect keypoints and compute descriptors using SIFT (or ORB)
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    # Match descriptors using BFMatcher (Brute-Force Matcher) with default settings
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)

    # Apply ratio test to keep good matches (Lowe's ratio test)
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

    # Extract the matched keypoints' locations
    pts1 = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    pts2 = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    # Estimate the Fundamental Matrix using RANSAC
    F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC)

    # Draw matches - only inliers are used for this step
    img_inlier_matches = cv2.drawMatches(img1, kp1, img2, kp2, [good_matches[i] for i in range(len(good_matches)) if mask[i]], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    return pts1, pts2, img_inlier_matches, mask

def compute_disparity_map(img1, img2_warped, img_inlier_matches):
    # Stereo Matching to compute the disparity map
    # StereoSGBM parameters
    min_disp = 0
    num_disp = 16 * 3  # Must be divisible by 16
    block_size = 5  # The size of the block window. Must be odd
    P1 = 8 * 3 * block_size ** 2
    P2 = 32 * 3 * block_size ** 2
    disp12_max_diff = 1
    uniqueness_ratio = 15
    speckle_window_size = 20
    speckle_range = 2
    pre_filter_cap = 63

    stereo = cv2.StereoSGBM_create(minDisparity=min_disp,
                                numDisparities=num_disp,
                                blockSize=block_size,
                                P1=P1,
                                P2=P2,
                                disp12MaxDiff=disp12_max_diff,
                                uniquenessRatio=uniqueness_ratio,
                                speckleWindowSize=speckle_window_size,
                                speckleRange=speckle_range,
                                preFilterCap=pre_filter_cap)

    # Compute the disparity map
    disparity_map = stereo.compute(img1, img2_warped).astype(np.float32) / 16.0

    # Apply the WLS filter to the disparity map
    lmbda = 8000  # Regularization parameter
    sigma = 1.5   # Smoothness parameter
    right_matcher = cv2.ximgproc.createRightMatcher(stereo)  # Create the right matcher to improve the result
    disparity_map_right = right_matcher.compute(img2_warped, img1).astype(np.float32) / 16.0  # Right disparity map

    # Create WLS filter and apply
    wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=stereo)
    wls_filter.setLambda(lmbda)
    wls_filter.setSigmaColor(sigma)
    filtered_disparity_map = wls_filter.filter(disparity_map, img1, disparity_map_right=disparity_map_right)

    # Normalize the filtered disparity map for visualization
    filtered_disparity_map_normalized = cv2.normalize(filtered_disparity_map, None, 0, 255, cv2.NORM_MINMAX)
    filtered_disparity_map_normalized = np.uint8(filtered_disparity_map_normalized)

    return disparity_map, filtered_disparity_map_normalized

def display_results(img1, img2_warped, img_inlier_matches, disparity_map, filtered_disparity_map):
    plt.figure(figsize=(20, 10))

    plt.subplot(2, 3, 1)
    plt.title('Image 1')
    plt.imshow(img1, cmap='gray')

    plt.subplot(2, 3, 2)
    plt.title('Warped Image 2 (Aligned with Image 1)')
    plt.imshow(img2_warped, cmap='gray')

    plt.subplot(2, 3, 3)
    plt.title('Matched Keypoints (Inliers Only)')
    plt.imshow(img_inlier_matches)

    plt.subplot(2, 3, 4)
    plt.title('Raw Disparity Map')
    plt.imshow(disparity_map, cmap='plasma')

    plt.subplot(2, 3, 5)
    plt.title('Filtered Disparity Map')
    plt.imshow(filtered_disparity_map, cmap='plasma')

    plt.show()

def compute_monocular_depth(image1_path, image2_path):
    # Load the two images
    img1 = cv2.imread(image1_path, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(image2_path, cv2.IMREAD_GRAYSCALE)

    pts1, pts2, img_inliner_matches, mask = find_matching_keypoints(img1, img2)

    # We select only inlier points
    pts1 = pts1[mask.ravel() == 1]
    pts2 = pts2[mask.ravel() == 1]

    # 6. Find Homography using the inlier points
    H, status = cv2.findHomography(pts1, pts2, cv2.RANSAC)

    # 7. Warp the second image to align with the first image using the Homography matrix
    h, w = img1.shape
    img2_warped = cv2.warpPerspective(img2, H, (w, h))

    disparity_map, filtered_disparity_map = compute_disparity_map(img1, img2_warped, img_inliner_matches)

    display_results(img1, img2_warped, img_inliner_matches, disparity_map, filtered_disparity_map)

if __name__ == "__main__":
    image1_path = '../inpainted123.png'
    image2_path = '../inpainted149.png'

    compute_monocular_depth(image1_path, image2_path)