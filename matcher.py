import os
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

def find_matching_keypoints(visual_image, touch_image):
    # Initialize SIFT detector
    sift = cv.SIFT_create()

    # Detect keypoints and compute descriptors for both images
    keypoints_1, descriptors_1 = sift.detectAndCompute(visual_image, None)
    keypoints_2, descriptors_2 = sift.detectAndCompute(touch_image, None)

    # Visualize the keypoints
    image1_keypoints = cv.drawKeypoints(visual_image, keypoints_1, None)
    image2_keypoints = cv.drawKeypoints(touch_image, keypoints_2, None)

    # FLANN parameters
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)  # Specify the number of checks

    # Initialize the FLANN-based matcher
    flann = cv.FlannBasedMatcher(index_params, search_params)

    # Perform KNN matching (find the 2 nearest neighbors)
    matches = flann.knnMatch(descriptors_1, descriptors_2, k=2)

    # Apply Lowe's ratio test to filter good matches
    good_matches = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:  # Lowe's ratio test
            good_matches.append(m)

    # Draw the matches
    matched_image = cv.drawMatches(visual_image, keypoints_1, touch_image, keypoints_2, good_matches, None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    plt.figure(figsize=(12, 6))
    plt.imshow(matched_image)
    plt.title('Top 50 Matches')
    plt.show()

    return keypoints_1, keypoints_2, good_matches


def compute_similarity_metric(keypoints_1, keypoints_2, good_matches):
    # 1. Number of good matches
    num_good_matches = len(good_matches)
    
    # 2. Average distance of matches
    if num_good_matches == 0:
        return 0  # If no matches, similarity is zero
    avg_distance = np.mean([m.distance for m in good_matches])

    # 3. Ratio of keypoints matched (normalized by the number of keypoints in both images)
    ratio_matched_1 = num_good_matches / len(keypoints_1)
    ratio_matched_2 = num_good_matches / len(keypoints_2)

    # 4. Geometric consistency: Calculate homography between the matched points
    if num_good_matches >= 4:  # Need at least 4 matches for homography
        src_pts = np.float32([keypoints_1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([keypoints_2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        H, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)
        # Use the number of inliers as a measure of geometric consistency
        num_inliers = np.sum(mask)
    else:
        num_inliers = 0
    
    # Final similarity score: A combination of the different metrics
    similarity_score = (num_good_matches * 10 / avg_distance) + (ratio_matched_1 + ratio_matched_2) * 10 + num_inliers

    return similarity_score

def compare_images(visual_image_path, touch_image_path):
    # Load the images
    visual_image = cv.imread(visual_image_path, cv.IMREAD_GRAYSCALE)
    touch_image = cv.imread(touch_image_path, cv.IMREAD_GRAYSCALE)

    # Find matching keypoints between the images
    keypoints_1, keypoints_2, good_matches = find_matching_keypoints(visual_image, touch_image)

    # Compute the similarity metric
    similarity_score = compute_similarity_metric(keypoints_1, keypoints_2, good_matches)

    return similarity_score
