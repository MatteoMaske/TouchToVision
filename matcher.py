import os
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import random
import torch

from RealESRGAN import RealESRGAN
from PIL import Image

def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='Compare images between different classes and domains')
    parser.add_argument('-c', "--class", type=str, default='rock', choices=['grass', 'rock', 'rock2', 'tree'], help='Class name to compare against all other classes')
    parser.add_argument("--index", type=int, help='Pair index to compare against all other pairs')
    
    args = vars(parser.parse_args())

    return args

def find_matching_keypoints(visual_image, touch_image, show_keypoints=False):
    # Initialize SIFT detector
    sift = cv.SIFT_create()

    # Detect keypoints and compute descriptors for both images
    keypoints_1, descriptors_1 = sift.detectAndCompute(visual_image, None)
    keypoints_2, descriptors_2 = sift.detectAndCompute(touch_image, None)

    # Visualize the keypoints
    # image1_keypoints = cv.drawKeypoints(visual_image, keypoints_1, None)
    # image2_keypoints = cv.drawKeypoints(touch_image, keypoints_2, None)

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

    if show_keypoints:
        matched_image = cv.drawMatches(visual_image, keypoints_1, touch_image, keypoints_2, good_matches, None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        plt.figure(figsize=(12, 6))
        plt.imshow(matched_image)
        plt.title('Top 50 Matches')
        plt.show()

    return keypoints_1, keypoints_2, good_matches


def compute_similarity_metric(keypoints_1, keypoints_2, good_matches):

    #Basic metrics to compute similarity
    # return 100 * len(good_matches) / min(len(keypoints_1), len(keypoints_2))
    # 1. Number of good matches
    num_good_matches = len(good_matches)
    
    # 2. Average distance of matches
    if num_good_matches == 0:
        return 0  # If no matches, similarity is zero
    avg_distance = np.mean([m.distance for m in good_matches])

    # 3. Ratio of keypoints matched (normalized by the number of keypoints in both images)
    ratio_matched_1 = num_good_matches / len(keypoints_1)
    ratio_matched_2 = num_good_matches / len(keypoints_2)
    
    # Final similarity score: A combination of the different metrics
    similarity_score = (num_good_matches * 10 / avg_distance) + (ratio_matched_1 + ratio_matched_2) * 10

    return similarity_score

def compare_images(visual_image, touch_image, show_keypoints=False):

    # Find matching keypoints between the images
    keypoints_1, keypoints_2, good_matches = find_matching_keypoints(visual_image, touch_image, show_keypoints)

    # Compute the similarity metric
    similarity_score = compute_similarity_metric(keypoints_1, keypoints_2, good_matches)

    return similarity_score

def super_resolve_image(filename, model):
    result_image_path = filename.replace('.png', '_hd.png')

    image = Image.open(filename).convert('RGB')
    sr_image = model.predict(np.array(image))
    sr_image.save(result_image_path)

def generate_crops(image):
    height, width = image.shape
    offset = width // 10
    crop_size = width // 4
    crops = []

    for i in range(offset, image.shape[0]-crop_size, offset):
        for j in range(offset, image.shape[1]-crop_size-offset, offset):
            crop = image[i:i+crop_size, j:j+crop_size]
            crops.append(crop)
            # tmp = image.copy()
            # cv.rectangle(tmp, (j, i), (j+crop_size, i+crop_size), (0, 0, 0), 3)
            # cv.imshow('Crops', tmp)
            # cv.waitKey(0)
    # print(f"{len(crops)} crops generated")
    return crops

def compute_best_matches(visual_crops, touch_image, k=5):
    similarities = []
    for visual_crop in visual_crops:
        similarity_score = compare_images(visual_crop, touch_image, show_keypoints=False)
        similarities.append(similarity_score)
    #return top-3 similarities
    assert k <= len(similarities), "k should be less than or equal to the number of similarities"
    top_k_similarities = sorted(similarities, reverse=True)[:k]
    return top_k_similarities

def compare_classes(args, sr_model):
    # Define the base directory for the dataset
    base_dir = 'data'
    video_dir = 'video_frame'
    gelsight_dir = 'gelsight_frame'

    # Define the classes and pairs to compare
    _class = args['class']
    class_samples = len(os.listdir(os.path.join(base_dir, _class, video_dir)))
    class_samples //= 2  # Divide by 2 to get the number of pairs

    index = args['index']
    if index is None:
        index = random.randint(0, class_samples - 1)
    print(f'Comparing class {_class} with index {index}')

    if not os.path.exists(os.path.join(base_dir, _class, video_dir,  f'rgb_image_{index:03d}_hd.png')):
        super_resolve_image(os.path.join(base_dir, _class, video_dir, f'rgb_image_{index:03d}.png'), sr_model)
    visual_sample = os.path.join(base_dir, _class, video_dir,  f'rgb_image_{index:03d}_hd.png')
    touch_sample = os.path.join(base_dir, _class, gelsight_dir, f'touch_image_{index:03d}.png')

    visual_image = cv.imread(visual_sample, cv.IMREAD_GRAYSCALE)
    touch_image = cv.imread(touch_sample, cv.IMREAD_GRAYSCALE)

    target_classes = ["rock", "grass", "tree", "rock2"]
    target_classes.remove(_class)

    # Source Visual vs Target Touch
    visual_crops = generate_crops(visual_image)
    for target_class in target_classes:
        target_samples = len(os.listdir(os.path.join(base_dir, target_class, gelsight_dir)))
        target_results = []
        for target_index in range(target_samples):
            target_touch_sample = os.path.join(base_dir, target_class, gelsight_dir, f'touch_image_{target_index:03d}.png')
            target_touch_image = cv.imread(target_touch_sample, cv.IMREAD_GRAYSCALE)
            best_matches = compute_best_matches(visual_crops, target_touch_image)
            best_matches_avg = sum(best_matches) / len(best_matches)
            # best_matches_avg = compare_images(visual_image, target_touch_image, show_keypoints=False)
            target_results.append(best_matches_avg)
            # print(f"Best matches for Visual:{_class} vs Touch:{target_class}_{target_index}: {best_matches_avg}")
        results_mean, results_std = np.mean(target_results), np.std(target_results)
        print(f"Best matches for touch:{_class} vs visual:{target_class}: {results_mean:.5f} +/- {results_std:.5f}")
    
    best_matches = compute_best_matches(visual_crops, touch_image)
    best_matches_avg = sum(best_matches) / len(best_matches)
    # best_matches_avg = compare_images(visual_image, touch_image, show_keypoints=False)
    print(f"Best matches for {_class} vs {_class}: {best_matches_avg}")

    # Source Touch vs Target Visual
    for target_class in target_classes:
        target_samples = len(os.listdir(os.path.join(base_dir, target_class, video_dir)))//2
        target_results = []
        for target_index in range(target_samples):
            if not os.path.exists(os.path.join(base_dir, target_class, video_dir,  f'rgb_image_{target_index:03d}_hd.png')):
                super_resolve_image(os.path.join(base_dir, target_class, video_dir, f'rgb_image_{target_index:03d}.png'), sr_model)

            target_visual_sample = os.path.join(base_dir, target_class, video_dir, f'rgb_image_{target_index:03d}_hd.png')
            target_visual_image = cv.imread(target_visual_sample, cv.IMREAD_GRAYSCALE)
            target_visual_samples = generate_crops(target_visual_image)

            best_matches = compute_best_matches(target_visual_samples, touch_image)
            best_matches_avg = sum(best_matches) / len(best_matches)
            target_results.append(best_matches_avg)
        results_mean, results_std = np.mean(target_results), np.std(target_results)
        print(f"Best matches for touch:{_class} vs visual:{target_class}: {results_mean:.5f} +/- {results_std:.5f}")
    
    visual_samples = generate_crops(visual_image)
    best_matches = compute_best_matches(visual_samples, touch_image)
    best_matches_avg = sum(best_matches) / len(best_matches)
    print(f"Best matches for touch:{_class} vs visual:{_class}: {best_matches_avg}")

def intialize_superesolution_model():

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('device:', device)

    model_scale = "2"

    model = RealESRGAN(device, scale=int(model_scale))
    model.load_weights(f'weights/RealESRGAN_x{model_scale}.pth')

    return model
    

if __name__ == '__main__':
    # Example usage
    args=parse_args()
    model = intialize_superesolution_model()
    compare_classes(args, model)