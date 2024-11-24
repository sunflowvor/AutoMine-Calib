import cv2
import numpy as np

def find_keypoints_and_matches(img1, img2):
    # 使用 SIFT 特征检测器
    sift = cv2.SIFT_create()

    # 检测和计算特征点
    keypoints1, descriptors1 = sift.detectAndCompute(img1, None)
    keypoints2, descriptors2 = sift.detectAndCompute(img2, None)

    # 使用 FLANN 特征匹配器
    index_params = dict(algorithm=1, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(descriptors1, descriptors2, k=2)

    # Lowe's ratio test to filter matches
    good_matches = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)

    return keypoints1, keypoints2, good_matches

def calculate_extrinsic_matrix(img1, img2, K1, K2):
    # 查找关键点和匹配
    keypoints1, keypoints2, good_matches = find_keypoints_and_matches(img1, img2)

    # 获取匹配点坐标
    points1 = np.float32([keypoints1[m.queryIdx].pt for m in good_matches])
    points2 = np.float32([keypoints2[m.trainIdx].pt for m in good_matches])

    # 计算基础矩阵
    F, mask = cv2.findFundamentalMat(points1, points2, cv2.FM_RANSAC)

    # 计算本质矩阵
    E = K2.T @ F @ K1

    # 分解本质矩阵以获得相对姿态（旋转和平移）

    _, R, t, mask = cv2.recoverPose(E, points1, points2, K1)

    return R, t, keypoints1, keypoints2, good_matches, points1, points2

def draw_matches(img1, img2, keypoints1, keypoints2, matches):
    matched_img = cv2.drawMatches(img1, keypoints1, img2, keypoints2, matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    return matched_img
