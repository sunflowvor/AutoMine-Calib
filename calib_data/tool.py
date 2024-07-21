import os
import shutil

base_path = "/Volumes/My Passport/scene_01/1000_select_frames/OCT-02/scene_01"
number_scene = 19

for i in range(number_scene):
    key_left_img_path = os.path.join(base_path, "point", str(i).zfill(6) + ".pcd")
    target_path = os.path.join("./", "folder_" + str(i).zfill(2), "point", "000000.pcd")
    shutil.copy(key_left_img_path, target_path)

    for j in range(1, 10):
        key_left_img_path = os.path.join(base_path, "clouds", str(i), str(i).zfill(6) + "_" + str(j) + ".pcd")
        target_path = os.path.join("./", "folder_" + str(i).zfill(2), "point", str(j).zfill(6) + ".pcd")
        shutil.copy(key_left_img_path, target_path)