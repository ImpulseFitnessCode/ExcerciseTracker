import tensorflow.compat.v1 as tf
import posenet
import cv2
import keyboard


class Tracker():
    def capture_pose_gen(self):
        model = posenet.load_model_keras(101, model_dir='./converted_models/saved_model')
        cap = cv2.VideoCapture(0)
        cap.set(3, 1280)
        cap.set(4, 720)
        cv2.namedWindow('posenet', cv2.WINDOW_NORMAL)
        while True:
            input_image, display_image, output_scale = posenet.read_cap(
                cap, scale_factor=0.7)

            output = model(tf.convert_to_tensor(input_image))

            pose_scores, keypoint_scores, keypoint_coords = posenet.decode_multi.decode_multiple_poses(
                output['heatmap'].numpy().squeeze(axis=0),
                output['offset_2'].numpy().squeeze(axis=0),
                output['displacement_fwd_2'].numpy().squeeze(axis=0),
                output['displacement_bwd_2'].numpy().squeeze(axis=0),
                max_pose_detections=10,
                output_stride=16,
                min_pose_score=0.15)

            keypoint_coords *= output_scale
            overlay_image = posenet.draw_skel_and_kp(
                display_image, pose_scores, keypoint_scores, keypoint_coords,
                min_pose_score=0.15, min_part_score=0.1)
            cv2.imshow('posenet', overlay_image)
            key = cv2.waitKey(100) & 0xFF
            if key == 32:
                print('space pressed')
            yield keypoint_coords

