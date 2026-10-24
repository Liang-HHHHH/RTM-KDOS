import os
import cv2
import numpy as np
import torch
import mmcv
import mmengine
from mmengine.registry import init_default_scope
from mmpose.apis import inference_topdown
from mmpose.apis import init_model as init_pose_estimator
from mmpose.evaluation.functional import nms
from mmdet.apis import inference_detector, init_detector
import time
from tqdm import tqdm
import random


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)


# input video file
video_path = r"F:\captures\calib_record_20260508_164002.mp4"
# output file
output_dir = r'E:\out'
os.makedirs(output_dir, exist_ok=True)

is_camera = 0  # 0: video, 1: RealSense相机
is_save = 0  # 0: don't save resule, 1: save result
is_all = 0  # 0: predict one file , 1: predict all file under the floder

camera_width = 640
camera_height = 480
camera_fps = 30


detector = init_detector(
    r"D:\Python\mm\dara_process\version_4\rtmdet_config.py",
    r"D:\Python\mm\dara_process\version_4\rtmdet.pth",
    device=device
)
pose_estimator = init_pose_estimator(
    r'D:\Python\mm\mmpose\work_dirs\rtmpose-score\pose_score\rtmpose-score.py',
    r'D:\Python\mm\mmpose\work_dirs\rtmpose-score\pose_score\rtmpose-score.pth',
    device=device,
    cfg_options={'model': {'test_cfg': {'output_heatmaps': False}}}
)


if not torch.cuda.is_available():
    from mmengine.model.utils import revert_sync_batchnorm

    pose_estimator = revert_sync_batchnorm(pose_estimator)


init_default_scope(detector.cfg.get('default_scope', 'mmdet'))


dataset_info = {
    'keypoint_info': {
        0: {'name': 'kp1', 'id': 0, 'color': [255, 0, 0], 'type': '', 'swap': ''},
        1: {'name': 'kp2', 'id': 1, 'color': [0, 255, 0], 'type': '', 'swap': ''},
        2: {'name': 'kp3', 'id': 2, 'color': [0, 0, 255], 'type': '', 'swap': ''}
    },
    'skeleton_info': {
        0: {'link': ('kp1', 'kp2'), 'id': 0, 'color': [100, 150, 200]},
        1: {'link': ('kp2', 'kp3'), 'id': 1, 'color': [200, 100, 150]},
    }
}
label2id = {info['name']: kid for kid, info in dataset_info['keypoint_info'].items()}


TARGET_COLORS = [(0, 255, 0), (0, 0, 255), (255, 0, 0), (255, 255, 0), (0, 255, 255), (255, 0, 255)]
bbox_colors = {i: color for i, color in enumerate(TARGET_COLORS)}
bbox_thickness = 2
kpt_radius = 5
skeleton_thickness = 2
CONF_THRES = 0.35
text_font = cv2.FONT_HERSHEY_SIMPLEX
text_scale = 0.5
text_thickness = 1
text_color = (255, 255, 255)
text_bg_color = (0, 0, 0)

IS_SHOW = True


PLOT_WIDTH = 400  
PLOT_WINDOW = 200 


def draw_score_curve(height, width, score_history, window_size=200):

    plot_img = np.zeros((height, width, 3), dtype=np.uint8)
    plot_img[:] = (30, 30, 30)


    margin_top = 60  
    margin_bottom = 20
    plot_h = height - margin_top - margin_bottom

    cv2.line(plot_img, (0, margin_top), (width, margin_top), (100, 100, 100), 1)
    cv2.putText(plot_img, "1.0", (5, margin_top - 5), text_font, 0.4, (150, 150, 150), 1)

    mid_y = int(margin_top + plot_h / 2)
    cv2.line(plot_img, (0, mid_y), (width, mid_y), (60, 60, 60), 1)
    cv2.putText(plot_img, "0.5", (5, mid_y - 5), text_font, 0.4, (100, 100, 100), 1)
    
    bot_y = margin_top + plot_h
    cv2.line(plot_img, (0, bot_y), (width, bot_y), (100, 100, 100), 1)
    cv2.putText(plot_img, "0.0", (5, bot_y - 5), text_font, 0.4, (150, 150, 150), 1)

    if not score_history:
        return plot_img


    data = score_history[-window_size:]


    step_x = width / window_size


    max_targets = 0
    for frame_data in data:
        if isinstance(frame_data, list):
            max_targets = max(max_targets, len(frame_data))


    for target_idx in range(max_targets):
        if target_idx >= len(TARGET_COLORS):
            break 

        color = TARGET_COLORS[target_idx]
        prev_point = None

        for i, frame_data in enumerate(data):

            if isinstance(frame_data, list) and len(frame_data) > target_idx:
                score = frame_data[target_idx]
            else:
                score = 0


            if score <= 0:
                prev_point = None
                continue

            x = int(i * step_x)

            y = int(bot_y - (score * plot_h))
            curr_point = (x, y)

            cv2.circle(plot_img, curr_point, 2, color, -1)

            if prev_point is not None:
                cv2.line(plot_img, prev_point, curr_point, color, 1, cv2.LINE_AA)

            prev_point = curr_point

    current_scores = score_history[-1]
    if isinstance(current_scores, list):

        for target_idx, score in enumerate(current_scores):
            if target_idx >= len(TARGET_COLORS):
                break
            color = TARGET_COLORS[target_idx]
            cv2.putText(plot_img, f"Target {target_idx+1}: Score {score:.3f}", 
                        (10, 20 + target_idx * 20),
                        text_font, 0.6, color, 1)
    else:

        title_color = (0, 255, 0) if current_scores > 0 else (100, 100, 100)
        cv2.putText(plot_img, f"Current Score: {current_scores:.3f}", (10, 25),
                    text_font, 0.7, title_color, 2)

    return plot_img


def process_frame(img, last_valid_state=None):

    init_default_scope(detector.cfg.get('default_scope', 'mmdet'))

    detect_result = inference_detector(detector, img)
    pred_instance = detect_result.pred_instances.cpu()
    bboxes = np.concatenate((pred_instance.bboxes, pred_instance.scores[:, None]), axis=1)

    valid_bboxes = []
    valid_classes = []
    valid_scores = []

    for class_id in [0, 1]:
        class_mask = (pred_instance.labels == class_id) & (pred_instance.scores > CONF_THRES)
        class_bboxes = bboxes[class_mask]

        if len(class_bboxes) > 0:
            keep_indices = nms(class_bboxes, 0.2)
            kept_results = class_bboxes[keep_indices]

            for row in kept_results:
                valid_bboxes.append(row[:4].astype('int'))
                valid_classes.append(class_id)
                valid_scores.append(row[4])


    if len(valid_bboxes) > 0:
        bboxes_array = np.array(valid_bboxes)

        pose_results = inference_topdown(pose_estimator, img, bboxes_array)

        point_scores = []
        keypoints_all = []
        keypoint_scores_all = []

        for result in pose_results:
            kpts = result.pred_instances.keypoints[0].astype(int)
            kpt_scores = result.pred_instances.keypoint_scores[0]
            if hasattr(result.pred_instances, 'point_score'):
                p_score = result.pred_instances.point_score[0]
            else:
                p_score = 0.0

            keypoints_all.append(kpts)
            keypoint_scores_all.append(kpt_scores)
            point_scores.append(p_score)


        for idx, bbox in enumerate(valid_bboxes):
            class_id = valid_classes[idx]
            current_point_score = point_scores[idx] if idx < len(point_scores) else 0.0


            color = TARGET_COLORS[idx % len(TARGET_COLORS)]
            

            cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, bbox_thickness)


            label_x = max(bbox[0], 0)
            label_y = max(bbox[1] - 5, 0)

            point_score_text = f"Target {idx+1}: Score:{float(current_point_score):.2f}"
            (tw2, th2), _ = cv2.getTextSize(point_score_text, text_font, text_scale, text_thickness)
            cv2.rectangle(img, (label_x, label_y - th2 - 5), (label_x + tw2, label_y + 5), text_bg_color, -1)
            cv2.putText(img, point_score_text, (label_x, label_y), text_font, text_scale, text_color,
                        text_thickness, cv2.LINE_AA)


            if idx < len(keypoints_all):
                kpts = keypoints_all[idx]
                for skeleton in dataset_info['skeleton_info'].values():
                    u, v = skeleton['link']
                    if u in label2id and v in label2id:
                        u_id, v_id = label2id[u], label2id[v]
                        cv2.line(img, tuple(kpts[u_id]), tuple(kpts[v_id]), skeleton['color'], skeleton_thickness)
                for kpt_idx, kpt in enumerate(kpts):
                    k_color = dataset_info['keypoint_info'][kpt_idx]['color']
                    cv2.circle(img, tuple(kpt), kpt_radius, k_color, -1)


        current_state = {
            'bboxes': valid_bboxes,
            'classes': valid_classes,
            'point_scores': point_scores,
            'keypoints': keypoints_all
        }
        return img, point_scores, current_state


    elif last_valid_state is not None:

        return img, [], None


    else:
        return img, [], None


def init_realsense_camera(width, height, fps):

    pipeline = rs.pipeline()
    config = rs.config()
    

    config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
    
    try:

        pipeline.start(config)
        profile = pipeline.get_active_profile()
        color_profile = rs.video_stream_profile(profile.get_stream(rs.stream.color))
        color_intrinsics = color_profile.get_intrinsics()
        
        return pipeline
    except Exception as e:
        return None

def process_single_video(video_file, output_dir, is_save):

    import datetime
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    cap = cv2.VideoCapture(video_file)
    if not cap.isOpened():
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    base_name = os.path.basename(video_file)
    name_without_ext = os.path.splitext(base_name)[0]
    ext = os.path.splitext(base_name)[1]
    file_name = f"{name_without_ext}_{timestamp}{ext}"
    

    if is_save:
        save_path = os.path.join(output_dir, f"result_{file_name}")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = None

    else:
        save_path = os.path.join(output_dir, f"result_{file_name}")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = None



    recording = is_save 

    score_history = []  
    last_valid_state = None 


    pbar = tqdm(total=total_frames, unit='frame')

    start_time = time.time()
    frame_count = 0

    try:
        while True:

            ret, frame = cap.read()
            if not ret:
                break

            try:
     
                frame_copy = frame.copy()
             
                processed_frame, current_scores, last_valid_state = process_frame(frame_copy, last_valid_state)

              
                score_history.append(current_scores)

              
                h, w = processed_frame.shape[:2]
                plot_img = draw_score_curve(h, PLOT_WIDTH, score_history, PLOT_WINDOW)

              
                final_frame = np.hstack([processed_frame, plot_img])

               
                if recording and is_save:
        
                    if video_writer is None:
                        final_h, final_w = final_frame.shape[:2]
                        
                        video_writer = cv2.VideoWriter(save_path, fourcc, fps, (final_w, final_h))
                    video_writer.write(final_frame)

     
                if IS_SHOW:
       
                    show_h, show_w = final_frame.shape[:2]
                    display_h = 800
                    if show_h > display_h:
                        scale = display_h / show_h
                        display_w = int(show_w * scale)
                        show_frame = cv2.resize(final_frame, (display_w, display_h))
                    else:
                        show_frame = final_frame

      
                    cv2.imshow('Video Inference', show_frame)
       
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
       
                        break
                    elif key == ord('v'):
           
                        if is_save:
             
                            recording = not recording
                            if recording:
                                print("continue...")
                            else:
                                print("pause...")
                        else:
                      
                            recording = not recording
                            if recording:
                          
                                if video_writer is None:
                                    final_h, final_w = final_frame.shape[:2]
                          
                                    video_writer = cv2.VideoWriter(save_path, fourcc, fps, (final_w, final_h))
                            
                            else:
                            
                                if video_writer is not None:
                                    video_writer.release()
                                    video_writer = None

            except Exception as e:
                print(f"\n err: {e}")
                import traceback
                traceback.print_exc()
                continue

            frame_count += 1
            pbar.update(1)

    except KeyboardInterrupt:

    except Exception as e:
        print(f"\n err: {e}")
        import traceback
        traceback.print_exc()
    finally:
      
        if 'pbar' in locals():
            pbar.close()
        cap.release()
        
     
        if video_writer:
            video_writer.release()
            if is_save:
                print(f" saved: {save_path}")
        elif is_save:
            print("not save")
        
        cv2.destroyAllWindows()

        total_time = time.time() - start_time
        if frame_count > 0:
            print(f"FPS: {frame_count / total_time:.2f}")
        if recording:
            print("stopped")


if __name__ == '__main__':
    if is_camera:
    
        pipeline = init_realsense_camera(camera_width, camera_height, camera_fps)
        if not pipeline:
            print("no camera")
            exit(1)
        

        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        fps = camera_fps
        total_frames = 0  
        file_name = f"realsense_camera_{camera_width}x{camera_height}_{camera_fps}fps_{timestamp}.mp4"
 
        if is_save:
            save_path = os.path.join(output_dir, f"result_{file_name}")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = None
            print(f"save at: {save_path}")
        else:
            save_path = os.path.join(output_dir, f"result_{file_name}")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = None
            print("not save")
       

    
        recording = False

        score_history = []  
        last_valid_state = None 

        start_time = time.time()
        frame_count = 0

        try:
            while True:
        
                frames = pipeline.wait_for_frames()
                color_frame = frames.get_color_frame()
                
                if not color_frame:
                    continue
                
                frame = np.asanyarray(color_frame.get_data())

                try:
                    
                    frame_copy = frame.copy()
                    
                   
                    processed_frame, current_scores, last_valid_state = process_frame(frame_copy, last_valid_state)

                  
                    score_history.append(current_scores)

                  
                    h, w = processed_frame.shape[:2]
                    plot_img = draw_score_curve(h, PLOT_WIDTH, score_history, PLOT_WINDOW)

                   
                    final_frame = np.hstack([processed_frame, plot_img])

                   
                    if recording and is_save and video_writer:
                        video_writer.write(final_frame)

                
                    if IS_SHOW:
                       
                        show_h, show_w = final_frame.shape[:2]
                        display_h = 800
                        if show_h > display_h:
                            scale = display_h / show_h
                            display_w = int(show_w * scale)
                            show_frame = cv2.resize(final_frame, (display_w, display_h))
                        else:
                            show_frame = final_frame

                    
                        cv2.imshow('Video Inference', show_frame)
                  
                        key = cv2.waitKey(1) & 0xFF
                        if key == ord('q'):
                            print("\n stop")
                            break
                        elif key == ord('v'):
                        
                            recording = not recording
                            if recording:
                                print("recoding...")
                           
                                if video_writer is None and is_save:
                                    final_h, final_w = final_frame.shape[:2]
                          
                                    video_writer = cv2.VideoWriter(save_path, fourcc, fps, (final_w, final_h))
                                elif not is_save:
                                    print("is_save=0")
                            else:
                        
                                if video_writer is not None:
                                    video_writer.release()
                                    video_writer = None
                                    if is_save:
                                        print(f"save at: {save_path}")

                except Exception as e:
                    print(f"\n erro: {e}")
                    import traceback
                    traceback.print_exc()
                    continue

                frame_count += 1

        except KeyboardInterrupt:
            print("\n stop")
        except Exception as e:
            print(f"\n erro: {e}")
            import traceback
            traceback.print_exc()
        finally:
  
            pipeline.stop()
       
            
 
            if video_writer:
                video_writer.release()
                if is_save:
                    print(f"save at: {save_path}")
            
            cv2.destroyAllWindows()

            total_time = time.time() - start_time
     
            if frame_count > 0:
                print(f"FPS: {frame_count / total_time:.2f}")
            if recording:
                print("stopped")
    else:
  
        if is_all:
      
            if not os.path.isdir(video_path):
                print(f"Error: video_path  {video_path}")
                exit(1)
            
            mp4_files = [f for f in os.listdir(video_path) if f.lower().endswith('.mp4')]
            if not mp4_files:
                print(f"Error:  {video_path}")
                exit(1)
            
            for i, mp4_file in enumerate(mp4_files):
           
                video_file_path = os.path.join(video_path, mp4_file)
                process_single_video(video_file_path, output_dir, is_save)
                print("\n" + "="*50)
        else:
            if not os.path.exists(video_path):
                print(f"Error:-> {video_path}")
                exit()
            
            process_single_video(video_path, output_dir, is_save)