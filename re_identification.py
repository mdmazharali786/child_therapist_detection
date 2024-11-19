import os
import cv2
import numpy as np
from collections import defaultdict
import torchreid
import torch
from pytubefix import YouTube
from ultralytics import YOLO
import time

class Retracking:
    def __init__(self, url):
        self.url = url
        self.id_set = set()
        self.history_by_frame = defaultdict(lambda: defaultdict(lambda: []))
        # Load the pre-trained person ReID model
        self.reid_model = torchreid.models.build_model(name='osnet_x1_0', num_classes=1000, pretrained=True)
        self.reid_model.eval()
    
        self.output_video_path = './output video/'+self.url.split('=')[-1]+'.avi'

        self.run()
  
    def create_video_writer(self, frame_size):
        fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Codec for .avi file
        fps = 24  # Frames per second
        out = cv2.VideoWriter(self.output_video_path, fourcc, fps, frame_size)
        return out

    def get_downloaded_video(self):
        folder = './download/'
        
        if os.path.exists(folder + self.url.split('=')[-1]+'.mp4'):
            video_path = folder + self.url.split('=')[-1]+'.mp4'
        else:
            yt = YouTube(self.url)
            yt.streaming_data
            video_path = yt.streams.first().download(folder, filename=self.url.split('=')[-1]+'.mp4')
        cap = cv2.VideoCapture(video_path)
        return cap

    def run(self):
        # Load the YOLOv8 model
        track_model = YOLO("./child_and_therapist_detection/runs/detect/train/weights/best.pt")
        first_write = False
        frame_size_indicator = True
        cap = self.get_downloaded_video()
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print("Processing url: ", self.url)
        i = 0
        update_frame_num = 0
        # Loop through the video frames
        while cap.isOpened():
            start = time.time()
            # Read a frame from the video
            success, frame = cap.read()

            if success:
                # Run tracking on the downloaded video
                track = track_model.track(frame, persist=True, verbose=False, tracker= 'D:\\Intro prac python notebook\\Computer Vision\\botsort.yaml')
                track = track[0]
                
                if frame_size_indicator:
                    output_video_frame_size = track.orig_shape[::-1]
                    out = self.create_video_writer(output_video_frame_size)
                    frame_size_indicator = False

                if track.boxes.id != None:
                    track_ids = track.boxes.id.int().cpu().tolist()
                    ids_in_curr_frame = []
                    boxes = []
                    label_within_frame = track.boxes.cls.int().cpu().tolist()
                    for j in range(len(track_ids)):
                        class_label = label_within_frame[j]

                        if first_write and track_ids[j] not in self.id_set and len(self.id_set)>0:
                            correct_id = self.re_assign_id(track, j, ids_in_curr_frame, class_label)
                            if correct_id != None:
                                track_ids[j] = correct_id
                        
                        self.id_set.add(track_ids[j])
                        ids_in_curr_frame.append(track_ids[j])

                        self.history_by_frame[class_label]['id'].append(track_ids[j])
                        self.history_by_frame[class_label]['vector'].append(self.box_to_vector(track, j))

                        boxes.append({'bbox': track.boxes.xyxy.int().cpu().tolist()[j], 'id': track_ids[j], 'label': track.names[class_label]})

                    # Visualize the results on the frame
                    frame_new = track.orig_img  
                    annotated_frame = self.draw_boxes(frame_new.copy(), boxes)

                    
                    out.write(annotated_frame)
                    
                    first_write = True
                    update_frame_num+=1
                else:
                    out.write(track.orig_img)

            else:
                break
            i += 1
            end = time.time()
            delta = end - start
            print(f'Frame #{i}/{total_frames}, Time to Process: {round(delta, 3)}s', end='\r')
        out.release()
        cap.release()
        print("Process complete on url:", self.url)
        print("Video saved at ", os.path.abspath(self.output_video_path))

    def box_to_vector(self, frame, box_idx):
        x1, y1, x2, y2 = frame.boxes.xyxy.cpu().tolist()[box_idx]
        box1 = frame.orig_img[int(y1):int(y2),int(x1):int(x2),:]  # Extract region from image1
        box1 = cv2.resize(box1, (256, 128))
        # Convert images to tensors
        box1 = torch.from_numpy(box1)
        # Extract features from the model
        with torch.no_grad():
            feature1 = self.reid_model(box1.unsqueeze(0).permute(0, 3, 1, 2).float())
        return feature1.squeeze(0)

    
    def box_similarity_score(self, feat1, feat2):
        similarity = torch.nn.functional.cosine_similarity(feat1, feat2)
        return similarity.numpy()

    
    def re_assign_id(self, frame, box_idx, curr_frm_ids, cls):
        feat1 = self.box_to_vector(frame, box_idx)
        temp_vectors = self.history_by_frame[cls]['vector']
        if len(temp_vectors) == 0:
            return None

        similarity_scores = self.box_similarity_score(feat1, torch.stack(temp_vectors))
        
        for i in range(box_idx+1):
            max_similarity_idx = np.argmax(similarity_scores)
            if np.max(similarity_scores)<0.9995:
                return None
            new_id = self.history_by_frame[cls]['id'][max_similarity_idx]
            if new_id not in curr_frm_ids:
              return new_id
            else:
              similarity_scores[max_similarity_idx] = 0
        return None
        

    def draw_boxes(self, frame, boxes):
        for box in boxes:
            x1, y1, x2, y2 = box['bbox']  # Get box coordinates
            object_id = box['id']  # Get object ID
            label = box['label']  # Get object label

            if label == "therapist":
                box_color = (0, 0, 255)
            else:
                box_color = (255, 0, 0)
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 1)

            # Add text (ID and label)
            text = f"Id:{object_id} {label}"
            hdiff_tbox = 20
            hdiff_text = 5
            width_tbox = int(len(text)*8.5)
            wdiff = 0

            if x1 + width_tbox > frame.shape[1]:
                wdiff = x1 + width_tbox - frame.shape[1]

            if y1 - hdiff_tbox < 0:
                hdiff_text = hdiff_text - hdiff_tbox
                hdiff_tbox = -hdiff_tbox
                
            cv2.rectangle(frame, (x1-wdiff, y1-hdiff_tbox), (x1-wdiff+width_tbox, y1), box_color, -1)
            cv2.putText(frame, text, (x1-wdiff+2, y1 - hdiff_text), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        return frame    

def main():
    test_urls = open('./child_and_therapist/test_videos.txt', 'r')
    
    for url in test_urls.readlines():
        Retracking(url)

if __name__ == "__main__":
    main()
  