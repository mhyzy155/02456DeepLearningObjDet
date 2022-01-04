import sys
sys.path.append('/content/drive/MyDrive/Final Project/')

from centroid_tracking import CentroidTracker
import torch
import cv2
import numpy as np
import time
from IPython.display import clear_output
from collections import namedtuple

Prediction = namedtuple('Prediction', 'box label score')

def convert_frame(img, transforms):
  if transforms is None:
    frame = np.asarray(img)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
  else:
    frame = cv2.cvtColor(img.mul(255).permute(1, 2, 0).numpy().astype(np.uint8), cv2.COLOR_RGB2BGR)
  return frame

def filter_predictions(predictions, score_threshold):
  beers, colas = [], []
  for i, box in enumerate(predictions['boxes'].tolist()):
    score = predictions['scores'][i]
    if score >= score_threshold:
      box = [int(x) for x in box]
      label = predictions['labels'][i].item()
      if label == 1:
        beers.append(Prediction(box, label, score))
      elif label == 2:
        colas.append(Prediction(box, label, score))
  return [beers, colas]


class Painter():
  colors = {0: (255,255,255,128), 1: (0,255,0,128), 2: (100,100,255,128)}
  labels_text = {0: 'background', 1: 'beer', 2: 'cola'}

  @classmethod
  def draw_predictions(cls, frame, objs_list, frame_no, frames_total, time_exe, fps_arr):
    cv2.putText(frame, f"{frame_no:4}/{frames_total}, {time_exe*1000:.0f} ms, {fps_arr[frame_no]:.2f} fps, {np.mean(fps_arr[:frame_no+1]):.2f} avg fps", (5,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255,128), 1, cv2.LINE_AA)
    for objs in objs_list:
      for obj in objs:
        cv2.rectangle(frame,(obj.box[0], obj.box[1]), (obj.box[2], obj.box[3]), cls.colors[obj.label])
        cv2.putText(frame, f"{cls.labels_text[obj.label]} {obj.score:.2f}", (obj.box[0], obj.box[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, cls.colors[obj.label], 1, cv2.LINE_AA)

  @classmethod
  def draw_tracked(cls, frame, objects, label, text = "ID "):
    for (objectID, centroid) in objects.items():
      cv2.putText(frame, f"{text}{objectID}", (centroid[0] - 10, centroid[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, cls.colors[label], 1)
      cv2.circle(frame, (centroid[0], centroid[1]), 3, cls.colors[label], -1)


def benchmark_record(filename, model, dataset, score_threshold = 0.95, fps = 15, tracking = True, frames_forget = 20, use_amp = False, quantize_input = False, use_half = False):
  time_start_abs = time.time()
  input_dtype = None
  if quantize_input:
      input_dtype = torch.qint8
  device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
  cpu_text = '' if torch.cuda.is_available() else '_cpu'
  model.to(device)
  model.eval()
  img_size = dataset[0][0].size if dataset.transforms is None else (dataset[0][0].size(2), dataset[0][0].size(1))
  writer = cv2.VideoWriter(f'/content/drive/MyDrive/Final Project/result_video/{filename}{cpu_text}.avi', cv2.VideoWriter_fourcc(*'XVID'), fps, img_size)
  dataset_len = len(dataset)
  fps_arr = np.zeros(dataset_len)
  accuracy = 0
  accuracy2 = 0

  if tracking:
    ct_beer = CentroidTracker(frames_forget)
    ct_cola = CentroidTracker(frames_forget)

  for frame_no, (img, target) in enumerate(dataset):
    # predict and measure execution time
    time_start = time.time()
    with torch.cuda.amp.autocast(enabled=use_amp):
        with torch.no_grad():
            if use_half:
                prediction = model([img.to(device, dtype=input_dtype).half()])
            else:
                prediction = model([img.to(device, dtype=input_dtype)])
    time_exe = time.time() - time_start
    fps_arr[frame_no] = 1/time_exe
    
    prediction_data = prediction[0]

    # check if images are in PIL or Tensor format and convert to openCV
    frame = convert_frame(img, dataset.transforms)

    # returns [list_of_beers, list_of_colas]
    objs_list = filter_predictions(prediction_data, score_threshold)

    # update accuracy measure
    target_len = len(target['labels'])
    pred_len = len(objs_list[0]) + len(objs_list[1])
    if target_len == pred_len:
      accuracy += 1

    if target_len <= pred_len:
      accuracy2 += 1

    # draw predictions and metrics
    Painter.draw_predictions(frame, objs_list, frame_no, dataset_len, time_exe, fps_arr)

    if tracking:
      objs_beer = ct_beer.update([x.box for x in objs_list[0]])
      objs_cola = ct_cola.update([x.box for x in objs_list[1]])

      Painter.draw_tracked(frame, objs_beer, 1, "B")
      Painter.draw_tracked(frame, objs_cola, 2, "C")

    writer.write(frame)
    time_end = time.time()

    clear_output(wait=True)
    print(f"Frame {frame_no+1}/{dataset_len}")
    print(f"Prediction time: {time_exe:3.3f} s")
    print(f"Frame      time: {time_end - time_start:3.3f} s")
    print(f"Elapsed    time: {time_end - time_start_abs:3.3f} s")
    print(f"Accuracy : {accuracy/(frame_no+1):.3f}")
    print(f"Accuracy2: {accuracy2/(frame_no+1):.3f}")
  writer.release()
