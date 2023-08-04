import numpy as np
import cv2
import time

import torch
import torch.optim as optim
from models.sgcn import SGCNModel, connectivity, hand_adj_matrix, get_sgcn_identity
from utils.checkpoints import load_checkpoint

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2

def get_hand_landmarks(image):
    originalImage = image.copy()
    results = hands.process(image)
    landMarkList = []

    if results.multi_hand_landmarks:  # returns None if hand is not found
        for idx in range(len(results.multi_hand_landmarks)):
            hand = results.multi_hand_landmarks[idx] #results.multi_hand_landmarks returns landMarks for all the hands
            landMarkList.append([[landMark.x, landMark.y, landMark.z] for landMark in hand.landmark])

    return landMarkList


def draw_landmarks_on_image(rgb_image, hand_landmarks_list, dim=3):
    annotated_image = np.copy(rgb_image)

    # Loop through the detected hands to visualize.
    for idx in range(len(hand_landmarks_list)):
        hand_landmarks = hand_landmarks_list[idx]
        # handedness = handedness_list[idx]

        # Draw the hand landmarks.
        hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        if dim == 3:
            hand_landmarks_proto.landmark.extend([
              landmark_pb2.NormalizedLandmark(x=x, y=y, z=z) for (x, y, z) in hand_landmarks
            ])
        else:
            hand_landmarks_proto.landmark.extend([
              landmark_pb2.NormalizedLandmark(x=x, y=y) for (x, y) in hand_landmarks
            ])
        solutions.drawing_utils.draw_landmarks(
          annotated_image,
          hand_landmarks_proto,
          solutions.hands.HAND_CONNECTIONS,
          solutions.drawing_styles.get_default_hand_landmarks_style(),
          solutions.drawing_styles.get_default_hand_connections_style())

    return annotated_image



## exp arguments
class Args:
    ...
    
    def __str__(self):
        return 'Args:\n\t> ' + f'\n\t> '.join([f'{key:.<20}: { val}' for key, val in self.__dict__.items()])
    
    __repr__ = __str__
    
args = Args()
args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
args.num_nodes = 21
args.max_seq_len = 80
args.labels_encoder = None
args.connectivity = connectivity
args.num_features = 4
args.num_asymmetric_convs = 6
args.embedding_dims = 64
args.num_gcn_layers = 1
args.num_heads = 4
args.dropout = 0.5
args.lr = 1e-3
args.weight_decay = 5e-2
args.T_max = 20
args.model_path = './weights/model_best_ipn.pth'
connectivity = np.array([args.connectivity[i] / 3 for i in args.connectivity]).reshape(-1, 1)


IPN_labels = ['Non-gesture',
              'Pointing with one finger',
              'Pointing with two fingers',
              'Click with one finger',
              'Click with two fingers',
              'Throw up',
              'Throw down',
              'Throw left',
              'Throw right',
              'Open twice',
              'Double click with one finger',
              'Double click with two fingers',
              'Zoom in',
              'Zoom out']

args.num_classes = len(IPN_labels)
label_encoding = dict(zip(range(len(IPN_labels)), IPN_labels))
print(label_encoding)

## load model
sgcn = SGCNModel(args)
optimizer = optim.AdamW(sgcn.parameters(), lr=args.lr)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.T_max, eta_min=1e-6, last_epoch=-1, verbose=False)

sgcn, optimizer, scheduler, start_epoch, best_val = load_checkpoint(sgcn, optimizer, scheduler, args.model_path)
sgcn = torch.nn.DataParallel(sgcn).to(args.device)


## mediapipe hand model
mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)


gesture = 'unzoom'

## start recording
capture = cv2.VideoCapture(0)
fourcc = cv2.VideoWriter_fourcc('X','V','I','D')
videoWriter = cv2.VideoWriter(f'./images/video_{gesture}.avi', fourcc, 30.0, (640,480))
 
landmarks_sequence = []
sequence_completed = False

## text params
font = cv2.FONT_HERSHEY_SIMPLEX
org_geste = (50, 50)
org_time1 = (50, 80)
org_time2 = (50, 110)
fontScale = 0.5
color = (0, 0, 0)
thickness = 1


while (True):
 
    start_process = time.time()
    ret, frame = capture.read()

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    while sequence_completed:
        for i in range(20):
            videoWriter.write(annot_image)
        time.sleep(5)
        sequence_completed = False
        landmarks_sequence = []
     
    if ret:

        landmarks = get_hand_landmarks(frame)
        annot_image = draw_landmarks_on_image(frame, landmarks)
        if len(landmarks) > 0:
            landmarks_sequence.append(landmarks[0])


        if len(landmarks_sequence) == args.max_seq_len:
            extraction_time = time.time() - start_process

            x = np.array(landmarks_sequence)
            x = torch.from_numpy(x).unsqueeze(0)
            conn = torch.from_numpy(connectivity)
            conn = conn.repeat((*x.shape[:2], 1, 1))
            x = torch.cat((x, conn), dim=-1)
            x = x.to(args.device)
            identity = get_sgcn_identity(x.shape, args.device)

            start = time.time()
            prediction, *_ = sgcn(x.type(torch.float32), identity)
            prediction_time = time.time() - start

            pred_gest = prediction[0].argmax().item()


            annot_image = cv2.putText(annot_image, f"> predicted gesture: {label_encoding[pred_gest]}",
                                     org_geste, font, fontScale, color, thickness, cv2.LINE_AA)

            annot_image = cv2.putText(annot_image, f"> extraction time: {round(extraction_time, 3)}(s)",
                                     org_time1, font, fontScale, color, thickness, cv2.LINE_AA)

            annot_image = cv2.putText(annot_image, f"> prediction time: {round(prediction_time, 3)}(s)",
                                     org_time2, font, fontScale, color, thickness, cv2.LINE_AA)

            sequence_completed = True


    annot_image = cv2.cvtColor(annot_image, cv2.COLOR_RGB2BGR)
    cv2.imshow('video', annot_image)
    videoWriter.write(annot_image)

 
    if cv2.waitKey(1) == 27:
        break


 
capture.release()
videoWriter.release()
 
cv2.destroyAllWindows()