from glob import glob
import os
import numpy as np
from scipy.interpolate import interp2d
import cv2
import dlib
from imutils import face_utils


FACIAL_LANDMARKS_IDXS = dict([
    ('mouth', (48, 68)),
    ('right_eyebrow', (17, 22)),
    ('left_eyebrow', (22, 27)),
    ('right_eye', (36, 42)),
    ('left_eye', (42, 48)),
    ('nose', (27, 36)),
    ('jaw', (0, 17))
])


def extract_face_landmarks(video_filename, predictor_params, refresh_size=8):
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(predictor_params)
    tracker = dlib.correlation_tracker()

    cap = cv2.VideoCapture(video_filename)

    tracking_face = False # Keep track if we are using tracker
    i = 0 # Number of frames without detection
    landmarks = []
    rect = None
    
    while cap.isOpened():
        ret, frame = cap.read()
     
        if ret == True:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            if tracking_face and i < refresh_size:
                tracking_quality = tracker.update(gray)
                if tracking_quality >= 8.75:
                    t_pos = tracker.get_position()
                    x = int(t_pos.left())
                    y = int(t_pos.top())
                    w = int(t_pos.width())
                    h = int(t_pos.height())
                    i += 1
                else:
                    tracking_face = False

            if not (tracking_face and i < refresh_size):
                i = 0
                rects = detector(gray, 1)
                if rects:
                    rect = rects[0] # We suppose to have a single face detection
                    tracker.start_track(frame, rect)
                    tracking_face = True
        
            if rect:
                # determine the facial landmarks for the face region, then
                # convert the facial landmark (x,y)-coordinates to a numpy
                # array
                shape = predictor(gray, rect)
                shape = face_utils.shape_to_np(shape)
                landmarks.append(shape)
        else:
            break

    cap.release()

    return np.array(landmarks)


def show_face_landmarks(video_filename, predictor_params="", full_draw=False, landmarks_file="", fps=25.0, refresh_size=8):
    """
    Draws facial landmarks over original video frames. If full_draw is True connected lines
    of face landmark points are showed.
    """
    # Convert fps in frame len in milliseconds
    frame_len = int(1000 / fps) if fps > 0 else 0

    if landmarks_file:
        landmarks = np.loadtxt(landmarks_file, dtype=np.int32).reshape((-1, 68, 2))
    else:
        landmarks = extract_face_landmarks(video_filename, predictor_params, refresh_size)

    cap = cv2.VideoCapture(video_filename)

    for shape in landmarks:
        ret, frame = cap.read()
    
        if ret == True:
            # loop over the (x, y)-coordinates for the facial landmarks
            # and draw them on the image
            for (x, y) in shape:
                cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)

            if full_draw:
                for (i, name) in enumerate(FACIAL_LANDMARKS_IDXS.keys()):
                    # grab the (x, y)-coordinates associated with the
                    # face landmark
                    (j, k) = FACIAL_LANDMARKS_IDXS[name]
                    pts = shape[j:k]
 
                    if name in ('jaw', 'right_eyebrow', 'left_eyebrow'):
                        # since the jawline is a non-enclosed facial region,
                        # just draw lines between the (x, y)-coordinates
                        for l in range(1, len(pts)):
                            ptA = tuple(pts[l - 1])
                            ptB = tuple(pts[l])
                            cv2.line(frame, ptA, ptB, (0, 255, 0), 1)
                    if name in ('right_eye', 'left_eye'):
                        for l in range(len(pts)):
                            ptA = tuple(pts[l - 1])
                            ptB = tuple(pts[l])
                            cv2.line(frame, ptA, ptB, (0, 255, 0), 1)
                    if name == 'nose':
                        for l in range(1, len(pts)):
                            ptA = tuple(pts[l - 1])
                            ptB = tuple(pts[l])
                            cv2.line(frame, ptA, ptB, (0, 255, 0), 1)
                        cv2.line(frame, tuple(pts[-1]), tuple(pts[3]), (0, 255, 0), 1)
                    if name == 'mouth':
                        for l in range(0, 11):
                            ptA = tuple(pts[l])
                            ptB = tuple(pts[l + 1])
                            cv2.line(frame, ptA, ptB, (0, 255, 0), 1)
                        cv2.line(frame, tuple(pts[0]), tuple(pts[11]), (0, 255, 0), 1)
                        for l in range(12, len(pts) - 1):
                            ptA = tuple(pts[l])
                            ptB = tuple(pts[l + 1])
                            cv2.line(frame, ptA, ptB, (0, 255, 0), 1)
                        cv2.line(frame, tuple(pts[12]), tuple(pts[-1]), (0, 255, 0), 1)
         
            # show the output image with face landmarks
            cv2.imshow("Output", frame)
            cv2.waitKey(frame_len)
        else:
            break

    cap.release()
    cv2.destroyAllWindows()


def save_face_landmarks_speaker(video_path, dest_path, predictor_params, file_ext='mpg', refresh_size=8):
    # Create destination directory if not exists
    if not os.path.isdir(dest_path):
        os.makedirs(dest_path)

    video_filenames = glob(os.path.join(video_path, '*.' + file_ext))
    
    count = 0
    for v_file in video_filenames:
        landmarks = extract_face_landmarks(v_file, predictor_params, refresh_size)
        l_file = os.path.join(dest_path, os.path.basename(v_file).replace('.' + file_ext, '.txt'))
        np.savetxt(l_file, landmarks.reshape([-1, 136]), fmt='%d')
        print('{} - Video file: {}'.format(count, v_file))
        count += 1
        
    print('Done. Face landmark files created:', count)


def save_face_landmarks(dataset_path, list_of_speakers, video_dir, dest_dir, predictor_params, file_ext='mpg', refresh_size=8):
    # Every refresh_size frames a new face detection is forced (the face correlation tracker is ignored).

    for s in list_of_speakers:
        print('Computing face landmarks of speaker {:d}...'.format(s))
        video_path = os.path.join(dataset_path, 's' + str(s), video_dir)
        dest_path = os.path.join(dataset_path, 's' + str(s), dest_dir)
        
        save_face_landmarks_speaker(video_path, dest_path, predictor_params, file_ext, refresh_size)

        print('Speaker {:d} completed.'.format(s))