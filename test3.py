import face_recognition
import cv2

video_capture = cv2.VideoCapture("short_hamilton_clip.mp4")

frames = []
frame_count = 0

while video_capture.isOpened():

    ret, frame = video_capture.read()

    if not ret:
        break

    frame = frame[:, :, ::-1]

    frame_count += 1
    frames.append(frame)
    if len(frames) == 128:
        batch_of_face_locations = face_recognition.batch_face_locations(frames, number_of_times_to_upsample=0)

        for frame_number_in_batch, face_locations in enumerate(batch_of_face_locations):
            number_of_faces_in_frame = len(face_locations)

            frame_number = frame_count - 128 + frame_number_in_batch
            print("I found {} face(s) in frame #{}.".format(number_of_faces_in_frame, frame_number))

            for face_location in face_locations:
                top, right, bottom, left = face_location
                print(
                    " - A face is located at pixel location Top: {}, Left: {}, Bottom: {}, Right: {}".format(top, left,
                                                                                                             bottom,
                                                                                                             right))

        frames = []
