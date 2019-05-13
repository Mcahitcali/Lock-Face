import cv2, os
import knn_classifier
import exampFace
import pickle
import face_recognition
from time import sleep

def initData(name):
    """[summary]
    This func capture your face in webcam and save for dataset 
    Arguments:
        name {[str]} -- [name is user, showing on cam and folder name in training directory]
    
    Returns:
        [None] -- [Void]
    """
    video_capture = cv2.VideoCapture(0)
    count = 1 # data count

    def new_train_dir(name):
        """[summary]
        
        Arguments:
            name {[str]} -- [for folder name]
        
        Returns:
            [str] -- [your dataset folder path in training directory]
        """
        dir_path = os.path.join("knn_examples/train/",name)
        try:
            os.mkdir(dir_path)
            return dir_path
        except OSError:
            print("Folder exists or Error!")
            return dir_path
        else:
            print("created")
            return dir_path
            
    while count < 10:
        ret, frame = video_capture.read()
        # print(count)

        cv2.imshow('Video', frame)
        sleep(1)

        dir_path = new_train_dir(name)
        cv2.imwrite(dir_path +"/"+name+str(count)+".jpg", frame)
        count += 1

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()


def detect_face(knn=None, m_path=None):
    """[summary]
    Capture your face in real time with webcam
    and starting classification knn algorithm

    if find your face 
        show in rectangle with your name  
    
    Keyword Arguments:
        knn {[knn]} -- [knn algorithm] (default: {None})
        m_path {[str]} -- [model path] (default: {None})
    """
    # Initialize some variables
    video_capture = cv2.VideoCapture(0)
    face_locations = []
    face_encodings = []
    process_this_frame = True

    while True:
        # Grab a single frame of video
        ret, frame = video_capture.read()

        # Resize frame of video to 1/4 size for faster face recognition processing
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

        # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        rgb_small_frame = small_frame[:, :, ::-1]

        # Only process every other frame of video to save time
        if process_this_frame:
            # Find all the faces and face encodings in the current frame of video
            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(
                rgb_small_frame, face_locations)

        predictions = knn_classifier.predict(face_encodings, face_locations, knn_clf=knn, model_path=m_path)

        process_this_frame = not process_this_frame

        # Display the results
        for name, (top, right, bottom, left) in predictions:
            # Scale back up face locations since the frame we detected in was scaled to 1/4 size
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4

            # Draw a box around the face
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

            # Draw a label with a name below the face
            cv2.rectangle(frame, (left, bottom - 35),
                          (right, bottom), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6),
                        font, 1.0, (255, 255, 255), 1)

        # Display the resulting image
        cv2.imshow('Video', frame)

        # Hit 'q' on the keyboard to quit!
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release handle to the webcam
    video_capture.release()
    cv2.destroyAllWindows()

'''
def recognition(image_path):
    """[summary]
    default funcs in face_recognition lib
    Arguments:
        image_path {[str]} -- []
    """
    exampFace.init(image_path)
'''

if __name__ == "__main__":
    ## init ##
    name = "your name"
    train_dir_path = "knn_examples/train"

    initData(name) 

    """
    Trains a k-nearest neighbors classifier for face recognition.
    """
    classifier = knn_classifier.train(train_dir_path, model_save_path="trained_knn_model.clf", n_neighbors=2)
    detect_face(m_path="trained_knn_model.clf")
