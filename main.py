from sklearn import neighbors
import os
import pickle
from PIL import Image, ImageDraw
import face_recognition
from face_recognition.face_recognition_cli import image_files_in_folder
import logging
import argparse
from skimage import io
import urllib

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
UNKNOWN_FACE = 'unknown'

################### Train model ###################
def train(train_dir, model_save_path=None):
    X = []
    y = []
    idx=0
    len_dir_train = len(os.listdir(train_dir))
    for class_dir in os.listdir(train_dir):
        jdx=0
        try:
            if not os.path.isdir(os.path.join(train_dir, class_dir)):
                continue
            for img_path in image_files_in_folder(os.path.join(train_dir, class_dir)):
                try:
                    image = face_recognition.load_image_file(img_path)
                    face_coordinates = face_recognition.face_locations(image)
                    if len(face_coordinates) == 1:
                        X.append(face_recognition.face_encodings(image, known_face_locations=face_coordinates)[0])
                        y.append(class_dir)
                    print(f'Training {jdx}/{idx}/{len_dir_train}')
                    jdx+=1
                except Exception as e:
                    logging.exception(e)
                    continue
            idx+=1
        except:
            continue

    n_neighbors = len(X)
    knn_clf = neighbors.KNeighborsClassifier(n_neighbors=n_neighbors, algorithm='ball_tree', weights='distance')
    knn_clf.fit(X, y)

    if model_save_path is not None:
        with open(model_save_path, 'wb') as f:
            pickle.dump(knn_clf, f)

    return knn_clf

################### Predict images ###################
def predict(path_image, knn_clf=None, model_path=None):
    try:
        if not os.path.isfile(path_image) or os.path.splitext(path_image)[1][1:] not in ALLOWED_EXTENSIONS:
            raise Exception(f"Invalid image path: {path_image}")

        if knn_clf is None and model_path is None:
            raise Exception("Must supply knn classifier either through knn_clf or model_path")

        if knn_clf is None:
            with open(model_path, 'rb') as f:
                knn_clf = pickle.load(f)

        X_img = face_recognition.load_image_file(path_image)
        face_coordinates = face_recognition.face_locations(X_img)

        if len(face_coordinates) == 0:
            return []

        faces_encodings = face_recognition.face_encodings(X_img, known_face_locations=face_coordinates)

        distance_threshold = 0.6
        closest_distances = knn_clf.kneighbors(faces_encodings, n_neighbors=1)
        are_matches = [closest_distances[0][i][0] <= distance_threshold for i in range(len(face_coordinates))]
        return [(pred, loc) if rec else (UNKNOWN_FACE, loc) for pred, loc, rec in zip(knn_clf.predict(faces_encodings), face_coordinates, are_matches)]
    except:None

################### Visualize Predicted ###################
def visualize_predicted(img_path, coordinates):
    pil_image = Image.open(img_path).convert("RGB")
    draw = ImageDraw.Draw(pil_image)
    for name, (top, right, bottom, left) in coordinates:
        draw.rectangle(((left, top), (right, bottom)), outline=(0, 0, 255))
        name = name.encode("UTF-8")
        text_width, text_height = draw.textsize(name)
        draw.rectangle(((left, bottom - text_height - 10), (right, bottom)), fill=(0, 0, 255), outline=(0, 0, 255))
        draw.text((left + 6, bottom - text_height - 5), name, fill=(255, 255, 255, 255))
    del draw
    pil_image.show()

################### Fetch images ###################
def fetch_images():
    paths,codes=[],[]
    with open("image_paths.txt", "r") as files:
       for path in files:
            paths.append(path.replace("\n",""))

    with open("image_names.txt", "r") as files:
       for path in files:
           codes.append(path.replace("\n", ""))

    HOST = ""
    d={}
    len_paths=len(paths)
    root_train='data_training/train/'
    for i in range(len_paths):
        try:
            # Image name count ascending
            if codes[i] in d:
                d[codes[i]] += 1
            else:
                d[codes[i]] = 1
            # Path
            image_url = f'{HOST}{paths[i]}'
            path_folder = f'{root_train}{codes[i]}'
            filename = f'{path_folder}/{d[codes[i]]}.jpg'
            # Make dir
            if not os.path.exists(path_folder):
                os.makedirs(path_folder)

            img_request = urllib.request.urlopen(image_url)
            img_response = face_recognition.load_image_file(img_request)
            face_location = face_recognition.face_locations(img_response)
            if len(face_location) == 1:
                print(f'{i}/{len_paths}: Downloaded {filename}')
                top, right, bottom, left = face_location[0]
                face_image = img_response[top:bottom, left:right]
                pil_image = Image.fromarray(face_image)
                pil_image.save(filename)
        except Exception as e:
            logging.exception(e)
            continue
    print("Fetch images done!!!")

################### MAIN ###################
if __name__ == "__main__":
    train_path = 'data_training/train'
    test_path = 'data_training/test'
    model_path = 'trained_employees.clf'

    # Make default data_training
    try:
        os.makedirs(train_path)
        os.makedirs(test_path)
    except: None

    parser = argparse.ArgumentParser()
    parser.add_argument("-t", '--train', required=False,action='store_true',help="Train model")
    parser.add_argument("-p", '--predict', required=False,action='store_true',help="Predict data")
    parser.add_argument("-v", '--visualize', required=False,action='store_true',help="Visualize data predicted")
    parser.add_argument("-f", '--fetch', required=False,action='store_true',help="Fetch data for train")
    args = parser.parse_args()

    if not args.train and not args.predict and not args.visualize and not args.fetch:
        parser.error('No arguments provided.')

    if not args.predict and args.visualize:
        parser.error("--require option predict before")

    # Train model
    if args.train:
        print("Training...")
        classifier = train(train_path, model_save_path=model_path)
        print("Training completed!")

    # Predict image
    if args.predict:
        for test_image in os.listdir(test_path):
            path_image_test = os.path.join(test_path, test_image)
            eligible_ext = path_image_test[-3:] in ALLOWED_EXTENSIONS
            if eligible_ext:
                predictions = predict(path_image_test, model_path=model_path)
                try:
                    if len(predictions) != 0:
                        if predictions[0][0] != UNKNOWN_FACE:
                            print(f'Detected {path_image_test} match to: "{predictions[0][0]}"')
                        else:
                            print(f'Failed detected {path_image_test}:  unknown face ')
                    else:
                        print(f'Failed detected {path_image_test}: unknown face')
                except Exception as e:
                    print(f'Failed detected {path_image_test}: unknown face')
                    logging.exception(e)
            # Visualize results
            if args.visualize and eligible_ext:
                visualize_predicted(os.path.join(test_path, test_image), predictions)

    # Fetch images
    if args.fetch:
        fetch_images()

sql="""
SELECT * FROM (
    SELECT TOP 10000 row_number() over (partition by w.EmployeeId ORDER BY w.CreateDate DESC) as STT,w.EmployeeId,w.Photo
    FROM Attendants w
    WHERE w.AccountId=3 AND YEAR(w.CreateDate)=2019
    AND w.Photo IS NOT NULL
) a
where a.STT <= 10
ORDER BY a.EmployeeId,a.STT ASC
"""
