""" This module is responsible for preparing the datasets 
and training the AI models found in visualization_models"""

import visualization_models as AI
from image_transform import image_transform, binary_transformation, resize_to_original, visualize
from model_training import train_segmentation_model, train_scoring_model
from PIL import Image
import numpy as np
import torch
import cv2
import os

def VideoToFrames(path, image_directory):
    video_object = cv2.VideoCapture(path)
    os.chdir(image_directory)
    num_frame = 0
    capture_success = 1

    while capture_success:
        success, image = video_object.read()
        try:
            cv2.imwrite("Frame%d.png" %num_frame, image)
            num_frame += 1
        except:
            break
            
    return num_frame

def prepare_segmentation_dataset(start, end, image_directory):
    transform_function = image_transform(224, 224)
    # Input size is tensor(1,3,224,224) and output size is tensor(1,21,224,224)
    # data_set will hold dictionaries containing both input and output tensors
    data_set = []
    for i in range(start, end, 10):
        original_path = image_directory + "Frame%d.png" %i
        input_image = Image.open(original_path)
        input_image = transform_function.transform(transform_function.toTensor(input_image))
        output_segment = np.zeros((1,21,224,224), dtype=int)
        try:
            # binary_path = image_directory + "Binary Frame%d.png" %i
            binary_path = image_directory + "Martins Licis Frame%d (binary).png" %i
            output_image = Image.open(binary_path)
            output_image = transform_function.transform(transform_function.toTensor(output_image))       
            # Convert RGB to ResNet101 classes: Background is index 0; Person is index 15
            # After transformation: RGB mean is [0.485,0.456,0.406] and RGB std is [0.229,0.224,0.225]
            for i in range(224):
                for j in range(224):
                    if (output_image[0][i][j] >= 0.485) and (output_image[1][i][j] >= 0.456) and (output_image[2][i][j] >= 0.406):
                        output_segment[0][0][i][j] = 1
                        output_segment[0][15][i][j] = 0
                    else:
                        output_segment[0][0][i][j] = 0
                        output_segment[0][15][i][j] = 1
        except Exception as error:
            print(error)
            pass

        data = {
            'x':input_image.unsqueeze(0),
            'y':torch.from_numpy(output_segment),
            'original':Image.open(original_path)
        }
        data_set.append(data)

    return data_set

def prepare_scoring_dataset(segmented_dataset, score):
    image_sequence = np.empty((len(segmented_dataset),3,224,224))

    # Use target y_values from segmentation dataset for lstm_input
    for frame in segmented_dataset:
        y = frame['y'].float()
        fcn_output = torch.argmax(y.squeeze(), dim=0).detach().cpu().numpy()
        lstm_input = binary_transformation(fcn_output)
        np.append(image_sequence, lstm_input)

    # Let label for Martins Licis = 1, in order of video paths
    data_set = {
        'x':torch.from_numpy(image_sequence),
        'y':score
    }

    return data_set

def train_ai_models():
    fcn_model = AI.fcn_model
    lstm_model = AI.lstm_model
    scoring_train_set = []
    scoring_test_set = []

    # Paths to training/test data
    video_paths = [
        "/Users/christiannatajaya/Desktop/Form_Ranker/Deadlift Training Data/Martins Licis Deadlift.mp4"
    ]
    image_directories = [
        "/Users/christiannatajaya/Desktop/Form_Ranker/Deadlift Training Data/Martins Licis Frames/"
    ]
    scores = [
        1.
    ]
    
    # Loop through every training video available and prepare training and validation datasets
    for i in range(len(video_paths)):
        num_frames = VideoToFrames(video_paths[i], image_directories[i]) 
        segmentation_train_set = prepare_segmentation_dataset(0, num_frames, image_directories[i])
        segmentation_test_set = prepare_segmentation_dataset(5, num_frames, image_directories[i])
        scoring_train_set.append(prepare_scoring_dataset(segmentation_train_set, scores[i]))    
        scoring_test_set.append(prepare_scoring_dataset(segmentation_test_set, scores[i]))
        train_segmentation_model(segmentation_train_set, segmentation_test_set, fcn_model)

    train_scoring_model(scoring_train_set, scoring_test_set, lstm_model)
    PATH1 = '/Users/christiannatajaya/Desktop/Form_Ranker/ImageFCN_model.pth'
    PATH2 = '/Users/christiannatajaya/Desktop/Form_Ranker/ImageLSTM_model.pth'
    torch.save(fcn_model.state_dict(), PATH1)
    torch.save(lstm_model.state_dict(), PATH2)

# PUSH THIS TO OBJECT_VISUALIZATION MODULE!
def evaluate_ai_models(video_path, image_directory):
    # Load models
    fcn_model = AI.fcn_model
    lstm_model = AI.lstm_model
    fcn_model.load_state_dict(torch.load('/Users/christiannatajaya/Desktop/Form_Ranker/ImageFCN_model.pth'))
    lstm_model.load_state_dict(torch.load('/Users/christiannatajaya/Desktop/Form_Ranker/ImageLSTM_model.pth'))

    # Prepare dataset
    num_frames = VideoToFrames(video_path, image_directory) 
    image_dataset = prepare_segmentation_dataset(5, num_frames, image_directory)
    output_frames = []
    lstm_input = []

    # Skeletonize dataset
    for frame in image_dataset:
        fcn_input = frame['x'].float()
        fcn_output = fcn_model(fcn_input)['out']
        fcn_output = torch.argmax(fcn_output.squeeze(), dim=0).detach().cpu().numpy()
        skeleton_array = binary_transformation(fcn_output)
        overlay_image, overlay_base64 = resize_to_original(frame['original'], skeleton_array)
        output_frames.append(overlay_image)
        lstm_input.append(skeleton_array)
    
    # Get score, reshape from (15,3,224,224) to (1,15,150528)
    lstm_input = torch.reshape(torch.Tensor(lstm_input), (1, len(image_dataset), 150528))
    output_score = lstm_model(torch.Tensor(lstm_input))
    visualize(overlay_image)
    print(output_score)
    return output_frames, output_score


if __name__ == "__main__":
    train_ai_models()
    evaluate_ai_models(
        "/Users/christiannatajaya/Desktop/Form_Ranker/Deadlift Training Data/Martins Licis Deadlift.mp4",
        "/Users/christiannatajaya/Desktop/Form_Ranker/Deadlift Training Data/Martins Licis Frames/"
    )