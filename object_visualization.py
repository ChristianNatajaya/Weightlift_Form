from image_analysis import evaluate_ai_models
from PIL import Image
import numpy as np
import torchvision.transforms as T
import boto3
import sched
import time
import cv2

def poll_database():
    # Get oldest file in S3 bucket
    get_last_modified = lambda obj: int(obj['LastModified'].strftime('%s'))
    objects = s3.list_objects_v2(Bucket=bucket_name)['Contents']
    firstmod_time = [obj['LastModified'] for obj in sorted(objects, key=get_last_modified)][0]
    lastmod_time = [obj['LastModified'] for obj in sorted(objects, key=get_last_modified)][-1]
    firstmod_key = [obj['Key'] for obj in sorted(objects, key=get_last_modified)][0]
    lastmod_key = [obj['Key'] for obj in sorted(objects, key=get_last_modified)][-1]
    print("First Modified Time: ", firstmod_time, "\nLast Modified Time: ", lastmod_time)
    print("First Modified Key: ", firstmod_key, "\nLast Modified Key: ", lastmod_key)

    # Convert video to image frames
    video_path = '/Users/christiannatajaya/Desktop/Form_Ranker/S3_Video/video.mp4'
    image_directory = '/Users/christiannatajaya/Desktop/Form_Ranker/S3_Frames/'
    s3.download_file(Bucket=bucket_name, Key=firstmod_key, Filename=video_path)
    print("Successfully downloaded MP4 file!")
    return firstmod_key, video_path, image_directory

def frame_to_video(output_frames):
    fps = 2
    width, height = output_frames[0].size
    size = (width, height)
    file_path = '/Users/christiannatajaya/Desktop/Form_Ranker/project.mp4'
    # fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    video_output = cv2.VideoWriter(file_path, fourcc, fps, size)

    for i in range(len(output_frames)):
        # Convert RGBA image to RGB then convert PIL image to CV2 image
        output_frame = cv2.cvtColor(np.asarray(output_frames[i]), cv2.COLOR_RGBA2BGR)
        video_output.write(output_frame)
        
    video_output.release()
    print("Successfully created video!")
    return file_path

def upload_file(FILE_NAME, output_video):
    new_file_name = FILE_NAME.split('.mp4')[0] + '_Analyzed.mp4'
    with open(output_video, "rb") as f:
        s3.upload_fileobj(f, bucket_name, new_file_name)
    print("File uploaded to S3!\n")

def delete_file(FILE_NAME):
    s3_object = s3_resource.Object(bucket_name, FILE_NAME)
    s3_object.delete()
    print('File deleted from S3!')


def main():
    try:
        input_file, video_path, image_directory = poll_database()
        output_frames = evaluate_ai_models(video_path, image_directory)
        output_file = frame_to_video(output_frames)
        upload_file(input_file, output_file)
        delete_file(input_file)
    except Exception as error_message:
        print(error_message)

    event_schedule.enter(1,1,main)

if __name__ == "__main__":
    ACCESS_KEY = 'AKIARJB4VMNQQG3W3QHJ'
    SECRET_KEY = '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'
    bucket_name = "rankmyformadae3a1ae7584d9c9efffd8e30d9cca4161159-dev"

    s3 = boto3.client(
        's3',
        region_name='us-east-2',
        aws_access_key_id=ACCESS_KEY,
        aws_secret_access_key=SECRET_KEY
    )
    s3_resource = boto3.resource(
        's3',
        region_name='us-east-2',
        aws_access_key_id=ACCESS_KEY,
        aws_secret_access_key=SECRET_KEY
    )

    event_schedule = sched.scheduler(time.time, time.sleep)
    event_schedule.enter(1,1,main)
    event_schedule.run()



# queue_url = 'https://sqs.us-east-2.amazonaws.com/088174060385/ExecuteVisualizationEC2.fifo'
#     sqs = boto3.client(
#         'sqs', 
#         region_name='us-east-2',
#         aws_access_key_id=ACCESS_KEY,
#         aws_secret_access_key=SECRET_KEY
#     )

# def poll_queue(QUEUE_URL):
#     try:
#         response = sqs.receive_message(
#             QueueUrl = QUEUE_URL,
#             AttributeNames = ['MessageGroupId'],
#             MessageAttributeNames = ['All'],
#             MaxNumberOfMessages=1,
#             WaitTimeSeconds=0
#         )
#         message = response['Messages'][0]
#         message_group_id = message['Attributes']['MessageGroupId']
#         receipt_handle = message['ReceiptHandle']
#         file_name = message['Body']
#         print(message_group_id, 'Successfully received message!')
#         return message_group_id, file_name, receipt_handle

#     except Exception as error_message:
#         print('No message in SQS queue! %s' %error_message)
#         return None, None, None

# def delete_message(QUEUE_URL, RECEIPT_HANDLE):
#     sqs.delete_message(
#         QueueUrl = QUEUE_URL,
#         ReceiptHandle = RECEIPT_HANDLE
#     )
#     print('Message deleted from SQS!')
