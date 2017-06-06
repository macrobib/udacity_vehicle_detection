from  moviepy.editor import VideoFileClip
import sys
import os
import json
import argparse

def process_image(image):
    """Proecss single image frame."""
    pass

def load_traning_params(path):
    training_data = None
    with open(path, 'r') as f:
        training_data = json.load(f)
    return training_data

def save_params(path, data):
    """Save training parameters."""
    with open(path) as f:
        json.dump(data)

def train_data():
    """Training and validation pipeline."""
    pass

def processVid(inpath, outpath):
    print("Starting video processing.\n")
    clip = VideoFileClip(inpath)
    output_clip = clip.fl_image(process_frame)
    output_clip.write_videofile(outpath, audio=False)
    print("Completed video processing.\n")

def main():
    processVid()

if __name__ == '__main__':
    main()
