import argparse
import sys
import time
import torch
import os
import glob
import atexit
from datetime import timedelta
from moviepy.video.io.VideoFileClip import VideoFileClip
from moviepy.video.io.ffmpeg_writer import FFMPEG_VideoWriter
from moviepy.audio.io.ffmpeg_audiowriter import FFMPEG_AudioWriter
from moviepy.audio.io.AudioFileClip import AudioFileClip
from moviepy.audio.AudioClip import (
    CompositeAudioClip,
    AudioArrayClip
)

parser = argparse.ArgumentParser()
parser.add_argument(
    '--source', help='Video source file', required=True, type=str, action='append')
parser.add_argument(
    '--dest', help='Video destination file', required=True, type=str)
parser.add_argument(
    '--duration', help='Stop processing after N seconds', default=(90 * 60), type=int)
args = parser.parse_args()


def make_tmpfile(hint, suffix):
    return f"tmp_{os.getpid()}.{hint}{suffix}"


def exit_handler():
    for f in glob.glob(f"tmp_{os.getpid()}*"):
        try:
            os.unlink(f)
        except Exception:
            continue


atexit.register(exit_handler)


class VideoReader:
    def __init__(self, videoFile):
        self.video = VideoFileClip(videoFile)
        self.videoIter = self.video.iter_frames()
        self.audio = AudioFileClip(videoFile)
        self.audio.reader.seek(0)

    def readVideoChunk(self, duration):
        count = int(round(self.video.fps, 0)) * duration
        chunk = []
        while count > 0:
            frame = next(self.videoIter, None)
            if frame is None:
                self.video.close()
                break
            chunk.append(frame)
            count -= 1
        if count != 0:
            return None
        return chunk

    def readAudioChunk(self, duration):
        chunkSize = duration * self.audio.fps
        chunk = self.audio.reader.read_chunk(chunkSize)
        if chunk.size != 0:
            clip = AudioArrayClip(chunk, fps=self.audio.fps)
            if clip.duration == duration:
                return clip.to_soundarray(quantize=True, buffersize=chunk.size)
        return None


MODEL = torch.hub.load('ultralytics/yolov5', 'yolov5s')
MODEL.classes = 0, 7  # 0:person, 7:truck/zamboni


class ImageAnalysis:
    def __init__(self, personCount, personDensity, truckCount):
        self.personCount = personCount
        self.personDensity = personDensity
        self.truckCount = truckCount


def analyzeImage(image):
    detections = MODEL(image)
    rs = detections.pandas().xyxy[0]
    personCount = 0
    personDensity = 0
    truckCount = 0
    for index, row in rs.iterrows():
        if row["class"] == 0:
            personDensity += ((row["xmax"] - row["xmin"]) *
                              (row["ymax"] - row["ymin"])) * row["confidence"]
            personCount += 1
        elif row["class"] == 7:
            truckCount += 1
    return ImageAnalysis(personCount, personDensity, truckCount)


sources = [VideoReader(x) for x in args.source]

videoOut = FFMPEG_VideoWriter(
    filename=make_tmpfile(0, ".avi"),
    size=(1920, 1080),
    fps=30,
    codec='libx264',
    preset='ultrafast',
    threads=4,
)

audioOuts = [
    FFMPEG_AudioWriter(
        filename=make_tmpfile(index, ".mp3"),
        fps_input=44100,
        codec="libmp3lame",
    )
    for index in range(len(sources))
]

START_TIME = time.time()

SKIPSIZE = 2.0   # Inspect video every N seconds

duration = 0
while duration < args.duration:

    videoChunks = [input.readVideoChunk(SKIPSIZE) for input in sources]
    audioChunks = [input.readAudioChunk(SKIPSIZE) for input in sources]

    maxDensityScore = -1
    maxDensityIndex = -1
    for index, chunk in enumerate(videoChunks):
        if chunk is None:
            continue

        result = analyzeImage(chunk[0])
        densityScore = int(result.personCount * result.personDensity)
        if densityScore > maxDensityScore:
            maxDensityScore = densityScore
            maxDensityIndex = index

    if maxDensityIndex == -1:
        print("stopping video detection")
        break

    print(str(timedelta(seconds=(duration))),
          "writing video chunk from source", maxDensityIndex,
          "with density score", maxDensityScore)

    videoChunk = videoChunks[maxDensityIndex]
    for frame in videoChunk:
        videoOut.write_frame(frame)

    for index, audioChunk in enumerate(audioChunks):
        if audioChunk is None:
            continue
        audioOuts[index].write_frames(audioChunk)

    duration += SKIPSIZE

videoOut.close()
[x.close() for x in audioOuts]

with VideoFileClip(videoOut.filename) as finalVideo:
    finalVideo.audio = CompositeAudioClip(
        [AudioFileClip(x.filename) for x in audioOuts])
    finalVideo.audio.duration = finalVideo.duration
    finalVideo.write_videofile(
        args.dest,
        codec='libx264',
        audio_codec='libmp3lame',
        preset='veryfast',
        threads=4,
    )

print('Execution time:', str(timedelta(seconds=(time.time() - START_TIME))))
