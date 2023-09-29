import time
import torch
import os
from datetime import timedelta
import uuid
from moviepy.video.VideoClip import ImageClip
from moviepy.video.fx.resize import resize
from moviepy.video.compositing.concatenate import concatenate_videoclips
from moviepy.video.io.VideoFileClip import VideoFileClip
from moviepy.video.io.ffmpeg_writer import FFMPEG_VideoWriter
from moviepy.audio.io.ffmpeg_audiowriter import FFMPEG_AudioWriter
from moviepy.audio.io.AudioFileClip import AudioFileClip
from moviepy.audio.AudioClip import (
    CompositeAudioClip,
    concatenate_audioclips
)
from moviepy.video.io.ffmpeg_tools import ffmpeg_merge_video_audio
import moviepy.config as moviepyconf
from moviepy.tools import subprocess_call


class VideoReader:
    def __init__(self, file):
        if os.path.isdir(file):
            self.video = concatenate_videoclips(
                [VideoFileClip(os.path.join(file, filename), audio=False)
                 for filename in sorted(os.listdir(file))]
            )
            self.audio = concatenate_audioclips(
                [AudioFileClip(os.path.join(file, filename))
                 for filename in sorted(os.listdir(file))]
            )
        else:
            self.video = VideoFileClip(file, audio=False)
            self.audio = AudioFileClip(file)

        self.videoIter = self.video.iter_frames()
        self.audioIter = None

    def readVideoChunk(self, duration):
        chunk = []
        count = int(round(self.video.fps, 0) * duration)
        while count > 0:
            frame = next(self.videoIter, None)
            if frame is None:
                break
            chunk.append(frame)
            count -= 1
        return chunk if count == 0 else None

    def readAudioChunk(self, duration):
        if self.audioIter is None:
            self.audioIter = self.audio.iter_chunks(
                chunk_duration=duration, quantize=True)
        return next(self.audioIter, None)

    def close(self):
        # TODO: File moviepy bug CompositeAudioClip doesn't close all clips on close()
        # HACK: Manually close all clips if audio is CompositeAudioClip
        audioClips = getattr(self.audio, "clips", None)
        if audioClips is not None:
            [x.close() for x in audioClips]
        self.audio.close()
        self.video.close()


class ImageAnalysis:
    def __init__(self, personCount, personDensity, truckCount):
        self.personCount = personCount
        self.personDensity = personDensity
        self.truckCount = truckCount


class VideoProcessor():

    def __init__(self):
        self.TEMP_FOLDER = "temp"
        self.MODEL = torch.hub.load('ultralytics/yolov5', 'yolov5s')
        self.MODEL.classes = 0, 7  # 0:person, 7:truck/zamboni

        # This belongs in init and not every time you run a process.  Way less often.
        try:
            os.makedirs(self.TEMP_FOLDER)
        except FileExistsError:
            # directory already exists
            pass

    def analyzeImage(self, image):
        detections = self.MODEL(image)
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

    def process(self, files, output_length, destination, logoFile, keep):

        START_TIME = time.time()
        INSTANCE_ID = str(uuid.uuid4())

        tempfiles = []

        def make_tmpfile(hint, suffix):
            tempfile = os.path.join(
                self.TEMP_FOLDER, f"tmp_{hint}_{INSTANCE_ID}{suffix}")
            tempfiles.append(tempfile)
            return tempfile

        skip_size = 2.0   # Inspect video every N seconds

        print(f"Files to process: {files}")
        print(f"duration: {output_length} skip_size: {skip_size}")
        sources = [VideoReader(x) for x in files]

        mergedVideoFile = make_tmpfile("merged", ".mp4")
        mergedAudioFile = make_tmpfile("merged", ".m4a")

        videoOut = FFMPEG_VideoWriter(
            filename=mergedVideoFile,
            size=(1920, 1080),
            fps=30,
            codec='h264_nvenc',
            preset='lossless',
            threads=4,
        )

        audioOuts = [
            FFMPEG_AudioWriter(
                filename=make_tmpfile(index, ".m4a"),
                fps_input=44100,
                codec="aac",
            )
            for index in range(len(sources))
        ]

        duration = 0
        while duration < output_length:

            videoChunks = [input.readVideoChunk(
                skip_size) for input in sources]
            audioChunks = [input.readAudioChunk(
                skip_size) for input in sources]

            maxDensityScore = -1
            maxDensityIndex = -1
            for index, chunk in enumerate(videoChunks):
                if chunk is None:
                    continue

                result = self.analyzeImage(chunk[0])
                densityScore = int(result.personCount * result.personDensity)
                if densityScore > maxDensityScore:
                    maxDensityScore = densityScore
                    maxDensityIndex = index

            if maxDensityIndex == -1:
                print("stopping video detection")
                break

            print(str(timedelta(seconds=(duration))),
                  f"writing video chunk from source {maxDensityIndex} with density score {maxDensityScore}")

            videoChunk = videoChunks[maxDensityIndex]
            for frame in videoChunk:
                videoOut.write_frame(frame)

            for index, audioChunk in enumerate(audioChunks):
                if audioChunk is None:
                    continue
                audioOuts[index].write_frames(audioChunk)

            duration += skip_size

        videoOut.close()
        [x.close() for x in audioOuts]
        [x.close() for x in sources]

        finalAudio = CompositeAudioClip(
            [AudioFileClip(x.filename) for x in audioOuts]
        )
        finalAudio.duration = duration
        finalAudio.write_audiofile(
            filename=mergedAudioFile,
            fps=44100,
            codec='aac',
        )
        # TODO: File moviepy bug CompositeAudioClip doesn't close all clips on close()
        # HACK: Manually close all clips in CompositeAudioClip
        [x.close() for x in finalAudio.clips]
        finalAudio.close()

        ffmpeg_params = None
        if logoFile is None:
            ffmpeg_params = "-vcodec copy -acodec copy"
        else:
            ffmpeg_params = f'-i {logoFile} -filter_complex "[2]scale=-1:50[b];[0][b] overlay=25:25"'

#         subprocess_call([f"""
# {moviepyconf.get_setting("FFMPEG_BINARY")} \
# -i {mergedVideoFile} -i {mergedAudioFile} {ffmpeg_params} \
# -y {destination}
# """])
        cmd = [moviepyconf.get_setting("FFMPEG_BINARY"),
               "-i", mergedVideoFile,
               "-i", mergedAudioFile,
               "-vcodec", "copy",
               "-acodec", "copy",
               "-y", destination]
        cmd = []

        subprocess_call(cmd)
        exit()

        if not keep:
            for file in tempfiles:
                print(f"Removing temp files: [{file}]")
                os.remove(file)

        print('Execution time:', str(
            timedelta(seconds=(time.time() - START_TIME))))
