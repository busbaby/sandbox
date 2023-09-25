from moviepy.editor import *
import argparse
from pathlib import Path


argParser = argparse.ArgumentParser()
argParser.add_argument(
    '--starttime', help='Start time of first input file - format HH:MM:SS.ms')
argParser.add_argument(
    '--endtime', help='End time of last input file - format HH:MM:SS.ms')
argParser.add_argument(
    '--dir', help='Directory containing files to merge', required=True)
argParser.add_argument(
    '--dest', help='Name of output file', required=True)

args = argParser.parse_args()

target_dir = Path(args.dir)
if not target_dir.exists():
    raise SystemExit(f"The target directory '{target_dir}' doesn't exist")

inputClips = [VideoFileClip(str(f.absolute())) for f in target_dir.iterdir()]
if len(inputClips) < 2:
    raise SystemExit("Need 2 or more files to merge")

if args.starttime is not None:
    firstFile = inputClips[0]
    print("Setting start position of", firstFile.filename, "at", args.starttime)
    inputClips[0] = firstFile.subclip(args.starttime)
if args.endtime is not None:
    lastFile = inputClips[-1]
    print("Setting end position of", lastFile.filename, "at", args.endtime)
    inputClips[-1] = lastFile.subclip(0, args.endtime)

merged = concatenate_videoclips(inputClips)
merged.write_videofile(
    filename=args.dest,
    codec='libx264',
    preset='ultrafast',
    threads=4
)
