from moviepy.editor import *
import librosa
import numpy as np
import inference
import os
from separator import Separate

config='config/default.yaml'
embedder='embedder.pt'
checkpoint='voiceSplit-trained-with-Si-SRN-GE2E-CorintinJ-best_checkpoint.pt'

output_dir1 = 'split1'
output_dir2 = 'split2'

separate = Separate(config=config, embedder=embedder, checkpoint=checkpoint)

def process2(video1, video2, mixed_audio, reference_audio1, reference_audio2, split_by=60):
    whole_clip = AudioFileClip(mixed_audio)
    parts = int(whole_clip.duration // split_by)

    files = []
    output_names = []
    for part in range(parts):
        if (part+1) * split_by < whole_clip.duration:
            part_clip = whole_clip.subclip(part * split_by, (part+1) * split_by)
        else:
            part_clip = whole_clip.subclip(part * split_by)
        path = 'together_' + str(part) + '.wav'
        # librosa.output.write_wav(path, np.asfortranarray(part_clip.to_soundarray().copy().reshape([2, -1])), sr=8000, norm=True) # path without wav?
        part_clip.write_audiofile('mixed/' + path)

        files.append('mixed/' + path)
        output_names.append(path)

    separate.many(files, [reference_audio1]*len(files), [output_dir1]*len(files), output_names)
    separate.many(files, [reference_audio2]*len(files), [output_dir2]*len(files), output_names)

    s1_list = []
    for file in os.listdir(output_dir1):
        s1_list.append(AudioFileClip(output_dir1 + '/' + file))

    s2_list = []
    for file in os.listdir(output_dir2):
        s2_list.append(AudioFileClip(output_dir2 + '/' + file))

    left = concatenate_audioclips(s1_list).to_soundarray().mean(axis=1)
    right = concatenate_audioclips(s2_list).to_soundarray().mean(axis=1)
    duration = int(concatenate_audioclips(s2_list).duration)
    print(duration, whole_clip.duration)
    # right = right - np.mean(right)
    main = whole_clip.subclip(0, duration).to_soundarray().mean(axis=1)

    martin = VideoFileClip(video1).subclip(0, duration)
    patrik = VideoFileClip(video2).subclip(0, duration)

    def number_per_second(array, seconds):
        try:
            return array.reshape(-1, array.shape[0] // seconds).mean(axis=1)
        except BaseException as e:
            print(array.shape, seconds, array.shape[0] // seconds)
            assert seconds is None

    # right = main-left
    # left = main-right
    # main = number_per_second(main, 10)
    left = number_per_second(left, duration)
    right = number_per_second(right, duration)
    logits = np.where(left > right, 1, 0)

    def split_to_second_subclips(clip, length):
        clips = []
        for i in range(length):
            clips.append(clip.subclip(i, i + 1))

        return clips

    martin = split_to_second_subclips(martin, duration)
    patrik = split_to_second_subclips(patrik, duration)

    def concatenate_final(indices, clips0, clips1):
        final_list = []
        for indx, clip0, clip1 in zip(indices, clips0, clips1):
            if indx:
                final_list.append(clip1)
            else:
                final_list.append(clip0)

        return concatenate_videoclips(final_list)

    final = concatenate_final(logits, martin, patrik)
    final = final.set_audio(whole_clip)
    final.write_videofile('concat0.mp4')
