Description
-----------
Code for the `videosum` Python package. Given a video file, this package produces a single image that summarises the video. The summary image is constructed as a collage of video frames evenly spaced over time.

Install dependencies
--------------------
* Ubuntu/Debian:
```
$ sudo apt install ffmpeg
```

Install with pip
----------------
```
$ python3 -m pip install videosum --user
```


Install from source
-------------------
```
$ python3 setup.py install --user
```


Run video summarisation on a single video
-----------------------------------------
```
$ python3 -m videosum.run --input video.mp4 --output collage.jpg --nframes 100 --height 1080 -width 1920 --algo time
```
Options:
  * `--input`: path to the input video file.
  * `--output`: path where the output collage will be saved.
  * `--nframes`: number of frames that you want to see in the collage image.
  * `--height`: height of the collage image.
  * `--width`: width of the collage image.
  * `--time-segmentation`: set it to either `0` or `1`. If 1, the clustering results are displayed in a bar underneath the collage (i.e. the columns of the bar represent the frames of the video, and the colours represent the clustering label).
  * `--fps`: number of frames you want to read per second of video, used to downsample the input video and have less frames to describe and cluster.
  * `--algo`: algorithm used to select the key frames of the video.
    * `time`: evenly spaced frames are selected.
    * `inception`: k-medoids clustering (l2-norm metric) on InceptionV3 latent space vectors.
    * `fid` : k-medoids clustering ([Frechet Inception Distance](https://en.wikipedia.org/wiki/Fr%C3%A9chet_inception_distance) metric) on InceptionV3 latent space vectors.
    * `scda`: k-medoids clustering (l2-norm metric) on SCDA image descriptors ([Wei et al. 2017 Selective Convolutional Descriptor Aggregation for Fine-Grained Image Retrieval](https://arxiv.org/abs/1604.04994)). InceptionV3 was trained on ImageNet as opposed to Wei et al. where authors used VGG-16.


Exemplary code snippet
----------------------
```python
import cv2
import videosum

# Choose the number of frames you want in the summary
nframes = 100

# Choose the dimensions of the collage
widtth = 1920
height = 1080

# Choose the algotrithm that selects the key frames
algo = 'inception'  # The options are: 'time', 'inception', 'fid', 'scda'

# Create video summariser object
vs = videosum.VideoSummariser(algo, nframes, width, height)

# Create collage image
im = vs.summarise('video.mp4')

# Save collage to file
cv2.imwrite('collage.jpg', im)

# Retrieve a list of Numpy/OpenCV BGR images corresponding to the key frames of the video
key_frames = vs.get_key_frames('video.mp4')       

# Print the video frame indices of the key frames, after calling summarise() or get_key_frames()
print(vs.indices_)
```


Run unit tests
--------------
Run this from the root directory of the repository:
```
$ python3 test/test_videosum.py
```


Exemplary result
----------------

* Exemplary video: [here](https://raw.githubusercontent.com/luiscarlosgph/videosum/main/test/data/video.mp4)

* Summary based on `time` algorithm: 

`$ python3 -m videosum.run --input video.mp4 --output time.png --nframes 16 --height 1080 --width 1920 --algo time`

![](https://github.com/luiscarlosgph/videosum/blob/main/test/data/time.jpg) 

* Summary based on `inception` algorithm:

`$ python3 -m videosum.run --input video.mp4 --output time.png --nframes 16 --height 1080 --width 1920 --algo inception`

![](https://github.com/luiscarlosgph/videosum/blob/main/test/data/inception.jpg) 

* Summary based on `fid` algorithm:

`$ python3 -m videosum.run --input video.mp4 --output time.png --nframes 16 --height 1080 --width 1920 --algo fid`

![](https://github.com/luiscarlosgph/videosum/blob/main/test/data/fid.jpg) 

* Summary based on `scda` algorithm:

`$ python3 -m videosum.run --input video.mp4 --output time.png --nframes 16 --height 1080 --width 1920 --algo scda`

![](https://github.com/luiscarlosgph/videosum/blob/main/test/data/scda.jpg) 


Timing
------

| Method    | Time (s) |
| --------- | -------- |
| time      | 0.027    |
| inception | 0.668    |
| fid       | 2.089    |
| scda      | 0.583    |

`Time (s)` indicates how many seconds of summarisation are required for each second of raw video. 
<!-- This way you can easily approximate how long each algorithm will take to summarise a video of any given length. -->

Author
------
Luis Carlos Garcia Peraza Herrera (luiscarlos.gph@gmail.com), 2022.


License
-------

This code repository is shared under an [MIT license](https://github.com/luiscarlosgph/videosum/blob/main/LICENSE).

