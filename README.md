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
  * `--algo`: algorithm used to select the key frames of the video.
    * `time`: evenly spaced frames are selected.
    * `inception`: medoids retrieved with k-medoids clustering on InceptionV3 latent space vectors corresponding to each video frame. The clustering metric used to compute the distance between two latent vectors is the l2-norm.
    * `fid` : medoids retrieved with k-medoids clustering on InceptionV3 latent space vectors corresponding to each video frame. The clustering metric used to compute the distance between two latent vectors is the [Frechet Inception Distance (FID)](https://en.wikipedia.org/wiki/Fr%C3%A9chet_inception_distance).
    * `scda`: medoids retrieved with k-medoids clustering on SCDA image descriptors ([Wei et al. 2017 Selective Convolutional Descriptor Aggregation for Fine-Grained Image Retrieval](https://arxiv.org/abs/1604.04994)), but with latent tensor from InceptionV3 trained on ImageNet as opposed to VGG-16.


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
algo = 'fid'  # The options are: 'time', 'inception', 'fid', 'scda'

# Create video summariser object
vs = videosum.VideoSummariser(algo, nframes, width, height)

# Create collage image
im = vs.summarise('video.mp4')

# Save collage to file
cv2.imwrite('collage.jpg', im)

# Retrieve a list of Numpy/OpenCV BGR images corresponding to the key frames of the video
key_frames = vs.get_key_frames('video.mp4')       
```


Run unit tests
--------------
Run this from the root directory of the repository:
```
$ python3 test/test_videosum.py
```


Exemplary result
----------------

* Video (click on the thumbnail to watch the video in Youtube):

[![Exemplary surgery video](https://img.youtube.com/vi/45dRNoqGZCg/0.jpg)](https://www.youtube.com/watch?v=45dRNoqGZCg)

* Summary based on `time` algorithm: 

`$ python3 -m videosum.run --input video.mp4 --output time.png --nframes 16 --height 1080 --width 1920 --algo time`

![](https://github.com/luiscarlosgph/videosum/blob/main/test/data/time.png) 

* Summary based on `fid` algorithm:

`$ python3 -m videosum.run --input video.mp4 --output time.png --nframes 16 --height 1080 --width 1920 --algo fid`

![](https://github.com/luiscarlosgph/videosum/blob/main/test/data/fid.png) 

* Summary based on `scda` algorithm:

`$ python3 -m videosum.run --input video.mp4 --output time.png --nframes 16 --height 1080 --width 1920 --algo scda`

![](https://github.com/luiscarlosgph/videosum/blob/main/test/data/scda.png) 


Author
------
Luis Carlos Garcia Peraza Herrera (luiscarlos.gph@gmail.com), 2022.


License
-------

This code repository is shared under an [MIT license](https://github.com/luiscarlosgph/videosum/blob/main/LICENSE).

