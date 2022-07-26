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
$ python3 -m videosum.run --input video.mp4 --output collage.jpg --nframes 100 --height 1080 -width 1920
```


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
algo = 'fid'  # The options are: 'time', 'fid'

# Create video summariser object
vs = videosum.VideoSummariser(algo, nframes, width, height)

# Create collage image
im = vs.summarise('video.mp4')

# Save collage to file
cv2.imwrite('collage.jpg', im)

# Retrieve a list of Numpy/OpenCV BGR images corresponding to the key frames of the video
key_frames = vs.get_key_frames('video.mp4')       # Uses the algorithm passed in the constructor

# Alternatively, you can specify which algorithm you want to use
key_frames = vs.get_key_frames_time('video.mp4')  # 'time' algo
key_frames = vs.get_key_frames_fid('video.mp4')   # 'fid'  algo
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


Author
------
Luis Carlos Garcia Peraza Herrera (luiscarlos.gph@gmail.com), 2022.


License
-------

This code repository is shared under an [MIT license](https://github.com/luiscarlosgph/videosum/blob/main/LICENSE).

