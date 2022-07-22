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

# Create video summariser object
vs = videosum.VideoSummariser(nframes, width, height)

# Create collage image
im = vs.summarise('video.mp4')

# Save collage to file
cv2.imwrite('collage.jpg', im)
```


Run unit tests
--------------
Run this from the root directory of the repository:
```
$ python3 test/test_videosum.py
```


Exemplary result
----------------

* Command:
```
python3 -m videosum.run --input test/data/test.mp4 --output test/data/test.png --nframes 9 --height 480 --width 640
```

* Video:

[![Exemplary surgery video](https://img.youtube.com/vi/45dRNoqGZCg/0.jpg)](https://www.youtube.com/watch?v=45dRNoqGZCg)

<!-- https://user-images.githubusercontent.com/3996630/172403028-4515cbec-0216-408c-99b8-c03cf15949af.mp4 -->

* Summary:

![](https://github.com/luiscarlosgph/videosum/blob/main/test/data/test.png) 


Author
------
Luis Carlos Garcia Peraza Herrera (luiscarlos.gph@gmail.com), 2022.


License
-------

This code repository is shared under an [MIT license](https://github.com/luiscarlosgph/videosum/blob/main/LICENSE).

