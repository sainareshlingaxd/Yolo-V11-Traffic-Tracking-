# Python Traffic Counter

The purpose of this project is to detect and track vehicles on a video stream and count those going through a defined line. 

It uses:

* [YOLO](https://www.pyimagesearch.com/2018/11/12/yolo-object-detection-with-opencv) to detect objects on each of the video frames.

* [SORT](https://github.com/abewley/sort) to track those objects over different frames.

Once the objects are detected and tracked over different frames a simple mathematical calculation is applied to count the intersections between the vehicles previous and current frame positions with a defined line.

This is just a prototype and yet to be refined for real-time purposes.
