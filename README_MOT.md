# Multi Object Tracker

## Getting started

The following projects are used for this Multi Object Tracker: HRNet/DEKR and pyvision from Lennart Bastian (visualize_reid folder). Make sure to follow the instructions on how to install them. Also, don't forget to change the paths within the inference.py file. Use the following command to run the Multi Object Tracker:

```
python tools/inference_demo.py --cfg experiments/coco/inference_demo_coco.yaml --videoFile /path/to/video --outputDir /path/to/output/dir --visthre 0.3 TEST.MODEL_FILE model/pose_coco/pose_dekr_hrnetw32_coco.pth
```

Checkout the example_outputs folder for some results on the Atlas data set.

