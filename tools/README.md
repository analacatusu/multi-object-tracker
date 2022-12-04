# Multi Object Tracker

## Getting started

This project is a clone of the HRNet/DEKR one. Make sure to follow the instructions on how to pre-download the COCO dataset and also change the paths within the inference.py file to the path of your data set. Use the following command to run the Multi Object Tracker:

```
python tools/inference_demo.py --cfg experiments/coco/inference_demo_coco.yaml --videoFile /path/to/video --outputDir /path/to/output/dir --visthre 0.3 TEST.MODEL_FILE model/pose_coco/pose_dekr_hrnetw32_coco.pth
```

Checkout the example_outputs folder for some results on the Atlas data set.

