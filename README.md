using YOLOv8 only P1, P2 
small Object Detection inference code 

### Installation

```
conda create -n YOLO python=3.8
conda activate YOLO
conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch-lts
pip install opencv-python==4.5.5.64
pip install PyYAML
pip install tqdm
pip install cv2
pip install einops
```

### inference

* Configure your dataset path in `inference.py` for testing
* Run `python inference.py` for testing

### Results

| Version | Epochs | Box mAP |
|:-------:|:------:|--------:|
|  v8_n   |  500   |   92.4  |


### Dataset structure

    ├── data
        ├── sem
            ├── images
                ├── all
                    ├── xx_nnnn_nnnnn.tif
                    ├── xx_nnnn_nnnnn.tif
                ├── defect
                    ├── xx_nnnn_nnnnn.tif
                    ├── xx_nnnn_nnnnn.tif
            ├── yolo_annotations
                ├── all
                    ├── xx_nnnn_nnnnn.tif
                    ├── xx_nnnn_nnnnn.tif
                ├── defect
                    ├── xx_nnnn_nnnnn.tif
                    ├── xx_nnnn_nnnnn.tif

#### Reference

* https://github.com/ultralytics/yolov5
* https://github.com/ultralytics/ultralytics
