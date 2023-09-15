# SAM-with-point-match
This is the code for paper "Segment Anything with One Shot for Semantic SLAM Using Sparse Point Matching".

## Requirements
To install pytorch, use:
```
pip install torch==1.12.1+cu102 torchvision==0.13.1+cu102 --extra-index-url https://download.pytorch.org/whl/cu102
```
Other packages required are:
```
opencv-python
einops
scikit-learn-extra
pycocotools
scikit_image
timm
```

Installed segment anything model or its mobile variant is also required. Our paper uses MobileSAM:
```
git clone git@github.com:ChaoningZhang/MobileSAM.git
cd MobileSAM
git checkout e08982f
pip install -e .
```

## Evaluation
1. Download KITTI MOTS.
2. Download model parameters of MobileSAM and Pips(if compare with SAM-PT).
3. Install TrackEval: https://github.com/JonathonLuiten/TrackEval
4. Specify the data path and code path in `config/defaut.yaml` and `track_eval.py`.
5. Run `python track_eval.py`.

## Run ROS segmentation node
1. Our node is implemented for Kimera. Kimera's installation reference can be found in https://github.com/MIT-SPARK/Kimera. 
2. Eunsure that ROS Python is installed. 
3. Run `python node_select.py`.