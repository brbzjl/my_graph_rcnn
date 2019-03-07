# Graph R-CNN

### Introduction

This is the pytorch implementation of our graph rcnn model. It completes three tasks: object detection, attribute recognition and relation detection for each image jointly, and obtain the graph representations meanwhile. Besides a good performance on these three seperate tasks, it is also expected to help a bunch of high level (downstream) tasks, such as image captioning, visual question answering, expression reference, etc. It is developed based on the following two projects:

1. [jwyang/faster-rcnn.pytorch](https://github.com/jwyang/faster-rcnn.pytorch), a pure, fast and memory efficient pytorch implementation of faster-rcnn.
2. [peteanderson80/bottom-up-attention](https://github.com/peteanderson80/bottom-up-attention), a more sophisticated visual representation for image captioning and visual question answering based on bottom-up attention model.
3. [danfeiX/scene-graph-TF-release](https://github.com/danfeiX/scene-graph-TF-release), a secene graph detection model based on iterative message passsing.

### Installations

### Preparation

#### Dataset

Download [Visual Genome](http://visualgenome.org). For this project, you will need:

1. Images ([part1](https://cs.stanford.edu/people/rak248/VG_100K_2/images.zip), [part2](https://cs.stanford.edu/people/rak248/VG_100K_2/images2.zip))
2. [Metadata](http://visualgenome.org/static/data/dataset/image_data.json.zip)
3. [Object Annotations](http://visualgenome.org/static/data/dataset/objects.json.zip), [Object Alias](http://visualgenome.org/static/data/dataset/object_alias.txt)
4. [Attribute Annotations](http://visualgenome.org/static/data/dataset/attributes.json.zip)
5. [Relation Annotations](http://visualgenome.org/static/data/dataset/relationships.json.zip), [Relation Alias](http://visualgenome.org/static/data/dataset/relationship_alias.txt)
6. [Scene Graph](http://visualgenome.org/static/data/dataset/scene_graphs.json.zip)

After downloading the above data, unzip all of them and put into a single folder. Then make a soft link to it via:
```
cd $REPO_ROOT
ln -s PATH/TO/YOUR/DATA_DIR data/vg
```

Then, prepare the pascal voc like xmls for visual genome using:
```
cd $REPO_ROOT
python data/genome/setup_vg.py
```

#### Pretrained Models

Create a folder:
``` 
cd $REPO_ROOT
mkdir data/pretrained_model
```

Download pretrained resnet101 and vgg16 models and put them into the folder.

### Training

To train a resnet101, run:
```
CUDA_VISIBLE_DEVICES=0 python trainval_grcnn.py --dataset vg1 --net res101
```
Alternatively, to train a vgg16, run:
```
CUDA_VISIBLE_DEVICES=0 python trainval_grcnn.py --dataset vg1 --net vgg16
```

### Evaluation

```
CUDA_VISIBLE_DEVICES=0 python test_grcnn.py --dataset vg1 --net res101 --checksession 1  --checkepoch 11 --checkpoint 6224 --imdb test --mGPUs True
```
