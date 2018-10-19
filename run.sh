#!/bin/bash

#DIR="$( cd "$( dirname "$0"  )" && pwd  )"
DIR=$PWD
cd $DIR
echo current dir is $PWD

export PYTHONPATH=$PYTHONPATH:$DIR:$DIR/slim:$DIR/object_detection

ls /data/weixin-42679665/w8-data-wuzhong

output_dir=/output
dataset_dir=/data/weixin-42679665/w8-data-wuzhong

train_dir=$output_dir/train
checkpoint_dir=$train_dir
eval_dir=$output_dir/eval

config=ssd_mobilenet_v1_pets.config
pipeline_config_path=$output_dir/$config

cp ./$config $pipeline_config_path

for i in {0..4}
do
    echo "############" $i "runnning #################"
    last=$[$i*100]
    current=$[($i+1)*100]
    sed -i "s/^  num_steps: $last$/  num_steps: $current/g" $pipeline_config_path
    more $pipeline_config_path

    echo "############" $i "training #################"
    echo "./object_detection/train.py --train_dir=$train_dir --pipeline_config_path=$pipeline_config_path"
    ls $train_dir

    python ./object_detection/train.py --train_dir=$train_dir --pipeline_config_path=$pipeline_config_path

    echo "############" $i "evaluating, this takes a long while #################"
    python ./object_detection/eval.py --checkpoint_dir=$checkpoint_dir --eval_dir=$eval_dir --pipeline_config_path=$pipeline_config_path
done

python ./object_detection/export_inference_graph.py --input_type image_tensor --pipeline_config_path $pipeline_config_path --trained_checkpoint_prefix $train_dir/model.ckpt-$current  --output_directory $output_dir/exported_graphs

python ./inference.py --output_dir=$output_dir --dataset_dir=$dataset_dir
