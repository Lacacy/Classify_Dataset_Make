#!/bin/bash

__conda_setup="$('/root/anaconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/root/anaconda3/etc/profile.d/conda.sh" ]; then
        . "/root/anaconda3/etc/profile.d/conda.sh"
    else
        export PATH="/root/anaconda3/bin:$PATH"
    fi
fi
unset __conda_setup
# <<< conda initialize <<<

/etc/init.d/ssh start



cd /wangkaixiong/model/Data_Manage/5分类模型数据预处理

# python make_org_5class_dataset.py
python five_dataset.py

# cp /wangkaixiong/dataset/five_class_final_all_dataset_exp_ADD_MASK/train/Mask/* /wangkaixiong/dataset/five_class_final_all_dataset_exp/train/Mask
# cp /wangkaixiong/dataset/five_class_final_all_dataset_exp_ADD_MASK/val/Mask/* /wangkaixiong/dataset/five_class_final_all_dataset_exp/val/Mask


# python five_dataset_mask_random_box.py