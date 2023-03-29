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


# python five_dataset_mask.py  # 增加动物数据集

# python five_dataset_mask_random_box.py
python occupy.py