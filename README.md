# Disentangled-Open-SEt-Recognition (DOSER)

Official pytorch implementation of Heejeong Nam, 2023, **"Enhanced Open Set Recognition via Disentangled Representation Learning",** _The 4th Korea Artificial Intelligence Conference_

### Environment setting and Requirements

```
conda create -n DOSER python=3.8 -y
conda activate DOSER
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch -y
pip install -U scikit-learn
```

### Git clone repo

```
git clone https://github.com/Hazel-Heejeong-Nam/DOSER.git
```

Our code hierarchy would be look like

```
DOSER
├── models
│      ├── __init__.py
│      ├──  base.py
│      ├──  doser.py
│      └── proser.py
├── utils
│      ├── __init__.py
│      ├── data.py
│      ├── util.py
│      └── wide_resnet.py
├── main.py
├── split.py
└── train.py
```

### Train model

Possible command
``` bash
python main.py --dataset mnist --backbone Toy --num_fold 5 --param_schedule 3 --param_step 0.03 --lr 0.003 --lambda1 0 --a 0.4 --b 1 --c 1 --d 0.5
```

### Notification 

- If there are any questions, feel free to contact with the author : Heejeong Nam (hatbi2000@yonsei.ac.kr)
- We refer to [Here](https://github.com/wjun0830/Difficulty-Aware-Simulator.git) in data split and loader





