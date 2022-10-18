## HGCAN: The source code of a paper that sumbmitted for ICDE and under reviewing
# We have used three public datasets in our paper, you can download the raw datasets from following url:

- DIG: <http://cikm2016.cs.iupui.edu/cikm-cup> 

- YC: <http://2015.recsyschallenge.com/challenge.html>

- JD: <https://jdata.jd.com/html/detail.html?id=8>

# We have provided the pre-processed datasets to support running our codes, you can find them in `datasets` directory.

## How to run this code

Firstly, we run the file `build_graph.py` to generate the graph files (We have provided the graph files in `datasets` directory).

Then we run the file `main.py` to train the model and obtain the performance.

Take DIG dataset as example:
```
python build_graph.py --dataset DIG
python main.py --dataset DIG --batch_size 100 --cate_select 5 --dice_threshold 1.0
```
## Requirements

- Python3
- pytorch==1.9.0+cu111
