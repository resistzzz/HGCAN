## HGCAN: The source code of a paper for ICDE 2023
## We have used three public datasets in our paper as follows: 

You can download the raw datasets from following url:

- DIG: <http://cikm2016.cs.iupui.edu/cikm-cup> 

- YC: <http://2015.recsyschallenge.com/challenge.html>

- JD: <https://jdata.jd.com/html/detail.html?id=8>

We have provided the pre-processed datasets to support running our codes, you can find them in `datasets` directory.

## How to run this code

Firstly, we run the file `build_graph.py` to generate the graph files.

(We have provided the generated graph files in `datasets` directory).

Then we run the file `main.py` to train the model and simulate the conversations to obtain the final performance.

Take DIG dataset as example:
```
python build_graph.py --dataset DIG
python main.py --dataset DIG --batch_size 100 --cate_select 5 --dice_threshold 1.0
```
## Requirements

- Python 3.6
- pytorch==1.9.0+cu111

After you have configured your environment, you can run this code directly to obtain experimental results.
