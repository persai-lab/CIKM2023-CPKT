# Lifelong Personalized Knowledge Tracing

## How to install and run 

For example, to run the DKT model:
```angular2html
cd LLPKT
pip install -e .
cd llpkt
python run.py -c configs/dkt_exp_0.json
```

Another way to run using conda:
```angular2html
cd LLPKT
conda env create -f environment.yml
source init_env.sh
cd llpkt
python run.py -c configs/dkt_exp_0.json
```

Code for our paper:

C. Wang, S. Sahebi, “Continuous personalized knowledge tracing: Modeling long-term learning in online environments,” In The 32nd ACM International Conference on Information and Knowledge Management (CIKM) 2023.


If you have any questions, please email cwang25@albany.edu


## Cite:

Please cite our paper if you use this code in your own work:

```
@inproceedings{wang2021dmkt,
  title={Continuous personalized knowledge tracing: Modeling long-term learning in online environments},
  author={Wang, Chunpai and Sahebi,Shaghayegh},
  booktitle={Proceedings of the 32nd ACM International Conference on Information and Knowledge Management (CIKM 2023)},
  year={2023}
}
```

## Acknowledgement:

This  paper is based upon work supported by the National Science Foundation under Grant No.1755910
