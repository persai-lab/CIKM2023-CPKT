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

