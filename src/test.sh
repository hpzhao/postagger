#!/bin/bash
python universal_postagger.py --emb_path ../data/cs_cltt.small-100.vectors  --data_path ../data --train_file cs_cltt-ud-train.conllu --dev_file cs_cltt-ud-dev.conllu --cluster_path ../data/cs_cltt.cluster --language cs_cltt --is_training 0  --model ../data/cs_cltt.upos.model
