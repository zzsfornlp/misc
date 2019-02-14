#!/usr/bin/env bash

# cnn encoder
ii=0

for dec_type in none lstm gru; do
for enc_layer in 1 2; do
for scorer_layer in 0 1; do
for lrate in 0.0002 0.0005 0.0008; do
for enc_type in cnn lstm; do
echo RUN with dec_type:$dec_type enc_type:$enc_type enc_layer:$enc_layer scorer_layer:$scorer_layer lrate:$lrate
python3 -u ../hw1/hw1.py train train:data/topicclass_train.txt dev:data/topicclass_valid.txt enc_type:$enc_type enc_layer:$enc_layer dec_type:$dec_type lrate:$lrate scorer_layer:$scorer_layer anneal_restarts:5 patience:5 pretrain:./data/emb.txt lrate_decay_valid:0.98 max_epochs:10 validate_freq:1000 model:"models/model${ii}"
python3 -u ../hw1/hw1.py test test:data/topicclass_test.txt enc_type:$enc_type enc_layer:$enc_layer dec_type:$dec_type scorer_layer:$scorer_layer model:"models/model${ii}"
(( ii++ ))
done
done
done
done
done
