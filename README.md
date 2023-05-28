# Revisiting Non-Autoregressive Translation at Scale
This is the implementaion of our [paper](https://arxiv.org/abs/2305.16155):
```
Revisiting Non-Autoregressive Translation at Scale
Zhihao Wang, Longyue Wang, Jinsong Su, Junfeng Yao, Zhaopeng Tu
ACL 2023 (long paper, findings)
```
## Requirements
* Python3.6
* Pytorch

## Model Training
We validate two advanced models, representing iterative and fully NAT respectively:
* [MaskT](Ghazvininejad et al., 2019) 
* [GLAT] (Qian et al., 2021) 
The projects for MaskT and GLAT are avaliable in [here](), while the training scripts for MaskT and GLAT are avaliable in [here](). 

## Evaluation
For fair comparison, we use case-insensitive tokenBLEU to measure the translation quality on WMT16 En-Ro and WMT14 En-De. We use SacreBLEU for the new benchmark WMT20 En-De.
### TokenBLEU
grep ^H fairseq.log | awk -F '\t' '{print $NF}' | perl -ple 's{(\S)-(\S)}{$1 ##AT##-##AT## $2}g' > test.sys

grep ^T fairseq.log | cut -f2- | perl -ple 's{(\S)-(\S)}{$1 ##AT##-##AT## $2}g' > test.ref

fairseq-score --sys test.sys --ref test.ref > test.compound_bleu
### SacreBLEU
cat fairseq.log | grep -P "^H" |sort -V |cut -f 3- > sys.ordered

mosesdecoder/scripts/tokenizer/detokenizer.perl -l en -penn < sys.ordered > sys.detoken

sacrebleu -t wmt20 -l en-de --detail < sys.detoken >  sys.sacre_bleu

##
The translations of different NAT models are listed in [here]().

## Citation
```
@inproceedings{scaling4nat,
  title={Revisiting Non-Autoregressive Translation at Scale},
  author={Wang, Zhihao and
          Wang, Longyue and
          Su, Jinsong and
          Yao, Junfeng and
          Tu, Zhaopeng},
  booktitle={ACL Findings},
  year={2023}
}
```
