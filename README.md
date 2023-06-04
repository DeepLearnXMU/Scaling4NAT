# Revisiting Non-Autoregressive Translation at Scale
This is the implementaion of our [paper](https://arxiv.org/abs/2305.16155):
```
Revisiting Non-Autoregressive Translation at Scale
Zhihao Wang, Longyue Wang, Jinsong Su, Junfeng Yao, Zhaopeng Tu
ACL 2023 (long paper, findings)
```
## Requirements
* Python >= 3.6
* Pytorch >= 1.7.1
* Sacrebleu
* Mosesdecoder

## Model Training
We validate two advanced models, representing iterative and fully NAT respectively:
* [MaskT](https://github.com/facebookresearch/fairseq) 
* [GLAT](https://github.com/FLC777/GLAT)

The projects for MaskT and GLAT are avaliable in [here](https://github.com/DeepLearnXMU/Scaling4NAT/tree/main/projects), while the training scripts for MaskT and GLAT are avaliable in [here](https://github.com/DeepLearnXMU/Scaling4NAT/tree/main/training%20scripts). 

## Evaluation
We evaluate the performance on an ensemble of 5 best checkpoints (ranked by validation BLEU). For fair comparison, we use case-insensitive tokenBLEU to measure the translation quality on WMT16 En-Ro and WMT14 En-De. We use SacreBLEU for the new benchmark WMT20 En-De.

### MaskT
```
databin=path_to_your_databin
model_path=path_to_your_checkpoint
log=path_to_your_generation_log

fairseq-generate ${databin} \
  --gen-subset test \
  --task translation_lev \
  --iter-decode-max-iter 9 \
  --remove-bpe \
  --iter-decode-with-beam 5 \
  --max-tokens 1000 \
  --path ${model_path} \
  --iter-decode-force-max-iter > ${log} 2>&1
```

### GLAT
```
databin=path_to_your_databin
model_path=path_to_your_checkpoint
log=path_to_your_generation_log

fairseq-generate ${databin} \
  --user-dir glat_plugins \
  --gen-subset test \
  --task translation_lev_modified \
  --path ${model_path} \
  --iter-decode-max-iter 0 \
  --iter-decode-eos-penalty 0 \
  --remove-bpe \
  --print-step \
  --iter-decode-with-beam 1 \
  --max-tokens 5000 \
  --iter-decode-force-max-iter > ${log} 2>&1 &
```

### TokenBLEU
```
log=path_to_your_fairseq_generation_log
sys=path_to_your_sys
ref=path_to_your_ref
compoundbleu=path_to_your_bleu_file

grep ^H ${log} | awk -F '\t' '{print $NF}' | perl -ple 's{(\S)-(\S)}{$1 ##AT##-##AT## $2}g' > ${sys}
grep ^T ${log} | cut -f2- | perl -ple 's{(\S)-(\S)}{$1 ##AT##-##AT## $2}g' > ${ref}
fairseq-score --sys ${sys} --ref ${ref} > ${compoundbleu}
```
### SacreBLEU
```
log=path_to_your_fairseq_generation_log
ordered=path_to_your_ordered_output
detoken=path_to_your_detoken_output
sacrebleu=path_to_your_bleu_file

cat ${log | grep -P "^H" |sort -V |cut -f 3- > ${ordered}
mosesdecoder/scripts/tokenizer/detokenizer.perl -l en -penn < ${ordered} > ${detoken}
sacrebleu -t wmt20 -l en-de --detail < ${detoken} >  ${sacrebleu}
```

## Translations
The translations of different NAT models are listed in [here](https://github.com/DeepLearnXMU/Scaling4NAT/tree/main/translations).

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
