databin=path_to_your_databin
save_path=path_to_save_your_checkpoint
log=path_to_your_training_log

fairseq-train ${databin} \
  --save-dir ${save_path} \
  --ddp-backend=no_c10d  --fp16 \
  --task translation_lev \
  --criterion nat_loss \
  --arch cmlm_transformer \
  --encoder-embed-dim 1024 \
  --encoder-ffn-embed-dim 4096 \
  --encoder-attention-heads 16 \
  --encoder-layers 12 \
  --decoder-embed-dim 256 \
  --decoder-ffn-embed-dim 1024 \
  --decoder-attention-heads 4 \
  --decoder-layers 3 \
  --label-smoothing 0.1 \
  --attention-dropout 0.0 \
  --activation-dropout 0.0 \
  --dropout 0.1 \
  --noise random_mask \
  --share-decoder-input-output-embed \
  --optimizer adam --adam-betas '(0.9,0.98)' \
  --lr 3e-4 --lr-scheduler cosine \
  --warmup-init-lr 1e-07 --warmup-updates 30000 --lr-shrink 1 --lr-period-updates 270000 \
  --max-update 300000 \
  --weight-decay 0.0 --clip-norm 0.1 \
  --max-tokens 15000 --update-freq 4 \
  --apply-bert-init \
  --save-interval-updates 2000 \
  --no-progress-bar --log-format 'simple' --log-interval 100 \
  --fixed-validation-seed 7 \
  --seed 1 \
  --keep-last-epochs 0 \
  --fp16-scale-tolerance 0.1 > ${log} 2>&1