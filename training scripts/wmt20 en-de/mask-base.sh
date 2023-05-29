databin=path_to_your_databin
save_path=path_to_save_your_checkpoint
log=path_to_your_training_log

fairseq-train ${databin} \
  --save-dir ${save_path} \
  --ddp-backend=no_c10d --fp16 \
  --task translation_lev \
  --criterion nat_loss \
  --arch cmlm_transformer \
  --encoder-embed-dim 512 \
  --encoder-ffn-embed-dim 2048 \
  --decoder-embed-dim 512 \
  --decoder-ffn-embed-dim 2048 \
  --encoder-attention-heads 8 \
  --decoder-attention-heads 8 \
  --encoder-layers 6 \
  --decoder-layers 6 \
  --label-smoothing 0.1 \
  --attention-dropout 0.0 \
  --activation-dropout 0.0 \
  --dropout 0 \
  --noise random_mask \
  --share-all-embeddings \
  --optimizer adam --adam-betas '(0.9,0.98)' \
  --lr 7e-4 --lr-scheduler cosine \
  --warmup-init-lr 1e-07 --warmup-updates 30000 --lr-shrink 1 --lr-period-updates 270000 \
  --max-update 300000 \
  --weight-decay 0.0 --clip-norm 0.1 \
  --max-tokens 30000 --update-freq 2 \
  --apply-bert-init \
  --no-progress-bar --log-format 'simple' --log-interval 100 \
  --fixed-validation-seed 7 \
  --seed 1 \
  --save-interval-updates 2000 \
  --keep-last-epochs 0 \
  --fp16-scale-tolerance 0.1 > ${log} 2>&1