databin=path_to_your_databin
save_path=path_to_save_your_checkpoint
log=path_to_your_training_log

fairseq-train ${databin} \
  --save-dir ${save_path} \
  --ddp-backend=no_c10d --fp16 \
  --task translation_lev \
  --criterion nat_loss \
  --arch cmlm_transformer \
  --encoder-embed-dim 1024 \
  --encoder-ffn-embed-dim 4096 \
  --decoder-embed-dim 1024 \
  --decoder-ffn-embed-dim 4096 \
  --encoder-attention-heads 16 \
  --decoder-attention-heads 16 \
  --encoder-layers 6 \
  --decoder-layers 6 \
  --label-smoothing 0.1 \
  --attention-dropout 0.3 \
  --activation-dropout 0.3 \
  --dropout 0.3 \
  --noise random_mask \
  --share-all-embeddings \
  --optimizer adam --adam-betas '(0.9,0.98)' \
  --lr 1e-3 --lr-scheduler cosine \
  --warmup-init-lr 1e-07 --warmup-updates 4000 --lr-shrink 1 --lr-period-updates 26000 \
  --max-update 30000 \
  --weight-decay 0.0 --clip-norm 0.1 \
  --max-tokens 20000 --update-freq 3 \
  --apply-bert-init \
  --no-progress-bar --log-format 'simple' --log-interval 100 \
  --fixed-validation-seed 7 \
  --seed 1 \
  --save-interval-updates 500 \
  --no-epoch-checkpoints \
  --keep-last-epochs 0 \
  --fp16-scale-tolerance 0.1 > ${log} 2>&1