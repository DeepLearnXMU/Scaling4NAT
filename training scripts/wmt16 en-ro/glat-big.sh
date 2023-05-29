databin=path_to_your_databin
save_path=path_to_save_your_checkpoint
log=path_to_your_training_log

fairseq-train ${databin} \
  --save-dir ${save_path} --fp16 \
  --task translation_lev_modified --criterion glat_loss --arch glat --user-dir glat_plugins \
  --noise full_mask --share-all-embeddings \
  --encoder-embed-dim 1024 \
  --decoder-embed-dim 1024 \
  --encoder-ffn-embed-dim 4096 \
  --decoder-ffn-embed-dim 4096 \
  --encoder-attention-heads 16 \
  --decoder-attention-heads 16 \
  --encoder-layers 6 \
  --decoder-layers 6 \
  --lr 5e-4 --max-tokens 15000 --update-freq 4 \
  --max-update 30000 \
  --lr-scheduler inverse_sqrt --warmup-updates 4000 --optimizer adam --adam-betas '(0.9, 0.999)' \
  --label-smoothing 0.1 --warmup-init-lr 1e-7 --stop-min-lr 1e-9 \
  --adam-eps 1e-6 --weight-decay 0.01 --dropout 0.3   --attention-dropout 0.3 --activation-dropout 0.3 \
  --max-source-positions 1000 --max-target-positions 1000 --clip-norm 5\
  --src-embedding-copy --length-loss-factor 0.05 \
  --decoder-learned-pos --encoder-learned-pos \
  --apply-bert-init --activation-fn gelu \
  --no-progress-bar --log-format 'simple' --log-interval 100 \
  --fixed-validation-seed 7 \
  --seed 1 \
  --save-interval-updates 500 \
  --no-epoch-checkpoints \
  --keep-last-epochs 0 \
  --fp16-scale-tolerance 0.1 > ${log} 2>&1