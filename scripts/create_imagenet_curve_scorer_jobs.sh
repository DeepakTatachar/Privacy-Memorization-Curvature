
for start_idx in {0..96..4}; do
    echo "python calc_curv_fz_imagenet_models --start_idx $start_idx --stop_idx $((start_idx+3))"
done