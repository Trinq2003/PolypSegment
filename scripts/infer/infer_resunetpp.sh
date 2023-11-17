python infer.py \
    --model ResUnetPlusPlus \
    --optimizer Adam \
    --checkpoint-path ./checkpoints/resunetpp_model.pth \
    --pretrained-path ./saved_models/ \
    --infer-path ./predicted_masks/ \
    --lr 2e-4 \
    --batch-size 4 
