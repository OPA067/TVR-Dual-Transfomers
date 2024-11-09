#### Training
```python MSVD
python train.py --exp_name=MSVD-train --dataset_name=MSVD --log_step=10 --evals_per_epoch=1 --batch_size=8 --videos_dir=MSVD/videos/
```

```python MSRVTT-ViT-B/32
python train.py --exp_name=MSRVTT-train --dataset_name=MSRVTT --clip_arch=ViT-B/32 --log_step=50 --evals_per_epoch=5 --batch_size=32 --videos_dir=MSRVTT/videos/   
```

```python MSRVTT-ViT-B/16
python train.py --exp_name=MSRVTT-train --dataset_name=MSRVTT --clip_arch=ViT-B/16 --log_step=50 --evals_per_epoch=5 --batch_size=32 --videos_dir=MSRVTT/videos/
```

```python DiDeMo
python train.py --exp_name=DiDeMo-train --dataset_name=DiDeMo --log_step=10 --evals_per_epoch=1 --batch_size=8  --num_workers=0 --videos_dir=DiDeMo/videos/ 
```

```python Charades
python train.py --exp_name=Charades-train --dataset_name=Charades --num_frames=12 --log_step=10 --evals_per_epoch=1 --batch_size=8 --videos_dir=Charades/videos/ 
```

```python VATEX
python train.py --exp_name=VATEX-train --dataset_name=VATEX --log_step=10 --evals_per_epoch=1 --batch_size=32 --videos_dir=VATEX/videos  
```

```python LSMDC
python train.py --exp_name=LSMDC-train  --dataset_name=LSMDC --num_epochs=5 --num_frames=12 --log_step=10 --evals_per_epoch=1 --batch_size=32 --videos_dir=LSMDC  
```

#### Testing
```python
python test.py --exp_name=DiDeMo-test --dataset_name=DiDeMo --batch_size=8 --num_workers=0 --videos_dir=DiDeMo/videos/
```
