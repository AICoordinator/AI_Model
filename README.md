# AI_Model
## Run model

Before running the model, make sure that sample images are in `./samples`. Also, download the             model checkpoint from [here](https://drive.google.com/file/d/1Icy_iuM68WlRdPFS9CIERwmGdTOPe3iH/view?usp=sharing) and place it in `./checkpoints`. To predict the attractiveness score, run:

```python
python3 test.py --gpu_ids 0 --test_dataroot ./samples --output_dir ./output/your_root --ckpt ./checkpoints/8000.pth
```

The results are saved in `./output/your_root/output.txt`.

## Results
### Images with the highest attractiveness scores
![Untitled](./attractiveness/assets/top.jpg)
### Images with the lowest attractiveness scores
![Untitled](./attractiveness/assets/bottom.jpg)

## run_test
Call run_test() in run.py.