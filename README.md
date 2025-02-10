## Command to run experiment:
```cd rl```
```python main.py --cuda --run_name=[name for tensorboard logging] --load_model=[path to model checkpoint] --num_steps=1500000 --use_moe```

To load the "equal experts" model, *uncomment lines 117 and 118* in rl/mt_sac/mt_sac.py. These don't affect the experiment, but when I trained that model I'd accidentally left those lines in so they need to be in there for the pytorch dict to load. Conversely, the need to be re-commented to load the regular moe model.
