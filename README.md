## Command to run regular MoE (full gating function):
```cd rl```
```python main.py --num_steps=[training steps] --use_moe --load_model=[path to regular_moe file] --run_name=[name of run for logging]```

## Command to run regular MoE (full gating function):
```cd rl```
```python main.py --num_steps=[training steps] --use_ee_moe --load_model=[path to moe_equal_experts file] --run_name=[name of run for logging]```

To train from scratch just don't include the load model parameter
