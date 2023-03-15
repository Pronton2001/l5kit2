# Issue
[] Object-based attention model do not train so well (attention multihead model is big and hard to learn) with PPO.
	- Float16
	- accumulate gradient
```
action in L5env at step() take first waypoints to move and do not care 11 next ones.
```
where is 'cur_speed'
batch_size = 1000 = rollout_fragment_length x num_workers x env
rollout_fragment_length = 15
sgd_minibatch_size = 256
num_workers = 63
env = 4

vs 
train_batch_size = 8000 = rollout_fragment_length x num_workers x env
rollout_fragment_length = 32
sgd_minibatch_size = 1024
num_workers = 63
env = 4
# Update
[x] Create L5env2: vectorize + transformer + 12 action output 
[x] Add multi-binary trong ray.rllib.utils.serialization.py
[ ] Create L5env3: vectorize + transformer + input(add route, history ego, agent, future ego) + 1 action output
[x] Change 'curr_speed' (Rasterized settings) to 'speed' in Vetorized setting.
[ ] Understand Transformer model
	[ ]: what is type embedding

