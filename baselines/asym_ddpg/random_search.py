import baselines.asym_ddpg.main as main

import re
import random
import time
from pathlib import Path
from sheets_util import record_run_data
import uuid
home = str(Path.home())

def wrap_run():
    kwargs={}

    kwargs["dense_layer_size"] = random.choice([64, 128,  256])
    kwargs["num_dense_layers"] = random.choice([2,3,4])
    kwargs["actor_lr"] = random.choice([1e-3, 1e-4, 1e-5]) 
    kwargs["critic_lr"] = random.choice([1e-3, 1e-4, 1e-5]) 
    kwargs["tau"] = random.choice([1e-2, 1e-3, 1e-4]) 
    kwargs["num_critics"] = random.choice([1, 2]) 
    kwargs["lambda_nstep"] = random.choice([0., 0.3,0.6, 0.9]) 
    kwargs["lambda_pretrain"] = random.choice([1, 5, 10]) 
    kwargs["nsteps"] = random.choice([5, 10, 15]) 
    kwargs["lambda_1step"] = random.choice([0., 0.3, 0.6, 1]) 
    kwargs["layer_norm"] = random.choice([True, False]) 
    kwargs["critic_l2_reg"] = random.choice([0, 1e-1, 1e-2, 1e-3]) 
    kwargs["reward_type"] = random.choice(["sparse", "positive"]) 
    kwargs["pretrain_steps"] = random.choice([250, 1000, 2000]) 
    kwargs["replay_alpha"] = random.choice([0.5, 0.7, 0.9]) 
    kwargs["replay_beta"] = random.choice([0.3, 0.5, 0.7, 0.9]) 
    kwargs["demo_epsilon"] = random.choice([0.1, 0.2, 0.5, 1]) 
    kwargs["reset_to_demo_rate"] = random.choice([0.5, 0.7, 0.9]) 
    kwargs["noise_type"] = random.choice(["normal_0.05", "normal_0.1", "normal_0.2"]) 
    kwargs["target_policy_noise"] = random.choice([0.0, 0.01]) 
    kwargs["update_delay"] = random.choice([1,2]) 
    kwargs["batch_size"] = random.choice([32, 64]) 
    kwargs["use_velocities"] = random.choice([True, False]) 
    kwargs["reset_terminality"] = random.choice([3, 5, 10])
    kwargs["gamma"] = random.choice([0.99, 0.999]) 
    kwargs["normalize_aux"] = random.choice([True, False]) 
    kwargs["normalize_observations"] = random.choice([True, False]) 
 

    velos_id = "velos" if kwargs["use_velocities"] else "ik"

    kwargs["env_id"] = "ClothEnv-low_dim-{}-{}-det-v1".format(kwargs["reward_type"], velos_id)
    kwargs['render_eval'] = True
    kwargs['render_demo'] = True
    kwargs['render'] =False
    kwargs['normalize_returns'] =False
    kwargs['seed'] = 0


    kwargs['reward_scale'] =1

    
    kwargs['popart'] = False
    kwargs['clip_norm'] = None
    
    kwargs['nb_epochs'] = 20
    kwargs['nb_epoch_cycles'] = 20
    kwargs['nb_train_steps'] = 50
    kwargs['nb_eval_steps'] = 400
    
    kwargs['nb_rollout_steps'] = 100   # per epoch cycle and MPI worker
    kwargs['load_from_file'] = False
    kwargs['num_demo_steps']  = 20
    kwargs['run_name'] = str(uuid.uuid1())
    kwargs['demo_policy'] ='folder'
    kwargs['evaluation'] =  True

    kwargs['result'] =  main.run(**kwargs)
    kwargs['load_from_file'] = False
    import json
    record_run_data(kwargs)
    with open(home+'/param_search/'+kwargs['run_name']+".txt", 'w+') as outfile:
        json.dump(kwargs, outfile)
    print (kwargs)

if __name__ == '__main__':
    while True:
        try:
            wrap_run()
        except:
            raise
            print("!!!!!!!!!!!!!!!!!!!!Run failed!!!!!!!!!!!!")
            time.sleep(10)


