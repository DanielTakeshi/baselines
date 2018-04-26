import baselines.asym_ddpg.main as main
from hyperopt import STATUS_OK

import re
def slugify(s):
    """
    Normalizes string, converts to lowercase, removes non-alpha characters,
    and converts spaces to hyphens.
    """
    s = str(s).strip().replace(' ', '_')
    return re.sub(r'(?u)[^-\w.]', '', s)
def wrap_run(args):
    print (str(args))
    kwargs={}
    kwargs["env_id"] = 'MicoEnv-pusher-sparse-v1'
    kwargs['render_eval'] = False 
    kwargs['render_demo'] = False
    kwargs['layer_norm'] =True
    kwargs['render'] =False
    kwargs['normalize_returns'] =False
    kwargs['normalize_observations'] =True
    kwargs['normalize_aux'] =True
    kwargs['seed'] = 0
    kwargs['critic_l2_reg'] = args["l2"]
    kwargs['batch_size'] = args["batch_size"]
    kwargs['actor_lr'] = args["lr"] / 10
    kwargs['critic_lr'] = args["lr"]
    kwargs['popart'] = False
    kwargs['gamma'] = args["gamma"]
    kwargs['reward_scale'] =1
    kwargs['clip_norm'] = None
    kwargs['nb_epochs'] = 30
    kwargs['nb_epoch_cycles'] = 20
    kwargs['nb_train_steps'] = 50
    kwargs['nb_eval_steps'] = 400
    kwargs['nb_rollout_steps'] = 100   # per epoch cycle and MPI worker
    kwargs['noise_type'] = args["noise"]
    kwargs['load_from_file'] = False
    kwargs['num_demo_steps']  = 2000
    kwargs['num_pretrain_steps'] =1000
    kwargs['run_name'] = slugify(str(args))
    kwargs['demo_policy'] ='None'
    kwargs['lambda_pretrain'] = args["lambda_pretrain"]
    kwargs['lambda_nstep'] = args["lambda_nstep"]
    kwargs['lambda_1step'] = args["lambda_1step"]
    kwargs['replay_beta'] = 0.4
    kwargs['reset_to_demo_rate'] = args["reset_to_demo_rate"]
    kwargs['tau'] = 0.01
    kwargs['evaluation'] =  True
    kwargs['demo_policy'] =  "pusher"
    loss =  main.run(**kwargs)
    return {'loss': loss, 'status': STATUS_OK }


if __name__ == '__main__':
    wrap_run({"lr": 1e-3, "gamma":0.999})

