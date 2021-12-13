import glob
import wandb
from main import start


sample_sizes = [500] #500,750,1000
proportions = [0.1]#0.1,0.3,0.7,0.9

for sample_size in sample_sizes:
    for prop in proportions:

        datasets = glob.glob('./datasets/n_'+str(sample_size)+'_p_'+str(prop).replace('.','_')+'/*.csv')

        sweep_config = {
            'method': 'grid',
            'parameters': {
                'TRAIN_FILE': {
                    'values': datasets
                },
                'PROPORTION': {
                    'values': [prop]
                },
                'SAMPLE_SIZE': {
                    'values': [sample_size]
                }

            }
        }

        sweep_id = wandb.sweep(sweep_config)
        wandb.agent(sweep_id, function=start)
