import torch
import torch.nn as nn
from model import ResNet18
from read_dataset import get_test_dataloader
from testing_uncertainty import regular_test
import glob
import pandas as pd


# Dataloader
test_loader = get_test_dataloader('./datasets/test_set.csv', 224, 1)

# Criterion
criterion = nn.CrossEntropyLoss()

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Weights
sample_sizes = [500, 750, 1000]
proportions = ['0_1', '0_3', '0_7', '0_9']

# Folder
folder = './weights_rotation/'


for sample in sample_sizes:
    for prop in proportions:

        # Classification Model load up
        print('==> Loading Classification Model')
        filew = folder+'n_'+str(sample)+'_p_'+prop+'/*/*.pth'
        # print(filew)
        result_df = pd.DataFrame()
                
        weights = glob.glob(filew)
        for w in weights:
            print('\n', w,'\n')
            model_class = ResNet18(name='test',num_classes=2)
            model_class.build_model()
            model_class = model_class.model
            model_class.load_state_dict(torch.load(w))

            # test(model=model_class, dataloader=test_loader, criterion=criterion, device=device, filew='./results/n_'+str(sample)+'_0_'+prop+'.csv')

            model_class.to(device)
            model_class.eval()

            _, _, _, _, _, metrics = regular_test(model_class, test_loader, criterion, device)

            df_metrics = pd.DataFrame.from_dict(metrics)
            df_metrics['Dataset'] = w.split('/')[-2]

            result_df = pd.concat([result_df, df_metrics], axis=0, sort=False)

        result_df.to_csv('./results/n_'+str(sample)+'_p_'+prop+'.csv')

