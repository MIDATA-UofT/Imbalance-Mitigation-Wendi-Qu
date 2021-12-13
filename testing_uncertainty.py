import torch
from torch.functional import F
import torch.nn as nn
from metrics import Metrics
import pandas as pd
import numpy as np

def regular_test(model, dataloader, criterion, device):
    
    metrics = Metrics()

    ids = []
    labels = []
    std_preds = []
    prob_0s = []
    prob_1s = []
    
    with torch.no_grad():
        for input, label in dataloader:
            # print(id_img[0])
            # Transforming inputs
            input, label = input.to(device), label.to(device)

            # Get predictions
            output = model(input)
            probs = F.softmax(output,dim=1).data.cpu().numpy()[0]
            _, predicted = torch.max(output.data, 1)

            # Get loss
            loss = criterion(output, label)

            # Register on metrics
            metrics.batch(labels=label, preds=predicted, loss=loss.item())

            # Logs
            # ids.append(id_img[0])
            # labels.append(label.item())
            # std_preds.append(predicted.item())
            # prob_0s.append(probs[0])
            # prob_1s.append(probs[1])

    metrics.print_summary()
    return ids, labels, std_preds, prob_0s, prob_1s, metrics.summary()

def turn_on_dropout(m):
    '''
    Layers with name nn.Dropout (for our transfer-learned ResNet, this is the 
    layer before the output one) will be enabled as in training.
    '''
    if type(m) == nn.Dropout:
        m.train()


def test_uncertainty(model, dataloader, criterion, device, dropout_t=10):

    # Turn On Dropout
    model.apply(turn_on_dropout)

    avg_preds = []
    var_preds = []
    avg_prob_0s = []
    avg_prob_1s = []
    var_probs = []

    with torch.no_grad():
        for id_img, input, label in dataloader:

            print(id_img[0])
            # Transforming inputs
            input, label = input.to(device), label.to(device)
            
            preds = []
            prob_0s = []
            prob_1s = []

            for d_t in range(dropout_t):
                # Get predictions
                output = model(input)
                probs = F.softmax(output,dim=1).data.cpu().numpy()[0]
                _, predicted = torch.max(output.data, 1)

                # Logs
                preds.append(predicted.item())
                prob_0s.append(probs[0])
                prob_1s.append(probs[1])
            
            avg_preds.append(np.array(preds).mean())
            var_preds.append(np.array(preds).var())
            avg_prob_0s.append(np.array(prob_0s).mean())
            avg_prob_1s.append(np.array(prob_1s).mean())
            var_probs.append(np.array(prob_1s).var())

    return avg_preds, var_preds, avg_prob_0s, avg_prob_1s, var_probs, np.array(var_probs).mean().item()


def test(model, dataloader, criterion, device, filew):
    model.to(device)
    model.eval()

    ids, labels, std_preds, prob_0s, prob_1s, metrics = regular_test(model,dataloader, criterion, device)
    avg_preds, var_preds, avg_prob_0s, avg_prob_1s, var_probs, avg_var = test_uncertainty(model,dataloader, criterion, device)

    reg_test = pd.DataFrame(columns=['IDs', 'Labels', 'Pred', 'Prob. of 0', 'Prob. of 1'])
    unc_test = pd.DataFrame(columns=['Avg. Pred', 'Var. Pred', 'Avg. Prob. of 0', 'Avg. Prob. of 1', 'Var. of Probs'])

    for i in range(len(ids)):
        reg_test.loc[i] = [ids[i], labels[i], std_preds[i], prob_0s[i], prob_1s[i]]

    for i in range(len(avg_preds)):
        unc_test.loc[i] = [avg_preds[i], var_preds[i], avg_prob_0s[i], avg_prob_1s[i], var_probs[i]]

    df_avg_var = pd.DataFrame([avg_var], columns=['Model Avg. Var.'])

    df_metrics = pd.DataFrame.from_dict(metrics)

    result = pd.concat([reg_test, unc_test, df_avg_var, df_metrics], axis=1, sort=False)

    result.to_csv(filew)
