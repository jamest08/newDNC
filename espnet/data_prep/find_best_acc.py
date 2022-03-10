"""Find best acc values from log files after training"""

import json

tag = "dvecgccphat"
log_path = "/home/mifs/jhrt2/newDNC/espnet/egs/ami/dnc1/exp/mdm_train_pytorch_%s/results/log" % tag

with open(log_path, 'r') as log_file:
    log = json.load(log_file)

train_accs = [dic['main/acc'] for dic in log]
val_accs = [dic['validation/main/acc'] for dic in log]


best_val_acc = 0
model_train_acc = 0

for i in range(len(val_accs)):
    if val_accs[i] >= best_val_acc:
        best_val_acc = val_accs[i]
        model_train_acc = train_accs[i]

print('tag: ', tag)
print('best train acc: ', model_train_acc)
print('best val acc: ', best_val_acc)
