"""Find best acc values from log files after training"""

import json
import configargparse

def get_parser():  # official paths should be maintained in asr_train.py
    parser = configargparse.ArgumentParser(
        description="Prepare eval files",
        config_file_parser_class=configargparse.YAMLConfigFileParser,
        formatter_class=configargparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--tag', type=str, help='')
    return parser

parser = get_parser()
args, _ = parser.parse_known_args()

tag = args.tag
log_path = "/home/mifs/jhrt2/newDNC/espnet/egs/ami/dnc1/exp/mdm_train_pytorch_%s/results/log" % tag

with open(log_path, 'r') as log_file:
    log = json.load(log_file)


best_val_acc = 0
model_train_acc = 0

for dic in log:
    try:
        val_acc = dic['validation/main/acc']
    except:
        continue
    if val_acc >= best_val_acc:
        best_val_acc = val_acc
        model_train_acc = dic['main/acc']

print('tag: ', tag)
print('train acc: ', model_train_acc)
print('val acc: ', best_val_acc)
