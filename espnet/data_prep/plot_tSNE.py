import numpy as np
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
import json
import configargparse

from data_loading import build_segment_dicts

np.random.seed(0)


def do_plot(args, dataset, json_path, sub_meeting_id, meeting_length, output_file, label_type):
    with open(json_path, 'r') as json_file:
        json_dict = json.load(json_file)

    if label_type == "DNC":
        labels = json_dict["utts"][sub_meeting_id]['output'][0]["rec_tokenid"][:-2]
    elif label_type == "true":
        # need to do permutation thing (1->2 or just do with colours)
        labels = json_dict["utts"][sub_meeting_id]['output'][0]["tokenid"]
    elif label_type == "SC":
        labels = json_dict["utts"][sub_meeting_id]['output'][0]["rec_tokenid"][:-2]
    else:
        raise ValueError("Invalid label type")

    print(label_type, sub_meeting_id, labels)
    
    labels =  [int(i) for i in labels.split()]


    # now find corresponding embeddings
    meetings, speakers = build_segment_dicts(args, dataset, filt=True, emb="dvec", tdoa=False, gccphat=False, average=True, tdoa_norm=False)
    input_vectors = np.array(meetings["AMIMDM-" + sub_meeting_id[4:-4]][:meeting_length])

    X_embedded = TSNE(n_components=2, learning_rate='auto', init='random').fit_transform(input_vectors)

    colormap = np.array(['g', 'r', 'b', 'y'])
    plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=colormap[labels])
    plt.gca().axes.get_xaxis().set_visible(False)
    plt.gca().axes.get_yaxis().set_visible(False)
    plt.savefig(output_file)



def get_parser():  # debugging only, official paths should be maintained in asr_train.py
    parser = configargparse.ArgumentParser(
        description="Load speech data",
        config_file_parser_class=configargparse.YAMLConfigFileParser,
        formatter_class=configargparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--eval-emb', type=str,
            default="/home/mifs/jhrt2/newDNC/data/arks.concat/eval.scp", help='')
    # parser.add_argument('--eval-emb', type=str,
    #         default="/home/mifs/jhrt2/newDNC/data/arks.meeting.cmn.tdnn/eval.scp", help='')
    # parser.add_argument('--eval-rttm', type=str,
    #         default="/home/mifs/jhrt2/newDNC/data/window_level_rttms/eval150_window_level.rttm", help='')
    # parser.add_argument('--eval-emb', type=str,
    #         default="/home/mifs/epcl2/project/embeddings/james/eval", help='')
    parser.add_argument('--eval-rttm', type=str,
            default="/home/mifs/jhrt2/newDNC/data/rttms.concat/eval.rttm", help='')

    parser.add_argument('--tdoa-directory', type=str,
            default="/data/mifs_scratch/jhrt2/BeamformIt/MDM_AMI_fixedref_10", help='')
    return parser



def main():
    parser = get_parser()
    args, _ = parser.parse_known_args()
    dataset = "eval"

    json_path = "/data/mifs_scratch/jhrt2/models/FinalResults/segment_level/mdm_train_pytorch_nonewdiac/decode_mdm_dev_decode/data.12.json"
    #json_path = "/data/mifs_scratch/jhrt2/models/FinalResults/spectral/segment50/eval95k24.1.json"
    # only do -000 submeetings so can access input vectors easily
    sub_meeting_id = "AMI-0IS1009d-000"
    meeting_length = 50
    label_type = "true"
    output_file = "nonewdiac" + label_type + "labels.png"

    do_plot(args, dataset, json_path, sub_meeting_id, meeting_length, output_file, label_type)

if __name__ == '__main__':
    main()
