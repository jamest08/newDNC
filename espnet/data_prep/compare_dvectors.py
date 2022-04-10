"""For visually inspecting and comparing my dvectors with the original DNC ones"""

import kaldiio
import numpy as np

from data_loading import open_scp

my_scp_path = '/home/mifs/jhrt2/newDNC/data/arks.concat/train.scp'
my_meeting_path_lists = open_scp(my_scp_path)
my_meeting_id = my_meeting_path_lists[-1][0]
my_meeting_path = my_meeting_path_lists[-1][1]
my_dvectors = kaldiio.load_mat(my_meeting_path)
print('my meeting id: ', my_meeting_id)

orig_scp_path = '/home/mifs/jhrt2/DNC/DNC/data/train.scp'
orig_meeting_path_lists = open_scp(orig_scp_path)
orig_meeting_id = orig_meeting_path_lists[-1][0]
orig_meeting_path = orig_meeting_path_lists[-1][1]
print('orig meeting id: ', orig_meeting_id)
orig_dvectors = kaldiio.load_mat(orig_meeting_path)

print(my_dvectors[194277:194361] - orig_dvectors)
