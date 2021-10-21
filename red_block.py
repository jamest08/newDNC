import kaldiio
import numpy as np

# open rttm file containing segmentation data
rttm_file = open("data/rttms/test_train.rttm", "r")
segments_desc_list = [(line.strip()).split() for line in rttm_file]
rttm_file.close()

# build dictionary segment_desc_dict with key as meeting name,
# value as list of tuples (start_time, duration, speaker_label)
segment_desc_dict = {}
for segment_desc in segments_desc_list:
    meeting_id = segment_desc[1]
    start_point = float(segment_desc[3])
    duration = float(segment_desc[4])
    speaker_label = segment_desc[7]
    if not meeting_id in segment_desc_dict:
        segment_desc_dict[meeting_id] = [(start_point, duration, speaker_label)]
    else:
        segment_desc_dict[meeting_id].append((start_point, duration, speaker_label))

# open scp file containing paths to meeting d-vectors
scp_file = open("data/arks/train.scp", "r")
meeting_path_lists = [(line.strip()).split() for line in scp_file] # list of lists [meeting_id, path_to_ark]
scp_file.close()

# create two dictionaries both with key as meeting_id:
segmented_meetings_dict = {}  # value is list of segments.  Each segment is a list of d-vectors
segmented_speakers_dict = {} # value is list of speakers aligning with segments 
# simultaneously build d-vector dictionary
# key is speaker, value is list of all d_vectors for that speaker (across all meetings)
dvec_dict = {}
for meeting_path_list in meeting_path_lists:
    meeting_id = meeting_path_list[0]
    meeting_path = meeting_path_list[1]
    meeting_dvectors_list = kaldiio.load_mat(meeting_path)
    segments = []
    speakers = []
    for segment_desc in segment_desc_dict[meeting_id]:
        start_index = max(int(round(segment_desc[0] / 0.01)), 0) # d-vectors cover a 2s period and start every 10ms (overlapping)
        end_index = min(int(round((segment_desc[0] + segment_desc[1]) / 0.01)), len(meeting_dvectors_list))
        segment = meeting_dvectors_list[start_index:end_index]
        speaker = segment_desc[2]
        segments.append(segment)
        speakers.append(speaker)
        if not speaker in dvec_dict:
            dvec_dict[speaker] = segment
        else:
            dvec_dict[speaker] = np.append(dvec_dict[speaker], segment, axis=0)
    segmented_meetings_dict[meeting_id] = segments
    segmented_speakers_dict[meeting_id] = speakers


print(kaldiio.load_mat("/home/dawna/flk24/diarization/NeuralSpeakerClustering/extractDVector.Paper.Meeting/data/arks/train.ark:16"))
print(segmented_meetings_dict["AMIMDM-00IB4005"])
print(segmented_speakers_dict["AMIMDM-00IB4005"])
print(dvec_dict["FIE038"])
