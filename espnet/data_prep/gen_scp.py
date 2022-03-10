"""Converts new scp files to correct format with correct meeting names
Duplicates ark value for splitting 5 speaker meetings"""


dataset = "eval"
old_scp_path = "/home/dawna/flk24/files4jhrt2/DNC/data/arks.meeting.cmn.tdnn/%s.scp" % dataset

with open(old_scp_path, "r") as old_scp, open(dataset + ".scp", "w") as new_scp:
    old_scp_list = [(line.strip()).split() for line in old_scp]
    for line in old_scp_list:
        partial_meeting_id = line[0][6:]
        if partial_meeting_id in ("-0EN2001a", "-0EN2001d", "-0EN2001e"):  # five speaker meetings
            for i in range(1, 6):
                new_scp.write("AMIMDM" + "-" + str(i) + partial_meeting_id[2:] + " " + line[1] + "\n")
        else:
            new_scp.write("AMIMDM" + line[0][6:] + " " + line[1] + "\n")

