import numpy as np
path = './RirsOfRooms/room1/dist_3/source_1.binary'
with open(path, 'rb') as f:
    records = np.fromfile(f)
    tmp = records.reshape(-1, 5)
    tmp = tmp[tmp[:, 0].argsort()]
    index1 = np.where(tmp[:, 4] <= 1)[0]
    tmp1 = tmp[index1, :]
  #  print(tmp1)
    index2 = np.where(tmp1[:, 3] == 0)[0]
    tmp2 = tmp1[index2]
    print(tmp2)
    

