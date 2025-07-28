import os
import pandas as pd
#490398542 genes
# Select data
dataset_name = 'HumanLiverCancerPatient1'
z_index_number = 0
base_path = '/media/huifang/data/vizgen/' + dataset_name +'/detected_transcripts/'
# 2,3,4: globalx, globaly,globalz
# 5,6:localx,localy
# 7: fov
# 8:gene

segments = os.listdir(base_path)
cnt = 0
for seg in segments:
    if not os.path.isdir(seg):
        print("Rearrange " + seg + '...')
        f = open(base_path + seg)
        iter_f = iter(f)
        for line in iter_f:
            data = line.split(',')
            if len(data)<10:
                continue
            cnt +=1
            data = line.split(',')
            z_index = data[4][0]
            fov = data[7]
            des_dir = base_path + z_index + '/'
            os.makedirs(des_dir, exist_ok=True)
            des_file = des_dir + 'z' + z_index + '_fov' + fov + '.txt'
            des_f = open(des_file,'a+')
            des_f.write(line)
            des_f.close()
        f.close()
        print('done')
print(cnt)

