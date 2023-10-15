import os

domains = ["CALTECH", "LABELME", "SUN", "VOC"]
subdir = ["bird", "car", "chair", "dog", "person"]
path = "/home/andyz3/Distill_CLIP_light/data/VLCS/"

for i in domains:
    file_path = path + i
    folder = os.listdir(file_path)
    f = open("txt_lists/" + i + "_train.txt", "w")
    
    for j in range(len(subdir)):
        image_path = file_path + "/" + subdir[j]
        images = os.listdir(image_path)
        
        for k in images:
            f.write(image_path + '/' + k + " " + str(j+1) + "\n")
        
    f.close()
    
for i in domains:
    file_path = path + i
    folder = os.listdir(file_path)
    f = open("txt_lists/" + i + "_test.txt", "w")
    
    for j in range(len(subdir)):
        image_path = file_path + "/" + subdir[j]
        images = os.listdir(image_path)
        
        for k in images:
            f.write(image_path + '/' + k + " " + str(j+1) + "\n")
        
    f.close()