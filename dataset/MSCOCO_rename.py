import os

def cocoimage_rename():
    folder_path = "$PATH$"
    filelist = os.listdir(folder_path)  
    for files in filelist:   
        Olddir = os.path.join(folder_path, files)
        if os.path.isdir(Olddir):
                continue
        filename = os.path.splitext(files)[0]     
        filetype = os.path.splitext(files)[1]
        new_name = "COCO_2014_" + filename.split('_')[2] + filetype
        Newdir = os.path.join(folder_path, new_name)
        os.rename(Olddir, Newdir)   
    return True

if __name__ == '__main__':
    cocoimage_rename()