import os 

path = r'C:\PC同步资料\Qun_Notes\Photos'

for i in range(2,18):
    oldname = 'Untitled '+str(i)+'.png'
    newname = 'Untitled-'+str(i)+'.png'
    os.rename(path+"\\"+oldname , path+"\\"+newname)
