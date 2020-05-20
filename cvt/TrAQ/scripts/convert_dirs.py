#!/usr/bin/python
import os
home = 'data'
os.listdir(home)
dirs = os.listdir(home)
files = {}
for dir in dirs:
  dir = os.path.abspath("%s/%s" %(home,dir))
  if os.path.isdir(dir):
    print(dir)
    subdirs = os.listdir(os.path.abspath(dir))
    files[dir] = subdirs
print(files)
print("done")

for dir in files:
  farr = files[dir]
  for f in farr:
    print( "           %s" % f )
    _f = f.split('_')
    _f.pop(0)
    _f = '_'.join(_f)
    dir_ext = _f.split('.')[0]
    if os.path.isfile(f):
      f_ext = _f.split('.')[1]
      if f_ext == 'mp4': 
        new_dir = os.path.abspath("%s_%s" % (dir,dir_ext))
        print(new_dir)
        print(os.stat().os.path.getmtime(_f))
#        if not os.path.isdir(new_dir):
#          os.mkdir(new_dir)
#          f_new = "%s/raw.%s" % (new_dir,f_ext) 
#          os.rename( os.path.abspath("%s/%s" % (dir,f)),
#                     os.path.abspath(f_new) )
#          print( "  %s" % (new_dir))
