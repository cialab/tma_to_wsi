from os import listdir
from os.path import isfile,join
import h5py
import numpy as np
import torch
import sys
import asyncio

def background(f):
  def wrapped(*args, **kwargs):
    return asyncio.get_event_loop().run_in_executor(None, f, *args, **kwargs)
  return wrapped

@background
def my_function(fin,fout):
  a=h5py.File(fin,'r')
  a=a[('features')]
  a=torch.from_numpy(np.array(a))
  torch.save(a,fout)


files = [f for f in listdir(sys.argv[1]) if isfile(join(sys.argv[1], f))]
for i in range(0,len(files)):
  my_function(sys.argv[1]+files[i],sys.argv[2]+files[i].split('.')[0]+'.pt')
  #a=h5py.File(sys.argv[1]+files[i])
  #a=a[('features')]
  #a=torch.from_numpy(np.array(a))
  #torch.save(a,sys.argv[2]+files[i].split('.')[0]+'.pt')
