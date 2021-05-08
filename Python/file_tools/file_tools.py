import argparse
parser = argparse.ArgumentParser(description='dyngq')
parser.add_argument('-s', '--source', type=str, default = None)
parser.add_argument('-t', '--target', type=str, default = None)
# parser.add_argument('--args', type=str, default = None)
parser.add_argument('-f', '--find', action='store_true', help="find the repeats.")
parser.add_argument('-ds', '--deletesource', action='store_true', help="delete the repeats from source.")
parser.add_argument('-dt', '--deletetarget', action='store_true', help="delete find the repeats from target.")
parser.add_argument('-r', '--rename', action='store_true', help="rename the files which have str like ‘2234-IMG-’.")
# parser.add_argument('--args', action='store_true', help="Whether to run eval on the dev set.")
args = parser.parse_args()
print(args)

import os
from tqdm import tqdm

sou_dir = args.source
tar_dir = args.target

sou = os.listdir(sou_dir)
tar = os.listdir(tar_dir)

# filename = 'write_data.txt'
# with open(filename,'w') as f: 
if args.find:
    count = 0
    for i in tqdm(sou):
        if i in tar:
            # print(tar_dir+i)
            # f.writelines(tar_dir+i+'\n')
            print(tar_dir+i+'\n')
            if args.deletesource:
                os.remove(sou_dir+i)
            if args.deletetarget:
                os.remove(tar_dir+i)
            count = count + 1
    # f.writelines(str(count)+" "+str(len(sou))+" "+str(len(tar))+'\n')
    print(str(count)+" "+str(len(sou))+" "+str(len(tar))+'\n')

if args.rename:
    re_dir = sou_dir
    ren = sou
    for s in ren:
    	po = s.find('-')
    	if po<=4 and po>=0:
        	os.rename(re_dir+s, re_dir+s[po+1:])
        	print(po,re_dir+s[po+1:])
    
    