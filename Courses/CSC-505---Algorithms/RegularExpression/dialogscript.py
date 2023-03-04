import subprocess

files = ['dracula.txt','sh.txt']
outputfiles = ['outputdracula.txt','outputsh.txt']
for f,o in zip(files,outputfiles):
	cmd = "python3 dialog.py "+f+" "+o
	returned_value = subprocess.call(cmd, shell=True)  # returns the exit code in unix