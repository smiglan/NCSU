import subprocess

files = ['dracula.txt','sh.txt']
searchstrings = ['red eyes','pity me','birds']
outputfiles = ['Searchdracula.txt','Searchsh.txt']
for f,o in zip(files,outputfiles):
	
	for s in searchstrings:
		print("filename : ",f)
		print("search string : ",s)
		if f == 'dracula.txt':
			c=1
		elif f == 'sh.txt':
			c = 2
		cmd = "python3 dialogsearch.py "+f+" "+str(c)+' "'+s+'" '+s.replace(" ","")+o
		returned_value = subprocess.call(cmd, shell=True)  # returns the exit code in unix
