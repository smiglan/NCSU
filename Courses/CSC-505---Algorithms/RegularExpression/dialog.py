# -*- coding: utf-8 -*-
import re
import sys

def Dialogoutput(lines,outputf):
	Outputdata = re.findall(r'[“"].*?[”"]', lines,re.DOTALL)
	Outputdataiter = [m.span() for m in re.finditer(r'[“"].*?[”"]', lines,re.DOTALL)]
	with open(outputf, 'w',encoding="utf8") as filehandle:
	    for listitem in Outputdata:
	        filehandle.write(listitem+"\n")
	filehandle.close()
	print('File output successful with file name '+outputf)
	return Outputdata,Outputdataiter	
inputf = sys.argv[1]
outputf = sys.argv[2]

with open(inputf,'r',encoding="utf8") as f:
		linesi = f.read()
Dialogoutput(linesi,outputf)
f.close()
