# -*- coding: utf-8 -*-

import re
import sys

def Dialogoutput(lines):
	Outputdata = re.findall(r'[“"].*?[”"]', lines,re.DOTALL)
	Outputdataiter = [m.span() for m in re.finditer(r'[“"].*?[”"]', lines,re.DOTALL)]
	# with open(output, 'w') as filehandle:
	#     for listitem in Outputdata:
	#         filehandle.write(listitem+"\n")
	return Outputdata,Outputdataiter	

def searchstringpreprocess(searchstringi,dialogsi,dialogsiteri):
	s = "[ |\\n]"
	s = s.join(searchstringi.split())
	newindex = []
	for i,d in enumerate(dialogsi):
		x = re.finditer(s, d,re.DOTALL)
		x = [m.span() for m in re.finditer(s, d,re.DOTALL)]
		if x:
			for xx in x:
				newindex.append(dialogsiteri[i][0])

	return newindex
def DialogsearchCh(lines,searchstring,outputf):
	dialogs,dialogsiter = Dialogoutput(lines)		
	keywordindex = searchstringpreprocess(searchstring,dialogs,dialogsiter)
	chapterindex = [m.start() for m in re.finditer('CHAPTER', lines)]
	chapterindex = chapterindex[int(len(chapterindex)/2):]
	name = ''
	outf = open(outputf,'w+',encoding="utf8")
	for i,c in enumerate(chapterindex):
		newlinecount = 0
		name = name+'\n'
		if i != len(chapterindex)-1:
			final = chapterindex[i+1]
		else:
			final = len(lines)	
		for j in range(chapterindex[i],final):
			if lines[j] == '\n':
				newlinecount += 1
			if j != '\n' and newlinecount <3:
				name = name+lines[j]
	name = name.strip().splitlines()
	while("" in name) : 
	    name.remove("")
	if not keywordindex:
		print('Search string doesnt exist')
	else:
		indexd = []
		for k in keywordindex:
			index = 'Searched String Found Before Chapter Start'
			i =0
			for c in chapterindex:
				if k-c <0:
					continue
				else:
					if i == 0:
						i = i+1
						d = k-c
						index = c
						continue
					i = i+1
					if k-c < d:
						d = k -c
						index = c
			
			if isinstance(index,str):
				print(index)
				outf.write(index+'\n')
			else:
				print("Found in ",name[2*(chapterindex.index(index)-1)],name[2*(chapterindex.index(index)-1)+1])
				outf.write("Found in "+name[2*(chapterindex.index(index)-1)]+"  "+name[2*(chapterindex.index(index)-1)+1]+'\n')
			indexd.append(index)
	outf.close()
def DialogsearchAn(lines,searchstring,outputf):
	result = lines[lines.find("Contents")+len("Contents"):].splitlines()
	while("" in result) : 
	    result.remove("")
	resultn = []
	resultn = result
	outf = open(outputf,'w+',encoding="utf8")

	final = [resultn[0]]
	for i in range(len(resultn)-1):
		if resultn[0].replace(' ','').lower() == resultn[i+1].replace(' ','').lower():
			break
		else:
			final.append(resultn[i+1])
	results = [i.split('.')[1].strip().upper() for i in result[:len(final)]]
	resultsindex = [lines.find(i) for i in results]

	dialogs,dialogsiter = Dialogoutput(lines)		

	keywordindex = searchstringpreprocess(searchstring,dialogs,dialogsiter)

	chapterindex = resultsindex
	if not keywordindex:
		print('Search string doesnt exist')
	else:
		indexd = []
		for k in keywordindex:
			i =0
			index = 'Searched String Found Before Chapter Start'
			for c in chapterindex:
				if k-c <0:
					continue
				else:
					if i == 0:
						i = i+1
						d = k-c
						continue
					i = i+1
					if k-c < d:
						d = k -c
						index = c
			indexd.append(index)
		for i in indexd:	
				print('Found in ', final[chapterindex.index(i)])
				outf.write('Found in '+ final[chapterindex.index(i)]+' \n')
	
	outf.close()
if __name__ == "__main__": 
	inputf = sys.argv[1]
	category = sys.argv[2]
	ss	= sys.argv[3]
	output = sys.argv[4]
	with open(inputf,'r',encoding="utf8") as f:
			linesi = f.read()
	if category != '1' and category	!= '2':
		print("Incorrect input cateogory selected")	
	elif category == '1':
		DialogsearchCh(linesi,ss,output)
	elif category == '2':
		DialogsearchAn(linesi,ss,output)


