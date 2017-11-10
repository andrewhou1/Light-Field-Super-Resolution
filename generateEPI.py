import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import sys
import numpy as np
import scipy.misc

def loadlf(folder):
	filelist = sorted(os.listdir(folder))
	for img in filelist[:]:
		if not(img.endswith(".png")):
			filelist.remove(img)
	
	LF_r = list()
	LF_g = list()
	LF_b = list()

	for i in range(len(filelist)):
		filename = folder+"/"+filelist[i]
		img = mpimg.imread(filename)
		print("read: "+filename)
		LF_r.append(img[:, :, 0])
		LF_g.append(img[:, :, 1])
		LF_b.append(img[:, :, 2])
	
	return (LF_r, LF_g, LF_b)		

def lf2epi(LF):
	return np.transpose(LF, (0, 2, 1))

def main():
	(LF_r, LF_g, LF_b) = loadlf(sys.argv[1])
	#LF_rt = lf2epi(LF_r)
	for i in range(17):
		row_start = i*17
		row_end = row_start+16
		LF_rt = lf2epi(LF_r[row_start:row_end+1])
		LF_gt = lf2epi(LF_g[row_start:row_end+1])
		LF_bt = lf2epi(LF_b[row_start:row_end+1])
		dims = LF_rt.shape
		lastdim = dims[2]
		for j in range(lastdim):
			EPI_r = LF_rt[:, :, j]
			EPI_g = LF_gt[:, :, j]
			EPI_b = LF_bt[:, :, j]
			img = np.zeros((17, dims[1], 3), 'uint8') 
			img[:, :, 0] = EPI_r*255
			img[:, :, 1] = EPI_g*255
			img[:, :, 2] = EPI_b*255
			
			#print(img[:, :, 0])
			#print(img[:, :, 1])
			#print(img[:, :, 2])
			
			'''for r in range(17):
				for c in range(dims[1]):
					if(img[r, c, 0] > 255):
						print("red")
					if(img[r, c, 1] > 255):
						print("green")
					if(img[r, c, 2] > 255):
						print("blue")'''
						
			if(i < 9):
				if(j < 9):
					EPIfilename = sys.argv[2]+"/"+"Chess_row_0"+str((i+1))+"_scanline_000"+str((j+1))+".png"
				elif(j < 99):
					EPIfilename = sys.argv[2]+"/"+"Chess_row_0"+str((i+1))+"_scanline_00"+str((j+1))+".png"
				elif(j < 999):
					EPIfilename = sys.argv[2]+"/"+"Chess_row_0"+str((i+1))+"_scanline_0"+str((j+1))+".png"
				else:
					EPIfilename = sys.argv[2]+"/"+"Chess_row_0"+str((i+1))+"_scanline_"+str((j+1))+".png"
			else:
				if(j < 9):
					EPIfilename = sys.argv[2]+"/"+"Chess_row_"+str((i+1))+"_scanline_000"+str((j+1))+".png"
				elif(j < 99):
					EPIfilename = sys.argv[2]+"/"+"Chess_row_"+str((i+1))+"_scanline_00"+str((j+1))+".png"
				elif(j < 999):
					EPIfilename = sys.argv[2]+"/"+"Chess_row_"+str((i+1))+"_scanline_0"+str((j+1))+".png"
				else:
					EPIfilename = sys.argv[2]+"/"+"Chess_row_"+str((i+1))+"_scanline_"+str((j+1))+".png"
					

			#mpimg.imsave(EPIfilename, img)
			scipy.misc.toimage(img).save(EPIfilename)
		

if __name__ == "__main__":
	main()	
