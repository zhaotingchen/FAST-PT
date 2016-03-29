'''
	This RG flow method usese a digital filter to smooth out the instabilities that develop at 
	high k. This is a good integrator to use when k_max is greater than 1 and the grid
	is sparsely sampled. 
	
	J. E. McEwen (c) 2016 
	mcewen.24@osu.edu 
'''

import numpy as np
from matter_power_spt import one_loop
import matplotlib.pyplot as plt 
 
from fastpt_extr import p_window
from scipy.signal import butter, lfilter, filtfilt, lfilter_zi
import time 
import sys
import FASTPT



def RG_RK4_filt(name,k,P,d_lambda,max,n_pad,P_window,C_window):
	
	x=max/d_lambda-int(max/d_lambda) 
	if ( x !=0.0 ):
		raise ValueError('You need to send a d_lambda step so that max/d_lambda has no remainder to reach Lambda=max')
	
	#save name 
	name='RG_RK4_filt_'+name
	N_steps=int(max/d_lambda)
	# The spacing in log(k)
	Delta=np.log(k[1])-np.log(k[0])
	
	# This window function tapers the edge of the power spectrum.
	# It is applied to each Runge-Kutta step.
	# You could change it here. 
	W=p_window(k,P_window[0],P_window[1])
	#W=1
	t1=time.time()

	# windowed initial power spectrum 
	P_0=P*W
	
	nu=-2
	spt_matrix=np.array([[0,0,0,0],[0,0,2,0],[0,0,4,0],[2,-2,2,0],\
							[1,-1,1,0],[1,-1,3,0],[2,-2,0,1] ])
	fastpt=FASTPT.FASTPT(k,nu,spt_matrix,n_pad=n_pad) 	
	P_spt=fastpt.one_loop(P_0,C_window=C_window) 
	# standard 1-loop correction 
	#_,P_22,P_13=one_loop(k,P_0, C_window=C_window,n_pad=n_pad)
	#P_spt=P_0 + P_22 + P_13
	
	# initial lambda 
	Lambda=0

	d_out=np.zeros((N_steps+3,k.size+1))
	d_out[0,1:]=k
	d_out[1,1:]=P_0
	d_out[2,1:]=P_spt
	
	# filtering specific 
	k_start=1; k_end=5
	id1=np.where( k > k_end)[0]
	id2=np.where( k <= k_end)[0]
	id3=np.where( k > k_start)[0]
	#id4=np.where( k <= k_start)[0]
	id4=np.where( (k > k_start) & ( k<= k_end))[0]
	
	order=6; wn=.1
	B,A=butter(order,wn, btype='low', analog=False)
	
	theta=np.linspace(1,0,id4.size)
	W_fad=theta - 1/2./np.pi*np.sin(2*np.pi*theta) 
	filt_pad=id3.size
	# end filtering specific 
	
	def filt_zp(k,P_filt):
		
		def zero_phase(sig):
		
			sig=np.pad(sig,(filt_pad,filt_pad), 'constant', constant_values=(0, 0))
			#zi=lfilter_zi(B,A)
			#x,_=lfilter(B,A,sig, zi=zi*sig[0])
			x=lfilter(B,A,sig)
			#y,_=lfilter(B,A,x,zi=zi*x[0])
			y=lfilter(B,A,x[::-1])
			y=y[::-1]
			#return y
			return y[filt_pad:id3.size+filt_pad]
	
		P_smoothed=zero_phase(P_filt[id3])
		P_patch=P_filt[id4]*W_fad
		P_filt[id3]=P_smoothed
		P_filt[id4]=P_patch+(1-W_fad)*P_filt[id4]
	
		return P_filt
	i=0
	while Lambda <= max:
	
		k1=fastpt.one_loop(P,C_window=C_window) 
		k1=filt_zp(k,k1)
		
		k2=fastpt.one_loop(k1*d_lambda/2.+ P,C_window=C_window) 
		k2=filt_zp(k,k2)

		k3=fastpt.one_loop(k2*d_lambda/2. +P,C_window=C_window) 
		k3=filt_zp(k,k3)

		k4=fastpt.one_loop(k3*d_lambda +P,C_window=C_window)
		k4=filt_zp(k,k4)
		
		
		# full Runge-Kutta step 
		P=P+1/6.*(k1+2*k2+2*k3+k4)*d_lambda
		
		# check for failure. 
		if (np.any(np.isnan(P))):
			print('RG flow has failed. It could be that you have not chosen a step size well.')
			print('You may want to consider a smaller step size.')
			print('Failure at lambda =', Lambda)
			sys.exit()
			
		if (np.any(np.isinf(P))):
			print('RG flow has failed. It could be that you have not chosen a step size well.')
			print('You may want to consider a smaller step size.')
			print('Failure at lambda =', Lambda)
			sys.exit()
			
		#update lambda and the iteration 
		i=i+1
		Lambda+=d_lambda
		#print(Lambda)

		# update data for saving 
		d_update=np.append(Lambda,P)
		d_out=np.row_stack((d_out,d_update))
# 		
#		if (i % 20 ==0):
# 			# I like to save at every 20th iteration. You
# 			# could turn this off if you like
# 			print('this is the iteration', i)
# 			print('this is Lambda', Lambda )
# 			print('this is d_lambda', d_lambda)
# 			print('this is time since start', time.time()-t1)
# 			np.save(name,d_out)	
# 		
# 		# You could plot each step, or plot some downsample of steps here
# 		# this is a good way to monitor instabilities 
# 		#if (i % 100 == 0 ): 
# 		#if (i % 1 == 0 ): 
		if (False):	
# 			
			ax=plt.subplot(111)
			ax.set_xscale('log')
			ax.set_yscale('log')
			ax.set_xlabel('k')
			
			ax.plot(k,P)
			ax.plot(k,P_0, color='red')
			
			plt.grid()
			plt.show()
		
	# save the data 
	t2=time.time()
	print('time to run seconds', t2-t1)
	print('time to run minutes', (t2-t1)/60.)
	np.save(name,d_out)	
	return P 
	
if __name__ == "__main__":
	
	if sys.version_info.major==2:
		from ConfigParser import SafeConfigParser
	if sys.version_info.major>=3:
		from configparser import SafeConfigParser
	parser = SafeConfigParser()
	
	
	name='kmax10_example.ini'
	
	parser.read(name)
	
	k_max=parser.getfloat('floats', 'k_max')
	k_min=parser.getfloat('floats', 'k_min')
	step=parser.getfloat('floats', 'step')
	max=parser.getfloat('floats', 'max')
	P_right=parser.getfloat('floats', 'P_w_right')
	P_left=parser.getfloat('floats', 'P_w_left')
	C_window=parser.getfloat('floats', 'C_window')
	n_pad=parser.getint('integers', 'n_pad')
	down_sample=parser.getint('integers', 'down_sample')
	read_name=parser.get('files', 'in_file')
	name=parser.get('files', 'out_file')
	
	
	d=np.loadtxt(read_name)	# load data
	k=d[:,0]
	P=d[:,1]

	id=np.where( (k >= k_min) & (k <= k_max) )[0]
	k=k[id]
	P=P[id]
	
	k=k[::down_sample]
	P=P[::down_sample]
	
	# if your array is not even in size, FAST-PT will not work-
	# trim if so. 
	if (k.size % 2 != 0):
		k=k[:-1]
		P=P[:-1]
		
	print('Details of run.')
	print('save name :', name)
	print('k min and max:', k_min, k_max )
	print('step size : ', step)
	print('grid size : ', k.size)
	print('d log k: ', np.log(k[1])-np.log(k[0]) )
	print('down sample factor:', down_sample)

	P_window=np.array([P_left,P_right])  
	

	P_rg=RG_RK4_filt(name,k,P,step,max,n_pad,P_window,C_window)	
	
	ax=plt.subplot(111)
	ax.set_xscale('log')
	ax.set_yscale('log')
	ax.set_xlabel('k')
	
	ax.set_ylabel(r'$P(k)$', size=30)
	ax.set_xlabel(r'$k$', size=30)
	
	ax.plot(k,P, label='linear power') 
	ax.plot(k,P_rg, label='RG' )
	
	plt.legend(loc=3) 
					
	plt.grid()
	plt.show()
	