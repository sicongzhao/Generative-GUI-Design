import torch.nn as nn
import torch
import numpy as np

class Attention(nn.Module):
	
	def __init__(self, in_channels, out_channels=None, dimension=1, sub_sample=False, bn=True, residual=True):
		super().__init__()
		if out_channels is None:
			self.out_channels = in_channels // 2 if in_channels > 1 else 1
		else:
			self.out_channels = out_channels
		self.residual = residual
		self.g = nn.Cov1d(in_channels, self.out_channels, kernel_size=1, stride=1, padding=0)
		self.theta = nn.Cov1d(in_channels, self.out_channels, kernel_size=1, stride=1, padding=0)
		self.phi = nn.Cov1d(in_channels, self.out_channels, kernel_size=1, stride=1, padding=0)
		nn.init.normal_(self.theta.weight, 0.02)
		nn.init.normal_(self.phi.weight, 0.02)

		self.W = nn.Conv1d(self.out_channels ,in_channels, kernel_size=1, stride=1, padding=0)
		# BatchNorm reduce overfitting
		if bn:
			self.W = nn.Sequential(self.W, nn.BatchNorm1d(in_channels))
			nn.init.constant_(self.W[1].weight, 0)
			nn.init.constant_(self.W[1].bias, 0)

		# Not sure if the implementation is correct
		if sub_sample:
			self.g = nn.Sequential(self.g, nn.MaxPool1d(2))
			self.phi = nn.Sequential(self.phi, nn.MaxPool1d(2))

	def forward(self,x):
		batch_size = x.size(0)
		g_x = self.g(x).view(batch_size, self.out_channels, -1)
		g_x = g_x.permute(0,2,1)

		theta_x = self.theta(x).view(batch_size, self.out_channels, -1)
		theta_x = theta_x.permute(0,2,1)

		phi_x = self.phi(x).view(batch_size, self.out_channels, -1)

		f = torch.matmul(theta_x, phi_x)

		N = f.size(-1)

		y = torch.matmul(f,g_x)/N
		y = y.permute(0,2,1).contiguous()

		W_y = self.W(y)

		if self.residual:
			output = W_y + x
		else:
			output = W_y

		return output

class Generator(nn.Module):
	def __init__(self, num_elements, geo_num=4, cls_num, attr_num=3):
		super().__init__()
		self.geo_num = geo_num # The number of geometric parameters
		self.cls_num = cls_num # The number of class
		self.attr_num = attr_num # The number of attribute parameters (expected area, aspect ratio, reading-order)
		self.feature_size = geo_num + cls_num + att_num
		# Encode
		self.encoder = nn.Sequential(
			nn.Linear(self.feature_size, self.feature_size*2),
			nn.LeakyReLU(0.02),
			nn.Linear(self.feature_size*2, self.feature_size*2*2),
			nn.LeakyReLU(0.02),
			nn.Linear(self.feature_size*2*2, self.feature_size*2*2)
			)
		# Attention
		self.attention_1 = Attention(self.feature_size**2)
		self.attention_2 = Attention(self.feature_size**2)
		self.attention_3 = Attention(self.feature_size**2)
		self.attention_4 = Attention(self.feature_size**2)
		# Decoder
		self.decoder = nn.Sequential(
			nn.Linear(self.feature_size*2*2, self.feature_size*2),
			nn.LeakyReLU(0.02),
			nn.Linear(self.feature_size*2, self.feature_size) 
			)

		self.fc_6 = nn.Linear(self.feature_size, geo_num)

	def forward(self, x):
		p = x[:,:self.cls_num]
		attr = x[:,-self.attr_num:] # (s, r, d) -> (area, ratio, order)
		x = self.encoder(x)
		x = x.permute(0,2,1).contiguous()
		x_hat = x.clone()
		x = self.attention_1(x)
		x = self.attention_2(x) + x_hat
		x_hat = x.clone()
		x = self.attention_3(x)
		x = self.attention_4(x) + x_hat
		x = x.permute(0,2,1).contiguous()
		x = self.decoder(x)

		out_geo = torch.sigmoid(self.fc_6(x)) # (x_L, y_T, x_R, y_B)
		# Conform attribute: ratio 
		out_geo[:,3] = (attr[:,1] == 0) * (out_geo[:,3] - out_geo[:,1]) + attr[:,1] * (out_geo[:,2] - out_geo[:,0]) + out_geo[:,1]

		output = torch.cat((p,out_geo),2)
		return output

class WireDiscriminator(nn.Module):

	def __init__(self, batch_size, w, h, cls_num, output_channels):
		super().__init__()
		self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
		self.batch_size = batch_size
		self.w = w
		self.h = h
		self.cls_num = cls_num

		# Global Conv Layers
		self.global_conv = nn.Sequential(
			nn.Conv2d(cls_num, output_channels, 3, 1, 1),
			nn.BatchNorm2d(output_channels),
			nn.ReLU(),
			nn.MaxPool2d(2, stride=2),
			nn.Conv2d(output_channels, output_channels*2, 3, 1, 1),
			nn.BatchNorm2d(output_channels*2),
			nn.ReLU(),
			nn.MaxPool2d(2, stride=2),
			nn.Conv2d(output_channels*2, output_channels*3, 3, 1, 1),
			nn.BatchNorm2d(output_channels*3),
			nn.ReLU()
			)
		# Local Conv Layers
		self.local_conv = nn.Sequential(
			nn.Conv2d(cls_num, output_channels, 3, 1, 1),
			nn.BatchNorm2d(output_channels),
			nn.ReLU(),
			nn.MaxPool2d(2, stride=2),
			nn.Conv2d(output_channels, output_channels*2, 3, 1, 1),
			nn.BatchNorm2d(output_channels*2),
			nn.ReLU(),
			nn.MaxPool2d(2, stride=2),
			nn.Conv2d(output_channels*2, output_channels*3, 3, 1, 1),
			nn.BatchNorm2d(output_channels*3),
			nn.ReLU()
			)
		# Init weight
		for l in [1,5,9]:
			nn.init.normal_(self.global_conv[l].weight, 0, 0.02)
			nn.init.normal_(self.local_conv[l].weight, 0, 0.02)
		# Global fc
		self.global_fc = nn.Sequential(
			nn.Linear(output_channels*3*w*h, 128),
			nn.ReLU(),
			nn.Linear(128, 1),
			nn.Sigmoid()
			)
		self.global_decoder = nn.Sequential(
			nn.Linear(output_channels*3*w*h, 128),
			nn.ReLU(),
			nn.Linear(128, cls_num)
			)
		for l in [0,2]:
			nn.init.normal_(self.global_fc[l].weight, 0, 0.02)
			nn.init.normal_(self.global_decoder[l].weight, 0, 0.02)

		# # Archived Layer Definition
		# self.conv_1 = nn.Conv2d(cls_num, output_channels, 3, 1, 1)
		# torch.nn.init.normal_(self.conv_1.weight, 0, 0.02)
		# self.conv_1_bn = nn.BatchNorm2d(output_channels)
		# self.conv_2 = nn.Conv2d(output_channels, output_channels*2, 3, 1, 1)
		# torch.nn.init.normal_(self.conv_2.weight, 0, 0.02)
		# self.conv_2_bn = nn.BatchNorm2d(output_channels*2)
		# self.conv_3 = nn.Conv2d(output_channels*2, output_channels*3, 3, 1, 1)
		# torch.nn.init.normal_(self.conv_3.weight, 0, 0.02)
		# self.conv_3_bn = nn.BatchNorm2d(output_channels*3)


		# # Fully Connected Layers - Probability
		# self.fc_1 = nn.Linear(output_channels*3*w*h, 128)
		# torch.nn.init.normal_(self.fc_1.weight, 0, 0.02)
		# self.fc_2 = nn.Linear(128, 1)
		# torch.nn.init.normal_(self.fc_2.weight, 0, 0.02)
		# # Expected Area
		# self.fc_3 = nn.Linear(output_channels*3*w*h, 128)
		# torch.nn.init.normal_(self.fc_3.weight, 0, 0.02)
		# self.fc_4 = nn.Linear(128, cls_num)
		# torch.nn.init.normal_(self.fc_4.weight, 0, 0.02)
		# pool = nn.MaxPool2d(2, stride=2)



	def wireframe_rendering(self, x):
		# x.size(): b, N, (cls_num + geo_num)
		def k(x):
			return torch.relu(1-torch.abs(x))

		def b(x):
			x = torch.relu(x)
			return -torch.relu(-x+1)+1

		p = x[:,:,:self.cls_num] 		# (b, N, cls_num)
		theta = x[:,:,self.cls_num:]	# (b, N, geo_num)

		batch_size, num_elements, geo_num = theta.size()
		assert(p.size(0)==batch_size and p.size(1)==num_elements)

		theta[:,:,0] *= self.w 			# (b, N)
		theta[:,:,1] *= self.h
		theta[:,:,2] *= self.w
		theta[:,:,3] *= self.h

		# Coordinates
		x_co = np.repeat(np.arange(self.w),self.h).reshape(self.w,self.h)
		y_co = np.repeat(np.arange(self.h),self.w).reshape(self.h,self.w).T

		x_tensor = torch.from_numpy(x_co)	
		y_tensor = torch.from_numpy(y_co)
		x_tensor = x_tensor.view(1, self.w, self.h)	# (1,w,h)
		y_tensor = y_tensor.view(1, self.w, self.h) # (1,w,h)


		base_tensor = torch.cat([x_tensor, y_tensor]).float().to(self.device) # (2,w,h)
		# What if N is different for each input?
		base_tensor = base_tensor.repeat(batch_size*num_elements, 1, 1, 1)	# (b*N,2,w,h), each element -> an image
		theta = theta.view(batch_size*num_elements, geo_num, 1, 1)			# (b*N,geo_num,1,1)

		F_0 = k(base_tensor[:,0] - theta[:,0]) * b(base_tensor[:,1]-theta[:,1]) * b(theta[:,3]-base_tensor[:,1]) # (b*N,w,h)
		F_1 = k(base_tensor[:,0] - theta[:,2]) * b(base_tensor[:,1]-theta[:,1]) * b(theta[:,3]-base_tensor[:,1])
		F_2 = k(base_tensor[:,1] - theta[:,1]) * b(base_tensor[:,0]-theta[:,0]) * b(theta[:,2]-base_tensor[:,0])
		F_3 = k(base_tensor[:,1] - theta[:,3]) * b(base_tensor[:,0]-theta[:,0]) * b(theta[:,2]-base_tensor[:,0])

		val, _ = torch.max(torch.stack((F_0,F_1,F_2,F_3),dim=2),dim=2) # (b*N,w,h)
		# output 	(b,cls_num,w,h)
		# p 		(b, N, cls_num) -> (b,1,1,N,num_class)
		# val 		(b*N,w,h) -> (b,w,h,N,1)
		p = p.unsqueeze(1).unsqueeze(1)
		val = val.view(batch_size,self.w,self.h,-1,1).float()
		prod = p*val # (b,w,h,N,num_class)
		res, _ = torch.max(prod, dim=3)
		I = res.permute(0,3,1,2).contiguous() # (b,cls_num,w,h)
		return I

	def forward(self,x):
		# Global forward
		global_x = self.wireframe_rendering(x)
		global_x = self.global_conv(global_x)
		global_x = global_x.view(x.size(0),-1)
		global_pred = self.global_fc(x)
		global_s = self.global_decoder(x)
		
		# Local forward

		return global_pred, global_s

	# def forward(self,x):
	# 	# Global forward
	# 	x = self.wireframe_rendering(x)
	# 	x = pool(torch.relu(self.conv_1_bn(self.conv_1(x))))
	# 	x = pool(torch.relu(self.conv_2_bn(self.conv_2(x))))
	# 	x = torch.relu(self.conv_3_bn(self.conv_3(x)))
	# 	x = x.view(x.size(0),-1)
	# 	# Prediction: True/False
	# 	pred = torch.relu(self.fc_1(x))
	# 	pred = torch.sigmoid(self.fc_2(pred))
	# 	# Expected Size
	# 	s = torch.relu(self.fc_3(x))
	# 	s = torch.sigmoid(self.fc_4(s))

	# 	# Local forward

	# 	return pred, s
		






















