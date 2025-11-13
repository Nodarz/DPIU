# Python helloworld
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from models.layer_utils import *


class model_encdec(nn.Module):

	def __init__(self, settings, pretrained_model1=None, pretrained_model2=None, pretrained_model3=None):
		super(model_encdec, self).__init__()

		self.name_model = 'AIO_autoencoder'
		self.use_cuda = settings["use_cuda"]
		self.dim_embedding_key = 64
		self.past_len = settings["past_len"]
		self.future_len = settings["future_len"]
		self.mode = settings["mode"]

		assert self.mode in ['trajectory'], 'WRONG MODE!'
		# LAYERS for different modes
		if self.mode == 'trajectory':
			# model1
			self.abs_past_encoder = pretrained_model1.abs_past_encoder
			self.norm_past_encoder = pretrained_model1.norm_past_encoder
			self.norm_fut_encoder = pretrained_model1.norm_fut_encoder
			self.res_past_encoder = pretrained_model1.res_past_encoder
			self.social_pooling_X = pretrained_model1.social_pooling_X
			self.decoder = pretrained_model1.decoder
			self.decoder_x = pretrained_model1.decoder_x
			self.decoder_2 = pretrained_model1.decoder_2
			self.decoder_2_x = pretrained_model1.decoder_2_x
			self.input_query_w = pretrained_model1.input_query_w
			self.past_memory_w = pretrained_model1.past_memory_w
			# mode 'trajectory'
			self.memory_past = torch.load(
				r'./training/saved_memory/sdd_0.5_15442_filter_past.pt',
				map_location='cuda:0')
			self.memory_fut = torch.load(
				r'./training/saved_memory/sdd_0.5_15442_filter_fut.pt',
				map_location='cuda:0')
			self.memory_dest = torch.load(
				r'./training/saved_memory/sdd_0.5_15442_part_traj.pt',
				map_location='cuda:0')[:, -1]

			# model2
			self.abs_past_encoder2 = pretrained_model2.abs_past_encoder
			self.norm_past_encoder2 = pretrained_model2.norm_past_encoder
			self.norm_fut_encoder2 = pretrained_model2.norm_fut_encoder
			self.res_past_encoder2 = pretrained_model2.res_past_encoder
			self.social_pooling_X2 = pretrained_model2.social_pooling_X
			self.decoder2 = pretrained_model2.decoder
			self.decoder_x2 = pretrained_model2.decoder_x
			self.decoder_22 = pretrained_model2.decoder_2
			self.decoder_2_x2 = pretrained_model2.decoder_2_x
			self.input_query_w2 = pretrained_model2.input_query_w
			self.past_memory_w2 = pretrained_model2.past_memory_w
			# mode 'trajectory'
			self.memory_past2 = torch.load(
				r'./training/saved_memory/sdd_0.375_15252_filter_past.pt',
				map_location='cuda:0')
			self.memory_fut2 = torch.load(
				r'./training/saved_memory/sdd_0.375_15252_filter_fut.pt',
				map_location='cuda:0')
			self.memory_dest2 = torch.load(
				r'./training/saved_memory/sdd_0.375_15252_part_traj.pt',
				map_location='cuda:0')[:, -1]

			# model3
			self.abs_past_encoder3 = pretrained_model3.abs_past_encoder
			self.norm_past_encoder3 = pretrained_model3.norm_past_encoder
			self.norm_fut_encoder3 = pretrained_model3.norm_fut_encoder
			self.res_past_encoder3 = pretrained_model3.res_past_encoder
			self.social_pooling_X3 = pretrained_model3.social_pooling_X
			self.decoder3 = pretrained_model3.decoder
			self.decoder_x3 = pretrained_model3.decoder_x
			self.decoder_23 = pretrained_model3.decoder_2
			self.decoder_2_x3 = pretrained_model3.decoder_2_x
			self.input_query_w3 = pretrained_model3.input_query_w
			self.past_memory_w3 = pretrained_model3.past_memory_w
			# mode 'trajectory'
			self.memory_past3 = torch.load(
				r'./training/saved_memory/sdd_0.25_14960_filter_past.pt',
				map_location='cuda:0')
			self.memory_fut3 = torch.load(
				r'./training/saved_memory/sdd_0.25_14960_filter_fut.pt',
				map_location='cuda:0')
			self.memory_dest3 = torch.load(
				r'./training/saved_memory/sdd_0.25_14960_part_traj.pt',
				map_location='cuda:0')[:,-1]

			for p in self.parameters():
				p.requires_grad = False

		# activation function
		self.relu = nn.ReLU()

	def get_memory_index(self, state_past, memory_past):
		# state_past: batch_size, feature_size
		# memory_past: memory_size, feature_size
		past_normalized = F.normalize(memory_past, p=2, dim=1)
		state_normalized = F.normalize(state_past, p=2, dim=1)
		weight_read = torch.matmul(state_normalized, past_normalized.transpose(0, 1))
		_, index_max = torch.sort(weight_read, descending=True)
		return index_max, weight_read

	def get_memory_index_batch(self, state_past, memory_past):
		# state_past: batch_size, 1, feature_size
		# memory_past: batch_size, 300, feature_size
		past_normalized = F.normalize(memory_past, p=2, dim=2)
		state_normalized = F.normalize(state_past, p=2, dim=2)
		weight_read = torch.matmul(state_normalized, past_normalized.transpose(1, 2))
		weight_read = weight_read.squeeze()
		_, index_max = torch.sort(weight_read, descending=True)
		return index_max, weight_read

	def prediction_generate(self, mul_num, num_sample, sample_memory_index, weight_read, memory_fut, past,
							state_past, abs_past_state_social):
		prediction = torch.Tensor()
		if self.use_cuda:
			prediction = prediction.cuda()
		if mul_num == 4:
			decoder = self.decoder
			decoder_x = self.decoder_x
			res_past_encoder = self.res_past_encoder
			decoder_2 = self.decoder_2
		elif mul_num == 3:
			decoder = self.decoder2
			decoder_x = self.decoder_x2
			res_past_encoder = self.res_past_encoder2
			decoder_2 = self.decoder_22
		elif mul_num == 2:
			decoder = self.decoder3
			decoder_x = self.decoder_x3
			res_past_encoder = self.res_past_encoder3
			decoder_2 = self.decoder_23
		for i_track in range(num_sample):
			i_ind = sample_memory_index[:, i_track]
			i_wgh = (weight_read[:, i_track].unsqueeze(1)) * mul_num
			i_wgh = i_wgh.unsqueeze(1)
			feat_fut = memory_fut[torch.arange(0, len(i_ind)), i_ind]
			state_conc = torch.cat((state_past, feat_fut), 1)
			input_fut = state_conc
			prediction_y1 = decoder(input_fut).contiguous().view(-1, 1, 2)
			reconstruction_x1 = decoder_x(input_fut).contiguous().view(-1, int(self.past_len*mul_num/4), 2)

			diff_past = past - reconstruction_x1  # B, T, 2
			diff_past_embed = res_past_encoder(diff_past)  # B, F

			state_conc_diff = torch.cat((diff_past_embed, abs_past_state_social, feat_fut), 1)
			prediction_y2 = decoder_2(state_conc_diff).contiguous().view(-1, 1, 2)
            # reconstruction_x2 = self.decoder_2_x(state_conc_diff).contiguous().view(-1, self.past_len, 2)

			prediction_single = prediction_y1 + prediction_y2
			prediction_single = torch.cat((prediction_single, i_wgh), dim=2)
			prediction = torch.cat((prediction, prediction_single.unsqueeze(1)), dim=1)
		return prediction

	def get_destination_from_memory(self, past, abs_past, seq_start_end, end_pose, return_abs=False):
		# temporal encoding for past
		norm_past_state = self.norm_past_encoder(past)
		abs_past_state = self.abs_past_encoder(abs_past)
		abs_past_state_social = self.social_pooling_X(abs_past_state, seq_start_end, end_pose)
		mul_num = 4
		num_sample = 120

		state_past = torch.cat((norm_past_state, abs_past_state_social), dim=1)
		index_max, _ = self.get_memory_index(state_past, self.memory_past)

		memory_past = torch.Tensor().cuda()
		memory_fut = torch.Tensor().cuda()

		# state concatenation and decoding
		for i_track in range(200):
			i_ind = index_max[:, i_track]
			memory_past = torch.cat((memory_past, self.memory_past[i_ind].unsqueeze(1)), dim=1)
			memory_fut = torch.cat((memory_fut, self.memory_fut[i_ind].unsqueeze(1)), dim=1)

		state_past_selector = self.input_query_w(state_past).unsqueeze(1)
		memory_past_selector = self.past_memory_w(memory_past)
		sample_memory_index, weight_read = self.get_memory_index_batch(state_past_selector, memory_past_selector)

		# torch.multinomial
		prediction = self.prediction_generate(mul_num, num_sample, sample_memory_index, weight_read, memory_fut, past,
											  state_past, abs_past_state_social)
		# prediction = self.k_means(prediction.squeeze(2), ncluster=20, iter=10).unsqueeze(2)
		if return_abs:
			return prediction, abs_past_state_social
		return prediction

	def get_destination_from_memory2(self, past, abs_past, seq_start_end, end_pose, return_abs=False):
		# temporal encoding for past
		norm_past_state = self.norm_past_encoder2(past)
		abs_past_state = self.abs_past_encoder2(abs_past)
		abs_past_state_social = self.social_pooling_X2(abs_past_state, seq_start_end, end_pose)
		mul_num = 3
		num_sample = 180

		state_past = torch.cat((norm_past_state, abs_past_state_social), dim=1)
		index_max, _ = self.get_memory_index(state_past, self.memory_past2)

		memory_past = torch.Tensor().cuda()
		memory_fut = torch.Tensor().cuda()

		# state concatenation and decoding
		for i_track in range(250):
			i_ind = index_max[:, i_track]
			memory_past = torch.cat((memory_past, self.memory_past2[i_ind].unsqueeze(1)), dim=1)
			memory_fut = torch.cat((memory_fut, self.memory_fut2[i_ind].unsqueeze(1)), dim=1)

		state_past_selector = self.input_query_w2(state_past).unsqueeze(1)
		memory_past_selector = self.past_memory_w2(memory_past)
		sample_memory_index, weight_read = self.get_memory_index_batch(state_past_selector, memory_past_selector)

		# torch.multinomial
		prediction = self.prediction_generate(mul_num, num_sample, sample_memory_index, weight_read, memory_fut, past,
											  state_past, abs_past_state_social)
		# prediction = self.k_means(prediction.squeeze(2), ncluster=20, iter=10).unsqueeze(2)
		if return_abs:
			return prediction, abs_past_state_social
		return prediction

	def get_destination_from_memory3(self, past, abs_past, seq_start_end, end_pose, return_abs=False):
		# temporal encoding for past
		norm_past_state = self.norm_past_encoder3(past)
		abs_past_state = self.abs_past_encoder3(abs_past)
		abs_past_state_social = self.social_pooling_X3(abs_past_state, seq_start_end, end_pose)
		mul_num = 2
		num_sample = 240

		state_past = torch.cat((norm_past_state, abs_past_state_social), dim=1)
		index_max, _ = self.get_memory_index(state_past, self.memory_past3)

		memory_past = torch.Tensor().cuda()
		memory_fut = torch.Tensor().cuda()

		# state concatenation and decoding
		for i_track in range(300):
			i_ind = index_max[:, i_track]
			memory_past = torch.cat((memory_past, self.memory_past3[i_ind].unsqueeze(1)), dim=1)
			memory_fut = torch.cat((memory_fut, self.memory_fut3[i_ind].unsqueeze(1)), dim=1)

		state_past_selector = self.input_query_w3(state_past).unsqueeze(1)
		memory_past_selector = self.past_memory_w3(memory_past)
		sample_memory_index, weight_read = self.get_memory_index_batch(state_past_selector, memory_past_selector)

		# torch.multinomial
		prediction = self.prediction_generate(mul_num, num_sample, sample_memory_index, weight_read, memory_fut, past,
											  state_past, abs_past_state_social)
		# prediction = self.k_means(prediction.squeeze(2), ncluster=20, iter=10).unsqueeze(2)
		if return_abs:
			return prediction, abs_past_state_social
		return prediction

	def check_same(self, final_pred, pred_y, pred_x, i, k):
		for j in range(len(pred_y[k])):
			if pred_x[k][i][0] == pred_y[k][j][0] and pred_x[k][i][1] == pred_y[k][j][1]:
				final_pred[k][i][2] = final_pred[k][i][2] + pred_y[k][j][2]
				del (pred_y[k][j])
				break
		return final_pred, pred_y

	def sort_weight(self, final_pred, pred_x, pred_y, ther):
		for k in range(len(pred_x)):
			del (final_pred[k][0])
			for i in range(int(len(pred_x[k]) * 0.8)):
				final_pred[k].append(pred_x[k][i])
				final_pred, pred_y, = self.check_same(final_pred, pred_y, pred_x, i, k)
		for k in range(len(pred_y)):
			for i in range(ther - int(len(pred_x[k]) * 0.8)):
				final_pred[k].append(pred_y[k][i])

	def soft_argmax(self, final_pred):
		index_max = final_pred[:, :, 2]
		prediction_ww = nn.functional.softmax(index_max * 10, dim=1)
		indices = torch.arange(start=0, end=len(prediction_ww[0])).unsqueeze(0).cuda()
		prediction_index_max = prediction_ww * indices
		prediction_index_max = prediction_index_max.sum(dim=1)
		prediction_max = torch.tensor([]).cuda()
		# return prediction_index_max
		for i in range(prediction_index_max.shape[0]):
			prd = final_pred[i, prediction_index_max[i].type(torch.long), :2]
			prediction_max = torch.cat((prediction_max, prd), dim=0)
		return prediction_max.contiguous().view(-1, 1, 2)

	#  traj_demo1
	def prediction_mix(self, prediction_1, prediction_2, prediction_3):
		pred_1 = prediction_1.cpu().numpy().tolist()
		pred_2 = prediction_2.cpu().numpy().tolist()
		pred_3 = prediction_3.cpu().numpy().tolist()
		mid_pred = torch.Tensor(size=(len(pred_1), 1))
		mid_pred = mid_pred.numpy().tolist()
		final_pred = torch.Tensor(size=(len(pred_1), 1))
		final_pred = final_pred.numpy().tolist()
		self.sort_weight(mid_pred, pred_1, pred_2, ther=150)
		self.sort_weight(final_pred, mid_pred, pred_3, ther=200)
		# final_pred.shape(num, ther, 3)
		final_pred = torch.tensor(final_pred).squeeze(2).cuda()
		print(final_pred.shape)
		prediction_f_max = self.soft_argmax(final_pred)
		final_pred = final_pred[:, :, :2]
		prediction_f = self.k_means(final_pred.squeeze(2), ncluster=19, iter=10).cuda()
		prediction_f = prediction_f.cuda()
		prediction_f_max = prediction_f_max.cuda()
		prediction_f_max = torch.cat((prediction_f_max, prediction_f), dim=1).unsqueeze(2)
		return prediction_f_max

	def k_means(self, batch_x, ncluster=20, iter=10):
		"""return clustering n cluster of x.
		Args:
			x (Tensor): B, K, 2
			ncluster (int, optional): Number of clusters. Defaults to 20.
			iter (int, optional): Number of iteration to get the centroids. Defaults to 10.
		"""
		B, N, D = batch_x.size()
		batch_c = torch.Tensor().cuda()
		for i in range(B):
			x = batch_x[i]
			c = x[torch.randperm(N)[:ncluster]]
			for i in range(iter):
				a = ((x[:, None, :] - c[None, :, :]) ** 2).sum(-1).argmin(1)
				c = torch.stack([x[a == k].mean(0) for k in range(ncluster)])
				nanix = torch.any(torch.isnan(c), dim=1)
				ndead = nanix.sum().item()
				c[nanix] = x[torch.randperm(N)[:ndead]]

			batch_c = torch.cat((batch_c, c.unsqueeze(0)), dim=0)
		return batch_c

	def get_trajectory(self, past, abs_past, past2, abs_past2, past3, abs_past3, seq_start_end, end_pose):
		prediction = torch.Tensor().cuda()
		destination_prediction1, abs_past_state_social = self.get_destination_from_memory(past, abs_past, seq_start_end,end_pose, return_abs=True)
		destination_prediction2, _ = self.get_destination_from_memory2(past2, abs_past2, seq_start_end,
																	   end_pose, return_abs=True)
		destination_prediction3, _ = self.get_destination_from_memory3(past3, abs_past3, seq_start_end,
																	   end_pose, return_abs=True)

		destination_prediction = self.prediction_mix(destination_prediction1, destination_prediction2,
													 destination_prediction3)
		destination_prediction = destination_prediction.squeeze(2)
		return destination_prediction

	def fix_process_to_get_destination(self, past, abs_past, seq_start_end, end_pose):

		b1, T, d = abs_past.size()  # (num, 8, 2)
		prediction = torch.Tensor()
		if self.use_cuda:
			prediction = prediction.cuda()

		# temporal encoding for past
		norm_past_state = self.norm_past_encoder(past)
		abs_past_state = self.abs_past_encoder(abs_past)
		abs_past_state_social = self.social_pooling_X(abs_past_state, seq_start_end, end_pose)

		state_past = torch.cat((norm_past_state, abs_past_state_social), dim=1)
		index_max, _ = self.get_memory_index(state_past, self.memory_past)

		memory_past = torch.Tensor().cuda()
		memory_fut = torch.Tensor().cuda()

		# state concatenation and decoding
		for i_track in range(200):
			i_ind = index_max[:, i_track]
			memory_past = torch.cat((memory_past, self.memory_past[i_ind].unsqueeze(1)), dim=1)
			memory_fut = torch.cat((memory_fut, self.memory_fut[i_ind].unsqueeze(1)), dim=1)

		state_past_selector = self.input_query_w(state_past).unsqueeze(1)
		memory_past_selector = self.past_memory_w(memory_past)

		# state_past_selector = state_past.unsqueeze(1)
		# memory_past_selector = memory_past.clone()

		sample_memory_index, weight_read = self.get_memory_index_batch(state_past_selector, memory_past_selector)

		for i_track in range(120):
			i_ind = sample_memory_index[:, i_track]
			feat_fut = memory_fut[torch.arange(0, len(i_ind)), i_ind]
			state_conc = torch.cat((state_past, feat_fut), 1)
			input_fut = state_conc
			prediction_y1 = self.decoder(input_fut).contiguous().view(-1, 1, 2)
			reconstruction_x1 = self.decoder_x(input_fut).contiguous().view(-1, self.past_len, 2)

			diff_past = past - reconstruction_x1  # B, T, 2
			diff_past_embed = self.res_past_encoder(diff_past)  # B, F

			state_conc_diff = torch.cat((diff_past_embed, abs_past_state_social, feat_fut), 1)
			prediction_y2 = self.decoder_2(state_conc_diff).contiguous().view(-1, 1, 2)
			# reconstruction_x2 = self.decoder_2_x(state_conc_diff).contiguous().view(-1, self.past_len, 2)

			prediction_single = prediction_y1 + prediction_y2
			prediction = torch.cat((prediction, prediction_single.unsqueeze(1)), dim=1)

		prediction = self.k_means(prediction.squeeze(2), ncluster=20, iter=10)
		return prediction

	def forward(self, past, abs_past, past2, abs_past2, past3, abs_past3, seq_start_end, end_pose):
		if self.mode == 'trajectory':
			destination_prediction = self.fix_process_to_get_destination(past, abs_past, seq_start_end, end_pose)
			return destination_prediction
