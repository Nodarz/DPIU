import os
import datetime
import torch
import torch.nn as nn
# from idea_test_demo3 import model_traj
from models.model_AIO_demo import model_encdec
from sddloader_test import *

torch.set_num_threads(1)


class Trainer:
	def __init__(self, config):

		# test folder creating
		self.name_test = str(datetime.datetime.now())[:10]
		self.folder_test = 'testing/' + self.name_test + '_' + config.info
		if not os.path.exists(self.folder_test):
			os.makedirs(self.folder_test)
		self.folder_test = self.folder_test + '/'

		self.test_dataset = SocialDataset(set_name="test", b_size=config.test_b_size, t_tresh=config.time_thresh,
										  d_tresh=config.dist_thresh)

		if torch.cuda.is_available(): torch.cuda.set_device(config.gpu)

		self.settings = {
			"mode": config.mode,
			"train_batch_size": config.train_b_size,
			"test_batch_size": config.test_b_size,
			"use_cuda": config.cuda,
			"dim_feature_tracklet": config.past_len * 2,
			"dim_feature_future": config.future_len * 2,
			"dim_embedding_key": config.dim_embedding_key,
			"past_len": config.past_len,
			"future_len": 12,
		}

		# model
		self.model_ae1 = torch.load(config.model_ae1, map_location='cuda:0')
		self.model_ae2 = torch.load(config.model_ae2, map_location='cuda:0')
		self.model_ae3 = torch.load(config.model_ae3, map_location='cuda:0')
		self.mem_n2n = model_encdec(self.settings, self.model_ae1, self.model_ae2, self.model_ae3)

		if config.cuda:
			self.mem_n2n = self.mem_n2n.cuda()

		self.start_epoch = 0
		self.config = config

		self.device = torch.device('cuda') if config.cuda else torch.device('cpu')

	# 记录模型变量
	def print_model_param(self, model):
		# numel()函数：返回数组中元素的个数
		total_num = sum(p.numel() for p in model.parameters())
		trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
		print("\033[1;31;40mTrainable/Total: {}/{}\033[0m".format(trainable_num, total_num))
		return 0

	#
	def fit(self):

		dict_metrics_test = self.evaluate(self.test_dataset)
		print('Test FDE_48s: {} ------ origin FDE: {}'.format(dict_metrics_test['fde_48s'], dict_metrics_test['fde_48s1']))
		print('-' * 100)

	# 评价
	def evaluate(self, dataset):
		# 初始化评价基准
		fde_48s = fde_48s1 = 0
		ade_48s = ade_48s1 = 0
		samples = 0
		dict_metrics = {}

		with torch.no_grad():
			for i, (traj, mask, initial_pos, seq_start_end) \
					in enumerate(zip(dataset.trajectory_batches, dataset.mask_batches, dataset.initial_pos_batches,
									 dataset.seq_start_end_batches)):
				traj, mask, initial_pos = torch.FloatTensor(traj).to(self.device), \
										  torch.FloatTensor(mask).to(self.device), \
										  torch.FloatTensor(initial_pos).to(self.device)
				# traj (B, T, 2)
				initial_pose = traj[:, 7, :] / 1000

				traj_norm = traj - traj[:, 7:8, :]
				#  使第八个点为坐标原点，依次划分过去和将来


				x = traj_norm[:, :self.config.past_len, :]
				abs_past = traj[:, :self.config.past_len, :]
				x2 = traj_norm[:, 2:self.config.past_len, :]
				abs_past2 = traj[:, 2:self.config.past_len, :]
				x3 = traj_norm[:, 4:self.config.past_len, :]
				abs_past3 = traj[:, 4:self.config.past_len, :]


				output = self.mem_n2n.get_trajectory(x, abs_past, x2, abs_past2, x3, abs_past3, seq_start_end, initial_pose)
				output_b = self.mem_n2n(x, abs_past, x2, abs_past2, x3, abs_past3, seq_start_end, initial_pose)
                
				output = output.unsqueeze(2)
				output = output.data
				output_b = output_b.unsqueeze(2)
				output_b = output_b.data
				# B, K, t, 2
				print(output.shape)

				destination = traj_norm[:, -1:, :].unsqueeze(1).repeat(1, 20, 1, 1)
				future_rep = traj_norm[:, 8:, :].unsqueeze(1).repeat(1, 20, 1, 1)
                
                
				distances = torch.norm(output - destination, dim=3)
				mean_distances = torch.mean(distances[:, :, -1:], dim=2)
				index_min = torch.argmin(mean_distances, dim=1)
				min_distances = distances[torch.arange(0, len(index_min)), index_min]

				distancesb = torch.norm(output_b - destination, dim=3)
				mean_distancesb = torch.mean(distancesb[:, :, -1:], dim=2)
				index_minb = torch.argmin(mean_distancesb, dim=1)
				min_distancesb = distancesb[torch.arange(0, len(index_minb)), index_minb]

                
				fde_48s += torch.sum(min_distances[:, -1])
                
				fde_48s1 += torch.sum(min_distancesb[:, -1])

				samples += distancesb.shape[0]

			dict_metrics['fde_48s'] = fde_48s / samples
			dict_metrics['fde_48s1'] = fde_48s1 / samples

		return dict_metrics
