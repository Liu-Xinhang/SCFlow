"""
这个文件用于我来测试一下他位姿估计的准确的。
"""

import numpy as np
import torch
import random
import cv2
import tqdm
import os
from matplotlib import image as mpimg
import json
import trimesh


class AddsAndShift():
	def __init__(self, model_points, diameter, device, K, debug = False, debug_dir=None):
		self._total_add = []
		self._total_adds = []
		self._total_shift = []
		self._bbx3d = AddsAndShift._get3d_boundingbox(model_points)
		self._model_full = model_points
		dellist = [j for j in range(0, len(model_points))]
		dellist = random.sample(dellist, len(model_points) - 2000)
		model_points = np.delete(model_points, dellist, axis=0)
		self.model_points = model_points
		self.diameter = diameter
		self.model = torch.from_numpy(model_points.astype(np.float32)).to(device)
		self.device = device
		self._add = 0
		self._adds = 0
		self._shift = 0
		self._current_frame = 0
		self._K = K
		self._debug_dir = debug_dir if debug_dir is not None else 'debug/pose'
		self._debug = debug
		self._current_add = 0
		self._current_adds = 0
		self._current_shift = 0
		if debug:
			os.makedirs(self._debug_dir, exist_ok=True)
	

	def VOCap(self, rec):
		rec = np.sort(np.array(rec) / self.diameter) ## 使用n.array将进行深拷贝
		n = len(rec)
		prec = np.arange(1,n+1) / float(n)
		rec = rec.reshape(-1)
		prec = prec.reshape(-1)
		index = np.where(rec<0.1)[0]
		rec = rec[index]
		prec = prec[index]

		mrec=[0, *list(rec), 0.1]
		mpre=[0, *list(prec), prec[-1]]

		for i in range(1,len(mpre)):
			mpre[i] = max(mpre[i], mpre[i-1])
		mpre = np.array(mpre)
		mrec = np.array(mrec)
		i = np.where(mrec[1:]!=mrec[0:len(mrec)-1])[0] + 1
		ap = np.sum((mrec[i] - mrec[i-1]) * mpre[i]) * 10
		return ap
	
	def refresh(self):
		self._add = 0
		self._adds = 0
		self._shift = 0
		self._current_frame = 0
		self._current_add = 0
		self._current_adds = 0
		self._current_shift = 0
		self._total_add = []
		self._total_adds = []
		self._total_shift = []

	@staticmethod
	def omit_z_direction(pose_gt, pose_pr):
		"""
		返回去除掉z方向的位姿
		"""
		pose_gt_euler = R.from_matrix(pose_gt[:3, :3]).as_euler("ZYX")
		pose_gt_euler[0] = 0 ## 将Z方向固定为0
		pose_pred_euler = R.from_matrix(pose_pr[:3, :3]).as_euler("ZYX")
		pose_pred_euler[0] = 0
		return R.from_euler("ZYX", pose_gt_euler).as_matrix(), R.from_euler("ZYX", pose_pred_euler).as_matrix()

	def add_new_frame(self, pose_gt, pose_pr, save_pose=False, save_img=False, use_ground_truth_rotation=False, img_path=None, sequence_name=None, isvalid=True, using_bounding_box=True, image_name=None, omit_z_direction=False):
		## 	去除掉z方向的自由度
		if omit_z_direction:
			pose_gt[:3, :3], pose_pr[:3, :3] = self.omit_z_direction(pose_gt, pose_pr)
			# print(pose_gt, pose_pr)
		## 计算add和adds指标
		if not isvalid:
			self._current_add = np.nan
			self._current_adds = np.nan
		else:
			self._current_add = self._compute_add(pose_gt, pose_pr)
			self._current_adds = self._compute_adds(pose_gt, pose_pr)
		if self._current_add <= 0.1 * self.diameter:
			self._add += 1
		if self._current_adds <= 0.1 * self.diameter:
			self._adds += 1
		self._total_add.append(self._current_add)
		self._total_adds.append(self._current_adds)
		## 计算2d shift指标
		if not isvalid:
			self._current_shift = np.nan
		else:
			self._current_shift = self._compute_2d_shift(pose_gt, pose_pr)
		if self._current_shift <= 5:
			self._shift += 1
		self._total_shift.append(self._current_shift)
		# self._add_xyz(pose_gt, pose_pr)
		if save_pose:
			if image_name is None:
				image_name = os.path.basename(img_path).spilt('.')[0]
			self._save_pose(pose_gt, pose_pr, sequence_name=sequence_name, image_name=image_name)
		if save_img:
			self._draw_bbx(pose_pr, pose_gt, use_ground_truth_rotation=use_ground_truth_rotation, img_path=img_path, sequence_name=sequence_name, using_bounding_box=using_bounding_box)
		self._current_frame += 1
		if self._current_frame % 50 == 0:
			add, adds = self.get_add_and_adds()
			# add_auc, adds_auc = self.get_AUC()
			print("add: {:.4f}, adds: {:.4f}".format(add, adds))
	
	def add_new_pose(self, pose_path, save_pose=False, save_img=False, use_ground_truth_rotation=False, img_path=None):
		"""
		用于评估提前保存好的序列的位姿
		其中pose_path中存储的是真实的位姿和预测的位姿，真实位姿在前3行，预测位姿在后3行
		"""
		pose = np.loadtxt(pose_path)
		if pose.shape[0] == 6:
			pose_gt = pose[:3]
			pose_pr = pose[3:]
		else:
			pose_gt = pose[:3]
			pose_pr = pose[4:7]
		self.add_new_frame(pose_gt, pose_pr, save_pose=save_pose, save_img=save_img, use_ground_truth_rotation=use_ground_truth_rotation, img_path=img_path)
		
	def get_shift(self, ratio=True):
		shift = self._shift
		if ratio:
			shift = shift / len(self._total_shift)
		return shift
	
	def get_AUC(self):
		if len(self._total_add) == 0:
			return 0, 0
		add_aps = self.VOCap(self._total_add)
		adds_aps = self.VOCap(self._total_adds)
		return add_aps, adds_aps

	def get_add_and_adds(self, ratio=True):
		add = self._add
		adds = self._adds
		if ratio and len(self._total_add) > 0:
			add = add / len(self._total_add)
			adds = adds / len(self._total_adds)
		return add, adds

	def _move_avg(self, a, n=10, mode="valid"):
		return np.convolve(a, np.ones((n, ))/n, mode)
	
	def _add_xyz(self, pose_gt, pose_pred):
		x_gt = pose_gt[0, 3]
		y_gt = pose_gt[1, 3]
		z_gt = pose_gt[2, 3]

		x_pred= pose_pred[0, 3]
		y_pred = pose_pred[1, 3]
		z_pred = pose_pred[2, 3]

		self._total_gt_x.append(x_gt)
		self._total_gt_y.append(y_gt)
		self._total_gt_z.append(z_gt)

		self._total_pred_x.append(x_pred)
		self._total_pred_y.append(y_pred)
		self._total_pred_z.append(z_pred)
	
	def get_total_xyz(self, smooth=False):
		if smooth:
			return map(lambda x: self._move_avg(np.array(x)), (self._total_gt_x, self._total_gt_y, self._total_gt_z, self._total_pred_x, self._total_pred_y, self._total_pred_z))
		return self._total_gt_x, self._total_gt_y, self._total_gt_z, self._total_pred_x, self._total_pred_y, self._total_pred_z

	def get_current_add_adds_shift(self):
		return self._current_add, self._current_adds, self._current_shift
	
	def get_total_shift(self, smooth=False):
		if smooth:
			return self._move_avg(np.array(self._total_shift))
		return np.array(self._total_shift)

	def get_total_add_and_adds(self, smooth=False):
		if smooth:
			return self._move_avg(np.array(self._total_add)), self._move_avg(np.array(self._total_adds))
		return np.array(self._total_add), np.array(self._total_adds)

	def _save_pose(self, pose_gt, pose_pr, combine=False, sequence_name=None, image_name=None):

		os.makedirs(os.path.join(self._debug_dir, sequence_name, "poses"), exist_ok=True)
		if self._debug == False:
			raise RuntimeError('You can only store pose when debug mode is open')
		if combine:
			total_pose = np.concatenate([pose_gt[:3], pose_pr[:3]], axis=0)
		else:
			total_pose = pose_pr[:3]
		if image_name is None:
			save_path = f'{self._current_frame:05d}.txt'
		else:
			save_path = image_name.split(".")[0] + ".txt"
		np.savetxt(os.path.join(self._debug_dir, sequence_name, "poses", save_path), total_pose)
	
	def _compute_2d_shift(self, pose_gt, pose_pr):
		pred = self._K @ (pose_pr[:3, :3] @ self.model_points.T + pose_pr[:, 3:])
		target = self._K @ (pose_gt[:3, :3] @ self.model_points.T + pose_gt[:, 3:])
		pred = (pred / pred[-1])[:2] # 2 * N
		target = (target / target[-1])[:2] # 2 * N
		dis = np.mean(np.linalg.norm(pred - target, axis=0))
		return dis

	def _compute_add(self, pose_gt, pose_pr):
		pred = np.dot(self.model_points, pose_pr[:3, :3].T) + pose_pr[:3, 3]
		target = np.dot(self.model_points, pose_gt[:3, :3].T) + pose_gt[:3, 3]
		dis_add = np.mean(np.linalg.norm(pred - target, axis=1))
		return dis_add

	def _compute_adds(self, pose_target, pose_pred):
		"""
		adds 指标， 放在GPU上面计算,加速;参考pvn3d
		@param pose_pred:
		@param pose_target:
		@param model:
		@param diameter:
		@param percentage:
		@return:
		"""
		N = self.model.shape[0]  # 3D模型点数
		
		pose_pred = torch.from_numpy(pose_pred.astype(np.float32))
		pose_target = torch.from_numpy(pose_target.astype(np.float32))
		pose_pred = pose_pred.to(self.device)
		pose_target = pose_target.to(self.device)

		model_pred = torch.mm(self.model, pose_pred[:, :3].permute(1, 0)) + pose_pred[:, 3]
		model_pred = model_pred.view(1, N, 3).repeat(N, 1, 1)
		model_target = torch.mm(self.model, pose_target[:, :3].permute(1, 0)) + pose_target[:, 3]
		model_target = model_target.view(N, 1, 3).repeat(1, N, 1)
		distance = torch.norm(model_pred - model_target, dim=2)
		mindistance = torch.min(distance, dim=1)[0]
		mean_mindistance = torch.mean(mindistance)
		mean_mindistance_np = mean_mindistance.cpu().numpy()
		return mean_mindistance_np.item()

	@staticmethod
	def _get3d_boundingbox(coordinate3d):
		"""generating 3d bounding box

		Parameters
		----------
		coordinate3d : n * 3
		"""
		assert coordinate3d.shape[1] == 3

		min_each_column = np.min(coordinate3d, axis=0)
		max_each_column = np.max(coordinate3d, axis=0)

		corner1 = (min_each_column[0],min_each_column[1], min_each_column[2])
		corner2 = (max_each_column[0],min_each_column[1], min_each_column[2])
		corner3 = (max_each_column[0],max_each_column[1], min_each_column[2])
		corner4 = (min_each_column[0],max_each_column[1], min_each_column[2])
		corner5 = (min_each_column[0],min_each_column[1], max_each_column[2])
		corner6 = (max_each_column[0],min_each_column[1], max_each_column[2])
		corner7 = (max_each_column[0],max_each_column[1], max_each_column[2])
		corner8 = (min_each_column[0],max_each_column[1], max_each_column[2])

		return np.stack((corner1, corner2, corner3, corner4, corner5, corner6, corner7, corner8))
	
	@staticmethod
	def _visulization_by_bounding_cubic(image, pose, intrinsic_matrix, corner, color=(255, 0, 0)):
		"""This function visulize the pose by a cubic bounding box. 
		Parameters
		----------
		image : ndarray
			Image to draw line on.
		pose : ndarray
			Pose matrix, which is 3 * 4 matrix
		intrinsic_matrix : ndarray
			Intrinsic matrix of a camera, which is 3 * 3 matrix.
		corner : ndarray
			The 8 corners of the cubic maxtrix, which is 8 * 3 matrix, each
			line represents the 3d coordinates of a corner.
		color : tuple, optional
			The color tuple, by default (0, 0, 255).
		Returns
		-------
		ndarray
			Image with line drawing on.
		"""

		def drawline(img,pt1,pt2,color,thickness=1,style='dotted',gap=20): 
			dist =((pt1[0]-pt2[0])**2+(pt1[1]-pt2[1])**2)**.5 
			pts= [] 
			for i in np.arange(0,dist,gap): 
				r=i/dist 
				x=int((pt1[0]*(1-r)+pt2[0]*r)+.5) 
				y=int((pt1[1]*(1-r)+pt2[1]*r)+.5) 
				p = (x,y) 
				pts.append(p) 
		
			if style=='dotted': 
				for p in pts: 
					cv2.circle(img,p,thickness,color,-1) 
			else: 
				s=pts[0] 
				e=pts[0] 
				i=0 
				for p in pts: 
					s=e 
					e=p 
					if i%2==1: 
						cv2.line(img,s,e,color,thickness) 
					i+=1 

		corner = np.concatenate([corner, np.ones((len(corner), 1))],axis=1)
		points = intrinsic_matrix @ pose @ corner.T
		points_uv = points[:2] / points[2]
		points_uv = np.floor(points_uv).astype(np.int32)
		points_uv = points_uv.T


		image = cv2.line(image, points_uv[0], points_uv[1], color, 2)
		image = cv2.line(image, points_uv[1], points_uv[2], color, 2)
		image = cv2.line(image, points_uv[3], points_uv[2], color, 2)
		image = cv2.line(image, points_uv[3], points_uv[0], color, 2)
		# image = cv2.line(image, points_uv[4], points_uv[5], shallow_color, 2)
		# image = cv2.line(image, points_uv[5], points_uv[6], shallow_color, 2)
		# image = cv2.line(image, points_uv[7], points_uv[6], shallow_color, 2)
		# image = cv2.line(image, points_uv[7], points_uv[4], shallow_color, 2)
		drawline(image, points_uv[4], points_uv[5], color, 2, 'dotted', 20)
		drawline(image, points_uv[5], points_uv[6], color, 2, 'dotted', 20)
		drawline(image, points_uv[6], points_uv[7], color, 2, 'dotted', 20)
		drawline(image, points_uv[7], points_uv[4], color, 2, 'dotted', 20)
		image = cv2.line(image, points_uv[4], points_uv[0], color, 2)
		image = cv2.line(image, points_uv[7], points_uv[3], color, 2)
		image = cv2.line(image, points_uv[1], points_uv[5], color, 2)
		image = cv2.line(image, points_uv[2], points_uv[6], color, 2)
		return image
	
	@staticmethod
	def _visulization_by_reprojection(image, pose, intrinsic_matrix, points, color=(255, 0, 0)):
		"""This function visulize the pose by a cubic bounding box. 
		Parameters
		----------
		image : ndarray
			Image to draw line on.
		pose : ndarray
			Pose matrix, which is 3 * 4 matrix
		points: np.ndarray
			3d points to be projected
		intrinsic_matrix : ndarray
			Intrinsic matrix of a camera, which is 3 * 3 matrix.
		color : tuple, optional
			The color tuple, by default (0, 0, 255).
		Returns
		-------
		ndarray
			Image with line drawing on.
		"""
		model_3d = np.concatenate([points, np.ones((len(points), 1))],axis=1)
		points = intrinsic_matrix @ pose @ model_3d.T
		points_uv = points[:2] / points[2]
		points_uv = np.floor(points_uv).astype(np.int32)
		points_uv = points_uv.T # N, 2

		for point in points_uv:
			image = cv2.circle(image, tuple(point), 2, color, -1)
		
		return image

	def _draw_bbx(self, pose_pr, pose_gt, use_ground_truth_rotation=False, img_path=None, sequence_name=None, using_bounding_box=True):
		os.makedirs(os.path.join(self._debug_dir, sequence_name, "projection"), exist_ok=True)
		if img_path is not None:
			fake_image = cv2.imread(img_path, cv2.IMREAD_COLOR)
			fake_image = cv2.cvtColor(fake_image, cv2.COLOR_BGR2RGB)
		else:
			fake_image = np.zeros((480, 640, 3), dtype=np.uint8) ## same dimention as the real image
		if use_ground_truth_rotation:
			pose_pr_ = pose_pr.copy()
			pose_pr_[:, :3] = pose_gt[:, :3]
			if using_bounding_box:
				image_with_bbx = AddsAndShift._visulization_by_bounding_cubic(fake_image, pose_pr_, self._K, self._bbx3d)
			else:
				image_with_bbx = AddsAndShift._visulization_by_reprojection(fake_image, pose_pr_, self._K, self._model_full)
		else:
			if using_bounding_box:
				image_with_bbx = AddsAndShift._visulization_by_bounding_cubic(fake_image, pose_pr, self._K, self._bbx3d)
			else:	
				image_with_bbx = AddsAndShift._visulization_by_reprojection(fake_image, pose_pr, self._K, self._model_full)
		if using_bounding_box:
			image_with_bbx = AddsAndShift._visulization_by_bounding_cubic(image_with_bbx, pose_gt, self._K, self._bbx3d, color=(0, 255, 0))
		else:
			image_with_bbx = AddsAndShift._visulization_by_reprojection(image_with_bbx, pose_gt, self._K, self._model_full, color=(0, 255, 0))
		image_save_path = os.path.join(self._debug_dir, sequence_name, "projection", f"{self._current_frame:06d}.png")
		mpimg.imsave(image_save_path, image_with_bbx)


def load_all_poses(pred_pose_dir, obj_id):
		total_pred_poses = dict()
		for i in range(48, 60):
			pred_pose = dict()
			scene_pred_path = os.path.join(pred_pose_dir, f"{i:06d}/scene_gt.json")

			## 提取预测的结果
			with open(scene_pred_path, 'r') as f:
				scene_pred = json.load(f)
			
			for key, value in scene_pred.items():
				res = list(filter(lambda x: x['obj_id'] == obj_id, value))
				if len(res) == 0: ## 当前序列没有这个内容
					break
				else:
					assert len(res) == 1
					rotation = np.array(res[0]['cam_R_m2c']).reshape(3, 3)
					translation = np.array(res[0]['cam_t_m2c']).reshape(3, 1)
					pose = np.concatenate([rotation, translation], axis=1)
					pred_pose[str(i)+"_"+key] = {"pred_pose": pose}
			 
			## 提取真实的结果
			scene_gt_path = os.path.join("data/ycbv/test", f"{i:06d}/scene_gt.json")
			with open(scene_gt_path, 'r') as f:
				scene_gt = json.load(f)
			
			for key, value in pred_pose.items(): ## 如果pred_pose中有内容，进入循环
				total_poses = scene_gt[key.split("_")[-1]]
				res = list(filter(lambda x: x['obj_id'] == obj_id, total_poses))
				assert len(res) == 1, print(i, res)
				rotation = np.array(res[0]['cam_R_m2c']).reshape(3, 3)
				translation = np.array(res[0]['cam_t_m2c']).reshape(3, 1)
				pose = np.concatenate([rotation, translation], axis=1)
				value["gt_pose"] = pose
			total_pred_poses.update(pred_pose)
		return total_pred_poses
	

if __name__ == "__main__":
	## 读取模型信息
	with open("data/ycbv/models_eval/models_info.json") as file:
		model_information = json.load(file)
	
	## 读取相机内参
	K = np.array([1066.778, 0.0, 312.9869, 0.0, 1067.487, 241.3109, 0.0, 0.0, 1.0]).reshape(3, 3)

	all_results = dict()

	for model_id in range(1, 22):
		diameter = model_information[str(model_id)]["diameter"]
		total_pred_poses = load_all_poses("results", model_id)

		## 读取模型
		mesh = trimesh.load(f"data/ycbv/models_eval/obj_{model_id:06d}.ply")
		point_cloud = mesh.sample(10000) ## 均匀采样10000个点


		add_and_shift = AddsAndShift(point_cloud, diameter, "cpu", K, True, f"debug/{model_id:02d}")

		sample_key = random.sample(total_pred_poses.keys(), 10)
		total_pred_poses = {key: total_pred_poses[key] for key in sample_key}

		with tqdm.tqdm(range(len(total_pred_poses))) as pbar:
			for key in total_pred_poses.keys():
				pred_pose = total_pred_poses[key]["pred_pose"]
				gt_pose = total_pred_poses[key]["gt_pose"]
				img_path = os.path.join("data/ycbv/test", f"{int(key.split('_')[0]):06d}", "rgb", f"{int(key.split('_')[1]):06d}.png")
				add_and_shift.add_new_frame(gt_pose, pred_pose, save_img=True, img_path = img_path, sequence_name=key.split("_")[0])
				pbar.update(1)
		
		add, adds = add_and_shift.get_add_and_adds()
		# add_auc, adds_auc = add_and_shift.get_AUC()

		print(f"objid: {model_id}, add: {add}, adds: {adds}")
		
		all_results[model_id] = {"add": add, "adds": adds}
	
	## 计算平均值

	total_add = [all_results[i]["add"] for i in all_results.keys()]
	total_adds = [all_results[i]["adds"] for i in all_results.keys()]

	print("total_add: ", np.mean(total_add))
	print("total_adds: ", np.mean(total_adds))
	with open("all_results.json", 'w') as f:
		json.dump(all_results, f)
