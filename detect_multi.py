#modified based on 
# https://github.com/Rajat-Mehta/yolov3_pytorch/blob/master/detect.py for 4 channels
# https://github.com/pieterbl86/yolov3/commit/687babbb7df73706428767d41c2b58dd1b18a257 for 1 channel
import argparse

from statistics import mean
from models import *  # set ONNX_EXPORT in models.py
from utils.datasets_multi import *
from utils.utils import *


def detect(cfg, save_img=False):
	imgsz = (320, 192) if ONNX_EXPORT else opt.img_size  # (320, 192) or (416, 256) or (608, 352) for (height, width)
	out, source, weights, half, view_img, save_txt = opt.output, opt.source, opt.weights, opt.half, opt.view_img, opt.save_txt
	webcam = source == '0' or source.startswith('rtsp') or source.startswith('http') or source.endswith('.txt')

	#initialize channel and cfg
	module_defs = parse_model_cfg(cfg)
	nchannels = module_defs[0]['channels']
	# Initialize
	device = torch_utils.select_device(device='cpu' if ONNX_EXPORT else opt.device)
	if os.path.exists(out):
		shutil.rmtree(out)  # delete output folder
	os.makedirs(out)  # make new output folder


	# Initialize model
	model = Darknet(opt.cfg, imgsz)

	# Load weights
	attempt_download(weights)
	if weights.endswith('.pt'):  # pytorch format
		model.load_state_dict(torch.load(weights, map_location=device)['model'])
	else:  # darknet format
		load_darknet_weights(model, weights)

	# Second-stage classifier
	classify = False
	if classify:
		modelc = torch_utils.load_classifier(name='resnet101', n=2)  # initialize
		modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model'])  # load weights
		modelc.to(device).eval()

	# Eval mode
	model.to(device).eval()

	# Fuse Conv2d + BatchNorm2d layers
	# model.fuse()

	# Export mode
	if ONNX_EXPORT:
		model.fuse()
		img = torch.zeros((1, 3) + imgsz)  # (1, 3, 320, 192)
		f = opt.weights.replace(opt.weights.split('.')[-1], 'onnx')  # *.onnx filename
		torch.onnx.export(model, img, f, verbose=False, opset_version=11,
						  input_names=['images'], output_names=['classes', 'boxes'])

		# Validate exported model
		import onnx
		model = onnx.load(f)  # Load the ONNX model
		onnx.checker.check_model(model)  # Check that the IR is well formed
		print(onnx.helper.printable_graph(model.graph))  # Print a human readable representation of the graph
		return

	# Half precision
	half = half and device.type != 'cpu'  # half precision only supported on CUDA
	if half:
		model.half()

	# Set Dataloader
	vid_path, vid_writer = None, None
	if webcam:
		view_img = True
		torch.backends.cudnn.benchmark = True  # set True to speed up constant image size inference
		dataset = LoadStreams(source, img_size=imgsz)
	else:
		save_img = True
		dataset = LoadImages(source, nchannels, img_size=imgsz)

	# Get names and colors
	names = load_classes(opt.names)
	colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]

	# Run inference
	t0 = time.time()
	comp_time = [] #list for all computation time
	img = torch.zeros((1, nchannels, imgsz, imgsz), device=device)  # init img
	_ = model(img.half() if half else img.float()) if device.type != 'cpu' else None  # run once
	for path_rgb, path_ir, img, im0s, vid_cap in dataset:
		img = torch.from_numpy(img.copy()).to(device)
		img = img.half() if half else img.float()  # uint8 to fp16/32
		img /= 255.0  # 0 - 255 to 0.0 - 1.0
		if img.ndimension() == 3:
			img = img.unsqueeze(0)

		# Inference
		t1 = torch_utils.time_synchronized()
		pred = model(img, augment=opt.augment)[0]
		t2 = torch_utils.time_synchronized()

		# to float
		if half:
			pred = pred.float()

		# Apply NMS
		pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres,
								   multi_label=False, classes=opt.classes, agnostic=opt.agnostic_nms)

		# Apply Classifier
		if classify:
			pred = apply_classifier(pred, modelc, img, im0s)

		# Process detections
		for i, det in enumerate(pred):  # detections for image i
			if webcam:  # batch_size >= 1
				p, s, im0 = path[i], '%g: ' % i, im0s[i].copy()
			else:
				s = ''
				im0 = im0s

				if nchannels == 4:
					p_rgb = path_rgb
					p_ir = path_ir

					filename_rgb = str(Path(p_rgb).name).split(".") #split filename.png
					# print("filename_rgb before", filename_rgb)
					filename_rgb = "".join((filename_rgb[0],"_rgb.", filename_rgb[1])) #join filename, _rgb, and .png
					# print("filename_rgb after", filename_rgb)
					save_path_rgb = str(Path(out) / Path(filename_rgb))

					filename_ir = str(Path(p_ir).name).split(".")
					filename_ir = "".join((filename_ir[0], "_ir.", filename_ir[1]))
					save_path_ir = str(Path(out) / Path(filename_ir))
				else:
					p = ''
					save_path = str(Path(out) / Path(p).name)
		
			s += '%gx%g ' % img.shape[2:]  # print string
			gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  #  normalization gain whwh
			if det is not None and len(det):
				# Rescale boxes from imgsz to im0 size
				det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

				# Print results
				for c in det[:, -1].unique():
					n = (det[:, -1] == c).sum()  # detections per class
					s += '%g %ss, ' % (n, names[int(c)])  # add to string

				# Write results
				for *xyxy, conf, cls in det:
					if save_txt:  # Write to file
						xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
						if nchannels == 4:
							with open(save_path_rgb[:save_path_rgb.rfind('.')] + '.txt', 'a') as file:
								file.write(('%g ' * 5 + '\n') % (cls, *xywh))  # label format
						else:
							with open(save_path[:save_path.rfind('.')] + '.txt', 'a') as file:
								file.write(('%g ' * 5 + '\n') % (cls, *xywh))  # label format

					if save_img or view_img:  # Add bbox to image
						label = '%s %.2f' % (names[int(cls)], conf)
						# if nchannels == 4:
							# r, g, b, img_ir = cv2.split(im0) # split 4 channels, already in RGB+IR format from datasets_multi.py
							# img_rgb = cv2.merge((r,g,b))
							# plot_one_box(xyxy, img_ir, label=label, color=[255, 255, 255])   #only plot 1 box per image 
							# plot_one_box(xyxy, img_rgb, label=label, color=colors[int(cls)])                       
						# else:
						plot_one_box(xyxy, im0, label=label, color=colors[int(cls)]) #without label, just comment the label and leave the label blank

			# Print time (inference + NMS)
			print('%sDone. (%.3fs)' % (s, t2 - t1))
			comp_time.append(t2-t1)

			# Stream results
			if view_img:
				cv2.imshow(p, im0)
				if cv2.waitKey(1) == ord('q'):  # q to quit
					raise StopIteration

			# Save results (image with detections)
			if save_img:
				if dataset.mode == 'images':
					if nchannels == 4:
						r, g, b, img_ir = cv2.split(im0) # split 4 channels, already in RGB+IR format from datasets_multi.py
						img_rgb = cv2.merge((r, g, b)) # for making rgb images
						
						#save image for each
						cv2.imwrite(save_path_rgb, img_rgb) 
						cv2.imwrite(save_path_ir, img_ir) 
					else:
						cv2.imwrite(save_path, im0)
				else:
					if vid_path != save_path:  # new video
						vid_path = save_path
						if isinstance(vid_writer, cv2.VideoWriter):
							vid_writer.release()  # release previous video writer

						fps = vid_cap.get(cv2.CAP_PROP_FPS)
						w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
						h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
						vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*opt.fourcc), fps, (w, h))
					vid_writer.write(im0)

	if save_txt or save_img:
		print('Results saved to %s' % os.getcwd() + os.sep + out)
		if platform == 'darwin':  # MacOS
			os.system('open ' + save_path)


	#delete model for clearing memory
	print("Deleting model and clear the memory")
	del model
	#empty memory
	torch.cuda.empty_cache() #clear memory
	print("memory is cleared")

	print('Done. (%.3fs)' % (time.time() - t0))
	print(f'Average computation time {round(mean(comp_time), 5)} s')


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--cfg', type=str, default='cfg/yolov3-spp-1cls-4channel.cfg', help='*.cfg path')
	parser.add_argument('--names', type=str, default='data/kaist/kaist_person.names', help='*.names path')
	parser.add_argument('--weights', type=str, default='../result_yolov3/night_multi/weights/best.pt', help='weights path')
	parser.add_argument('--source', type=str, default='data/samples/night', help='source')  # input file/folder, 0 for webcam
	parser.add_argument('--output', type=str, default='output', help='output folder')  # output folder
	parser.add_argument('--img-size', type=int, default=512, help='inference size (pixels)')
	parser.add_argument('--conf-thres', type=float, default=0.3, help='object confidence threshold')
	parser.add_argument('--iou-thres', type=float, default=0.6, help='IOU threshold for NMS')
	parser.add_argument('--fourcc', type=str, default='mp4v', help='output video codec (verify ffmpeg support)')
	parser.add_argument('--half', action='store_true', help='half precision FP16 inference')
	parser.add_argument('--device', default='', help='device id (i.e. 0 or 0,1) or cpu')
	parser.add_argument('--view-img', action='store_true', help='display results')
	parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
	parser.add_argument('--classes', nargs='+', type=int, help='filter by class')
	parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
	parser.add_argument('--augment', action='store_true', help='augmented inference')
	opt = parser.parse_args()
	opt.cfg = check_file(opt.cfg)  # check file
	opt.names = check_file(opt.names)  # check file
	print(opt)

	with torch.no_grad():
		detect(opt.cfg)
