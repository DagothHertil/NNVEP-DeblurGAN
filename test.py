import time
import os
from options.test_options import TestOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
from util.visualizer import Visualizer
from pdb import set_trace as st
from util import html
from util.metrics import PSNR
from util.metrics import SSIM
from PIL import Image

opt = TestOptions().parse()
opt.nThreads = 1   # test code only supports nThreads = 1
opt.batchSize = 1  # test code only supports batchSize = 1
opt.serial_batches = True  # no shuffle
opt.no_flip = True  # no flip

#print('-0000000000000000000000000000000000000000000000000000')
data_loader = CreateDataLoader(opt)
#print('-1111111111111111111111111111111111111111111111111111')
dataset = data_loader.load_data()
#print('-2222222222222222222222222222222222222222222222222222')
model = create_model(opt)
#print('-3333333333333333333333333333333333333333333333333333')
visualizer = Visualizer(opt)
#print('-4444444444444444444444444444444444444444444444444444')
# create website
web_dir = opt.results_dir #os.path.join(opt.results_dir, opt.name, '%s_%s' % (opt.phase, opt.which_epoch))
webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.which_epoch))
# test
avgPSNR = 0.0
avgSSIM = 0.0
counter = 0

if __name__ == '__main__':
	for i, data in enumerate(dataset):
		if i >= opt.how_many:
			break
		counter = i
		#print('1111111111111111111111111111111111111111111111111111')
		model.set_input(data)
		#print('2222222222222222222222222222222222222222222222222222')
		model.test()
		#print('3333333333333333333333333333333333333333333333333333')
		visuals = model.get_current_visuals()
		#print('4444444444444444444444444444444444444444444444444444')
		#avgPSNR += PSNR(visuals['fake_B'],visuals['real_B'])
		#pilFake = Image.fromarray(visuals['fake_B'])
		#pilReal = Image.fromarray(visuals['real_B'])
		#avgSSIM += SSIM(pilFake).cw_ssim_value(pilReal)
		img_path = model.get_image_paths()
		#print('5555555555555555555555555555555555555555555555555555')
		print('process image... %s' % img_path)
		#print('6666666666666666666666666666666666666666666666666666')
		visualizer.save_images(webpage, visuals, img_path)
		#print('7777777777777777777777777777777777777777777777777777')
		
	#avgPSNR /= counter
	#avgSSIM /= counter
	#print('PSNR = %f, SSIM = %f' %
	#				  (avgPSNR, avgSSIM))

	#webpage.save()
