from __future__ import print_function
import argparse, os
from datetime import datetime
from deblur import Deblur


def parse_args():
	parser = argparse.ArgumentParser(description="Deblur")
	parser.add_argument('--phase', type=str, default='test',help='test or psnr')
	parser.add_argument('--pretrained_dataset', type=str, default=None,help='Dataset on which the checkpoint is trained. NTIRE or GOPRO. NTIRE : 3 consecutive frames input, GOPRO : single frame input.')
	parser.add_argument('--kernel_size', type=int, default=5,help='kernel_size')
	parser.add_argument('--channels', type=int, default=3,help='# img channels')
	parser.add_argument('--ensemble', action = 'store_true', help='use this if self ensemble is needed')
	parser.add_argument('--test_dataset', type=str, default='../Dataset/val/',help='test dataset path')
	parser.add_argument('--working_directory', type=str, default='./data/',help='working_directory path')
	return parser.parse_args()

def main():
	args = parse_args()
	assert(args.kernel_size%2 == 1), "kernel_size should be an odd number"
	assert(args.pretrained_dataset in ['NTIRE', 'GOPRO']), "dataset arg should be NTIRE or GOPRO"
	model = Deblur(args)
	if args.phase == 'psnr':
		print("PSNR phase")
		model.test_psnr(args)
		exit(1)

	if args.phase == 'test':
		model.build_model(args)
		print("Test phase")
		model.test(args, model.list_test)

if __name__ == '__main__':
	main()
