import argparse
import os
import util

class BaseOptions():
    def __init__(self):
        self.initialized = False

    def initialize(self, parser):

        parser.add_argument("--epoch", type=int, default=20, help="number of training epochs")
        parser.add_argument('--GPU', type=str, default='0', help="the index of GPU you will use for computation (e.g. '0', '0,1', '0,1,2')")
        parser.add_argument('--patch_x', type=int, default=128, help="patch size in x and y")
        parser.add_argument('--patch_t', type=int, default=16, help="patch size in z, at least 16")
        parser.add_argument('--image_type', type=str, default='xyz', help="xyt for functional imaging and xyz for strctural imaging")
        parser.add_argument('--inference_mode', type=str, default='2D_center', help="2D_center for denoising inference of functional imaging")
        parser.add_argument('--iter_num', type=int, default=3, help="iterations in the inference stage")
        parser.add_argument('--train_datasets_size', type=int, default=10000, help="total number of training samples (patches)")
        parser.add_argument('--select_img_num', type=int, default=10000000000,help='How many frames will be used for training.')
        parser.add_argument('--overlap_factor', type=float, default=0.5, help="the overlap factor between two adjacent patches")
        parser.add_argument('--batch_size', type=int, default=1, help='input train batch size')
        parser.add_argument('--lr', type=float, default=0.0001, help='initial learning rate')
        parser.add_argument('--normalization', type=str, default='Mean', help='normalization method (Mean or Max-Min)')
        parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')
        parser.add_argument('--num_threads', default=0, type=int, help='# threads for loading data')
        parser.add_argument('--datapath', default=r'./Data', help='path of the raw data')
        parser.add_argument('--task_name', type=str, default='Brain', help='the current task name')
        parser.add_argument('--scale_factor', type=int, default=1, help='the factor for image intensity scaling')
        parser.add_argument('--model_save_fre', type=int, default=1, help='frequency of saving model')
        parser.add_argument('--postprocess', type=bool, default=True, help='post-processing for denoising image')

        self.initialized = True
        return parser

    def gather_options(self):
        if not self.initialized:  # check if it has been initialized
            parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            parser = self.initialize(parser)

        # get the basic options
        opt, _ = parser.parse_known_args()
        # save and return the parser
        self.parser = parser
        return parser.parse_args()

    def print_options(self, opt):
        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in sorted(vars(opt).items()):
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'
        print(message)

        # save to the disk
        expr_dir = os.path.join(opt.checkpoints_dir, opt.task_name,'model_parameter_list')
        util.mkdirs(expr_dir)
        file_name = os.path.join(expr_dir, 'option.txt')
        with open(file_name, 'wt') as opt_file:
            opt_file.write(message)
            opt_file.write('\n')

    def parse(self):
        opt = self.gather_options()
        #opt.isTrain = self.isTrain
        self.print_options(opt)
        self.opt = opt
        return self.opt

