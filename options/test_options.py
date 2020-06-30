import argparse

class TestOptions():
    def initialize(self):
        parser = argparse.ArgumentParser(description="test segmentation network")
        parser.add_argument("--model", type=str, default='DeepLab', help="available options : DeepLab and VGG")
        parser.add_argument("--GPU", type=str, default='0', help="which GPU to use")

        parser.add_argument("--data-dir-target", type=str, default='../data_semseg/cityscapes', help="Path to the directory containing the target dataset.")
        parser.add_argument("--data-list-target", type=str, default='./dataset/cityscapes_list/val.txt', help="list of images in the target dataset.")
        parser.add_argument("--num-classes", type=int, default=19, help="Number of classes for cityscapes.")
        parser.add_argument("--set", type=str, default='val', help="choose test set.")
        parser.add_argument("--restore-opt1", type=str, default=None, help="restore model parameters from beta1")
        parser.add_argument("--restore-opt2", type=str, default=None, help="restore model parameters from beta2")
        parser.add_argument("--restore-opt3", type=str, default=None, help="restore model parameters from beta3")

        parser.add_argument("--init-weights", type=str, default=None, help="initial model.")
        parser.add_argument("--restore-from", type=str, default=None, help="restore model parameters from")

        parser.add_argument("--save", type=str, default='../results', help="Path to save result.")
        parser.add_argument('--gt_dir', type=str, default='../data_semseg/cityscapes/gtFine/val', help='directory for CityScapes val gt images')
        parser.add_argument('--devkit_dir', type=str, default='./dataset/cityscapes_list', help='list directory of cityscapes')         

        return parser.parse_args()

