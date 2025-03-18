import torch
from model.model import Feature_extractor, Classifier
import numpy as np
import argparse

from data_loader.load_images import ImageList
import data_loader.transforms as transforms

import matplotlib.pyplot as plt
import cv2
from pytorch_grad_cam.grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image

import os

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.sum += val 
        self.count += n
        self.avg = self.sum / self.count

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='MemSAC')

    parser.add_argument('--gpu_id', type=str, nargs='?', default='0', help="device id to run")
    parser.add_argument('--dataset', type=str, nargs='?', default='c', help="target dataset")
    parser.add_argument('--target', type=str, nargs='?', default='c', help="target domain")
    parser.add_argument('--batch_size', type=int, default=64, help="batch size should be samples * classes")
    parser.add_argument('--checkpoint' , type=str, help="Checkpoint to load from.")
    parser.add_argument('--multi_gpu', type=int, default=0, help="use dataparallel if 1")
    parser.add_argument('--resnet', default="resnet50", help="Resnet backbone")

    args = parser.parse_args() 


    if args.dataset == "GeoImNet":
        data_dir ="./data/GeoImNet"
        file_path = {
            "asia": "./data_files/GeoImNet/asia_test.txt",
            "usa": "./data_files/GeoImNet/usa_test.txt"
        }
        nClasses = 600
    elif args.dataset == "GeoPlaces":
        data_dir = "./data/GeoPlaces"
        file_path = {
            "asia": "./data_files/GeoPlaces/asia_test.txt",
            "usa": "./data_files/GeoPlaces/usa_test.txt"
        }
        nClasses = 204
    elif args.dataset == "DomainNet":
        data_dir = "./data/DomainNet"
        file_path = {
            "real": "./data_files/DomainNet/real_test.txt",
            "sketch": "./data_files/DomainNet/sketch_test.txt",
            "painting": "./data_files/DomainNet/painting_test.txt",
            "clipart": "./data_files/DomainNet/clipart_test.txt"
        }
        nClasses = 345
    else:
        raise NotImplementedError

    dataset_test = file_path[args.target]
    print("Target" , args.target)

    dataset_loaders = {}

    dataset_list = ImageList(data_dir, open(dataset_test).readlines(), transform=transforms.image_test(resize_size=256, crop_size=224))
    print("Size of target dataset:" , len(dataset_list))
    dataset_loaders["test"] = torch.utils.data.DataLoader(dataset_list, batch_size=args.batch_size, shuffle=False,
                                                          num_workers=16, drop_last=False)

    ## network construction
    base_network = Feature_extractor(args.resnet).cuda()
    classifier_o = Classifier(2048, nClasses).cuda()

        
    accuracy = AverageMeter()
    top5_accuracy = AverageMeter()

    saved_state_dict_f = torch.load("./trained_model/best_model.pth.tar")
    base_network.load_state_dict(saved_state_dict_f, strict=True)
    saved_state_dict_o = torch.load("./trained_model/best_classifier_o.pth.tar")
    classifier_o.load_state_dict(saved_state_dict_o, strict=True)

    base_network.eval()
    classifier_o.eval()
    start_test = True
    iter_test = iter(dataset_loaders["test"])

    correct_num_list = torch.zeros(nClasses)
    num_list = torch.zeros(nClasses)
    conf_matrix = torch.zeros(nClasses, 3) # TP, FP, FN

    super_class = []
    if args.dataset == "GeoImNet":
        sp_correct_num_list = torch.zeros(20)
        sp_num_list = torch.zeros(20)
        sp_acc_list = torch.zeros(20)
        f = open("./data_files/GeoImNet/SuperClass.txt")
        for line in f.readlines():
            super_class.append(int(line))
        f.close()
        f = open("./data_files/GeoImNet/ClassList.txt")
        class_list = [line.strip() for line in f.readlines()]
        f.close()

    elif args.dataset == "GeoPlaces":
        sp_correct_num_list = torch.zeros(8)
        sp_num_list = torch.zeros(8)
        sp_acc_list = torch.zeros(8)
        f = open("./data_files/GeoPlaces/SuperClass.txt")
        for line in f.readlines():
            super_class.append(int(line))
        f.close()
        f = open("./data_files/GeoPlaces/ClassList.txt")
        class_list = [line.strip() for line in f.readlines()]
        f.close()

    elif args.dataset == "DomainNet":
        sp_correct_num_list = torch.zeros(24)
        sp_num_list = torch.zeros(24)
        sp_acc_list = torch.zeros(24)
        f = open("./data_files/DomainNet/SuperClass.txt")
        for line in f.readlines():
            super_class.append(int(line))
        f.close()
        f = open("./data_files/DomainNet/ClassList.txt")
        class_list = [line.strip() for line in f.readlines()]
        f.close()

    save_path = ('./pictures/Grad_CAM/' + str(args.dataset) + '/' + str(args.target) + 'test' )
    os.makedirs(save_path, exist_ok=True)

    target_layers = [base_network.model_fc.layer4[-1]]
    cam = GradCAM(model=base_network.model_fc, target_layers=target_layers, use_cuda=torch.cuda.is_available())

    for i in range(len(dataset_loaders['test'])):
        print("{0}/{1}".format(i,len(dataset_loaders['test'])) , end="\r")
        # data = iter_test.next()
        data = next(iter_test)
        inputs = data[0]
        labels = data[1]
        inputs = inputs.cuda()
        labels = labels.cuda()

        with torch.no_grad():
            outputs = classifier_o(base_network(inputs))
            predictions = outputs.argmax(1)

        for j in range(len(labels)):
            super_label = super_class[labels[j]]

            img = cv2.imread(paths[j], 1)[:, :, ::-1]
            img = cv2.resize(img, (224, 224))
            img = np.float32(img) / 255

            input_tensor = preprocess_image(img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            grayscale_cam = cam(input_tensor=input_tensor)
            grayscale_cam = grayscale_cam[0, :]

            visualization = show_cam_on_image(img, grayscale_cam, use_rgb=True)
            plt.gca().axis('off')
            if labels[j] == predictions[j]:
                plot_color = 'red'
            else:
                plot_color = 'blue'

            plt.text(43,240,'label: ',size='xx-large')
            plt.text(86,240,'{}'.format(str(class_list[labels[j].item()])),size='xx-large')
            plt.text(5,260,'prediction: ', size='xx-large')
            plt.text(86,260,'{}'.format(str(class_list[predictions[j].item()])), size='xx-large', color=plot_color)
            plt.imshow(visualization, vmin=0, vmax=255)
            os.makedirs(save_path + "/sp" + str(super_label) + '/' + str(class_list[labels[j]]), exist_ok=True)
            plt.savefig(save_path + '/sp' + str(super_label) + '/' + str(class_list[labels[j]]) + '/' + str(paths[j]).split('/')[-1].replace('.png', '.jpg'), format="jpg", dpi=100, bbox_inches='tight')

            plt.close()