import torch
from model.model import Feature_extractor, Classifier
import numpy as np
import argparse

from data_loader.load_images import ImageList
import data_loader.transforms as transforms

import matplotlib.pyplot as plt
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
    base_network = Feature_extractor(args.resnet).cuda()        # backbone (F)
    classifier_o = Classifier(2048, nClasses).cuda()            # object classifier (C_o^f)

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

    elif args.dataset == "GeoPlaces":
        sp_correct_num_list = torch.zeros(8)
        sp_num_list = torch.zeros(8)
        sp_acc_list = torch.zeros(8)
        f = open("./data_files/GeoPlaces/SuperClass.txt")
        for line in f.readlines():
            super_class.append(int(line))
        f.close()

    elif args.dataset == "DomainNet":
        sp_correct_num_list = torch.zeros(24)
        sp_num_list = torch.zeros(24)
        sp_acc_list = torch.zeros(24)
        f = open("./data_files/DomainNet/SuperClass.txt")
        for line in f.readlines():
            super_class.append(int(line))
        f.close()

    ## test
    with torch.no_grad():
        for i in range(len(dataset_loaders['test'])):
            print("{0}/{1}".format(i,len(dataset_loaders['test'])) , end="\r")
            # data = iter_test.next()
            data = next(iter_test)
            inputs = data[0]
            labels = data[1]
            inputs = inputs.cuda()
            labels = labels.cuda()

            # calculate Top-1 accuracy
            outputs = classifier_o(base_network(inputs))
            predictions = outputs.argmax(1)
            correct = torch.sum((predictions == labels).float())
            accuracy.update(correct, len(outputs))

            # calculate Top-5 accuracy
            _, top5_pred = outputs.topk(5, 1, True, True)
            top5_pred = top5_pred.t()
            top5_correct = torch.sum((top5_pred == labels.view(1, -1).expand_as(top5_pred)).float())
            top5_accuracy.update(top5_correct, len(outputs))

            # calculate accuracy for each superclass
            for i in range(len(outputs)):
                if labels[i] == predictions[i]:
                    sp_correct_num_list[super_class[labels[i]]] += 1
                sp_num_list[super_class[labels[i]]] += 1

            # update confusion matrix
            for i in range(len(outputs)):
                if labels[i] == predictions[i]:
                    conf_matrix[labels[i], 0] += 1
                else:
                    conf_matrix[predictions[i], 1] += 1
                    conf_matrix[labels[i], 2] += 1


    ## print Top-1 accuracy
    print_str1 = "\nCorrect Predictions(Top-1): {}/{}".format(int(accuracy.sum), accuracy.count)
    print_str2 = '\ntest_acc(Top-1):{:.4f}'.format(accuracy.avg)
    print(print_str1 + print_str2)

    ## print Top-5 accuracy
    print_str3 = "\nCorrect Predictions(Top-5): {}/{}".format(int(top5_accuracy.sum), top5_accuracy.count)
    print_str4 = '\ntest_acc(Top-5):{:.4f}'.format(top5_accuracy.avg)
    print(print_str3 + print_str4)

    ## print macro-F1
    macro_f1 = 0
    f1_score = torch.zeros(nClasses)
    for i in range(nClasses):
        f1_score[i] = (2 * conf_matrix[i, 0]) / (2 * conf_matrix[i, 0] + conf_matrix[i, 1] + conf_matrix[i, 2])
    f1_score = torch.nan_to_num(f1_score, nan=0)
    macro_f1 = torch.sum(f1_score) / nClasses
    print("\nmacro-F1:{:.4f}".format(macro_f1))

    ## print acc for each superclass
    print("\ntest acc for each superclasses")
    for i in range(len(sp_acc_list)):
        if sp_num_list[i] == 0:
            sp_acc_list[i] = 0
        else:
            sp_acc_list[i] = sp_correct_num_list[i] / sp_num_list[i]
        print("sp{}: {:.4f}".format(i, sp_acc_list[i]))