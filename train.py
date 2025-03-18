## import (MemSAC)
import torch
import torch.optim as optim
import torch.nn as nn
from model.model import Feature_extractor, Projector, Classifier, AdversarialLayer, discriminator
import numpy as np
import argparse
import os

from model.memory import MemoryModule, MemoryModule_rsf
from data_loader.load_images import ImageList
import data_loader.transforms as transforms

from torch.utils.tensorboard import SummaryWriter

import torch.backends.cudnn as cudnn
cudnn.enabled = False
torch.backends.cudnn.deterministic = True

## import (Visualization)
import random
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns

seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
os.environ['PYTHONHASHSEED'] = str(seed)

def test_target(loader, model, classifier):
    start_test = True
    with torch.no_grad():
        iter_test = iter(loader["test"])
        for data in iter_test:
            inputs = data[0]
            labels = data[1]
            inputs = inputs.cuda()
            labels = labels.cuda()
            outputs = classifier(model(inputs))
            if start_test:
                all_output = outputs.float()
                all_label = labels.float()
                start_test = False
            else:
                all_output = torch.cat((all_output, outputs.float()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)
    _, predict = torch.max(all_output, 1)
    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).detach() / float(all_label.size()[0])
    return accuracy

def inv_lr_scheduler(param_lr, optimizer, iter_num, gamma, power, init_lr=0.001):
    """Decay learning rate by a factor of 0.1 every lr_decay_epoch epochs."""
    lr = init_lr * (1 + gamma * iter_num) ** (-power)
    i = 0
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr * param_lr[i]
        i += 1
    return optimizer


def train_base(data_source, data_target, correct_count, total_count, fs, ft, fts, sls, slt):
    base_network.train()        # Backbone (F)
    classifier_o.train()        # Projector (P)
    projector.train()           # Object Classifier (C_o^f)
    embed_classifier_o.train()  # Object Classifier (C_o^z)
    embed_classifier_d.train()  # Domain Classifier (C_d^z)

    inputs_source, labels_source = data_source  # inputs_source:x_i^s, labels_source:y_i
    inputs_target, labels_target = data_target  # inputs_target:x_i^t, labels_target:y_i

    inputs = torch.cat((inputs_source, inputs_target), dim=0)
    inputs = inputs.cuda()

    labels_source = labels_source.cuda()
    labels_target = labels_target.cuda()
    assert len(inputs_source) == len(inputs_target)

    extracted_features = base_network(inputs)       # f
    features = projector(extracted_features)        # z
    logits = classifier_o(extracted_features)       # C_o^f(f)
    embed_logits = embed_classifier_o(features)     # C_o^z(z)
    embed_logits_d = embed_classifier_d(features)   # C_d^z(z)

    features_source = features[:len(inputs_source)]             # f^s
    features_target = features[len(inputs_source):]             # f^t
    logits_source = logits[:len(inputs_source)]                 # C_o^f(f^s)
    logits_target = logits[len(inputs_source):]                 # C_o^f(f^t)
    embed_logits_source = embed_logits[:len(inputs_source)]     # C_o^z(z^s)

    _, pred_labels_target = torch.max(logits_target, 1)         # predicted labels by classifier(C_o^f)


    ## assign superclass based on Ground-Truth (Source)
    superlabels_source = []
    for l in labels_source:
        superlabels_source.append(super_class[l])
    superlabels_source = torch.tensor(superlabels_source)

    ## assign superclass based on predicted label (Target)
    superlabels_target = []
    for l in pred_labels_target:
        superlabels_target.append(super_class[l])
    superlabels_target = torch.tensor(superlabels_target)


    ## estimate region-specific features for each superclass
    mean_features_source = memory_network_src(features_source, superlabels_source, num_superclass)
    mean_features_target = memory_network_tgt(features_target, superlabels_target, num_superclass)

    ## associate region-specific features with each samples in the mini-batch
    mfs = torch.zeros(32, 256).cuda()
    for i in range(len(superlabels_source)):
        mfs[i] = mean_features_source[superlabels_source[i]]
    mft = torch.zeros(32, 256).cuda()
    for i in range(len(superlabels_target)):
        mft[i] = mean_features_target[superlabels_target[i]]

    ## domain labels with superclasses
    domain_labels_target = []
    for i in superlabels_target:
        domain_labels_target.append(i + num_superclass)
    domain_labels_target = torch.tensor(domain_labels_target)
    super_labels = torch.cat((superlabels_source, domain_labels_target), dim=0).cuda()

    ## domain labels without superclasses (for CDAN Loss)
    domain_labels = torch.tensor([[1], ] * len(inputs_source) + [[0], ] * len(inputs_target),
                                 device=torch.device('cuda:0'), dtype=torch.float)

    ## matching distributions
    features_target_s = features_target + ((args.max_iteration - iter_num) / args.max_iteration) * (mfs - mft)
    transferred_features = torch.cat((features_source, features_target_s), dim=0)


    # t-SNE visualization
    if args.mode == "tsne":
        if iter_num % 2000 == 1:
            fs = features_source
            ft = features_target
            fts = features_target_s
            sls = superlabels_source
            slt = superlabels_target
        else:
            fs = torch.cat((fs, features_source), dim=0)
            ft = torch.cat((ft, features_target), dim=0)
            fts = torch.cat((fts, features_target_s), dim=0)
            sls = torch.cat((sls, superlabels_source), dim=0)
            slt = torch.cat((slt, superlabels_target), dim=0)

        if iter_num % 2000 == 0:
            fs = fs.detach().cpu()
            ft = ft.detach().cpu()
            fts = fts.detach().cpu()
            sls = sls.detach().cpu()
            slt = slt.detach().cpu()

            idx_list_s = []
            idx_list_t = []

            for i in range(num_superclass):
                idx = np.where(sls == i, True, False)
                idx_list_s.append(idx)
            for i in range(num_superclass):
                idx = np.where(slt == i, True, False)
                idx_list_t.append(idx)

            source_tsne = tsne.fit_transform(fs)
            target_tsne = tsne.fit_transform(ft)
            target_tsne_s = tsne.fit_transform(fts)

            ## sampling
            for i in range(num_superclass):
                num_samples_s = len(source_tsne[idx_list_s[i],0])
                num_samples_t = len(target_tsne[idx_list_t[i],0])
                if num_samples_s > 1000:
                    rand_idx = random.sample(range(len(source_tsne[idx_list_s[i],0])), 1000)
                    source_tsne_x = source_tsne[idx_list_s[i],0][rand_idx]
                    source_tsne_y = source_tsne[idx_list_s[i],1][rand_idx]
                else:
                    source_tsne_x = source_tsne[idx_list_s[i],0]
                    source_tsne_y = source_tsne[idx_list_s[i],1]

                if num_samples_t > 1000:
                    rand_idx = random.sample(range(len(target_tsne[idx_list_t[i], 0])), 1000)
                    target_tsne_x = target_tsne[idx_list_t[i], 0][rand_idx]
                    target_tsne_y = target_tsne[idx_list_t[i], 1][rand_idx]
                    target_s_tsne_x = target_tsne_s[idx_list_t[i], 0][rand_idx]
                    target_s_tsne_y = target_tsne_s[idx_list_t[i], 1][rand_idx]
                else:
                    target_tsne_x = target_tsne[idx_list_t[i],0]
                    target_tsne_y = target_tsne[idx_list_t[i],1]
                    target_s_tsne_x = target_tsne_s[idx_list_t[i], 0]
                    target_s_tsne_y = target_tsne_s[idx_list_t[i], 1]


                plt.rcParams["font.size"] = 25
                plt.figure(figsize=(10, 10))
                sns.scatterplot(x=source_tsne_x, y=source_tsne_y)
                sns.scatterplot(x=target_tsne_x, y=target_tsne_y)
                sns.scatterplot(x=target_s_tsne_x, y=target_s_tsne_y)
                # plt.legend()
                plt.xticks([-60, 0, 60])
                plt.yticks([-60, 0, 60])
                save_dir = './pictures/tsne/' + str(args.dataset) + '/' + str(args.source) + str(args.target)
                os.makedirs(save_dir, exist_ok=True)
                plt.savefig(save_dir + '/iter{:d}_sp{:d}.jpg'.format(iter_num, i), format="jpg", dpi=100)
                plt.close()



    ## Classifier Loss
    object_classifier_loss = criterion["classifier"](logits_source, labels_source)
    embed_object_classifier_loss = criterion["classifier"](embed_logits_source, labels_source)
    embed_domain_classifier_loss = criterion["classifier"](embed_logits_d, super_labels)

    obj_loss = args.lambda_cf * object_classifier_loss + args.lambda_cr * embed_object_classifier_loss
    dom_loss = args.lambda_dr * embed_domain_classifier_loss

    total_classifier_loss = obj_loss + dom_loss

    ## CDAN Loss
    domain_predicted = discriminator(grl.apply(features), torch.softmax(logits, dim=1).detach())
    transfer_loss = criterion["adversarial"](domain_predicted, domain_labels)
    transfer_loss = args.adv_coeff * transfer_loss

    ## MemSAC Loss
    sim_loss, num_correct = memory_network(transferred_features, features_target, labels_source, labels_target)
    sim_loss = args.sim_coeff * sim_loss * (iter_num > args.only_da_iter)

    ## Total Loss
    total_loss = total_classifier_loss + transfer_loss + sim_loss

    total_loss.backward()
    optimizer.step()
    writer.add_scalar("Loss/classifier_loss", total_classifier_loss.detach(), iter_num)
    writer.add_scalar("Loss/transfer", transfer_loss.detach(), iter_num)
    writer.add_scalar("Loss/sim_loss", sim_loss.detach(), iter_num)

    correct_count += num_correct
    total_count += len(inputs_source)

    return correct_count, total_count, superlabels_source, superlabels_target, fs, ft, fts, sls, slt


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Transfer Learning')

    ## Training parameters
    parser.add_argument('--gpu_id', type=str, nargs='?', default='0', help="device id to run")
    parser.add_argument('--dataset', required=True, help="Name of the dataset")
    parser.add_argument('--source', type=str, nargs='?', default='c', help="source dataset")
    parser.add_argument('--target', type=str, nargs='?', default='p', help="target dataset")
    parser.add_argument('--lr', type=float, nargs='?', default=0.03, help="target dataset")
    parser.add_argument('--max_iteration', type=int, nargs='?', default=102500, help="target dataset")
    parser.add_argument('--out_dir', type=str, nargs='?', default='e', help="output dir")
    parser.add_argument('--batch_size', type=int, default=32, help="batch size should be samples * classes")
    parser.add_argument('--data_dir', type=str, default="./data", help="Path for data directory")
    parser.add_argument('--multi_gpu', type=int, default=0)
    parser.add_argument('--total_classes', type=int, default=31, help="total # classes in the dataset")

    ## Testing parameters
    parser.add_argument('--test_10crop', action="store_true", help="10 crop testing")
    parser.add_argument('--test-iter', type=int, default=10000, help="Testing freq.")

    ## Architecture
    parser.add_argument('--resnet', default="resnet50", help="Resnet backbone")
    parser.add_argument('--bn-dim', type=int, default=256, help="bottleneck embedding dimension")

    ## Adaptation parameters
    parser.add_argument('--only_da_iter', type=int, default=100,
                        help="number of iterations when only DA loss works and MSC doesn't")
    parser.add_argument('--simi_func', type=str, default='cosine', choices=['cosine', 'euclidean', "gaussian"])
    parser.add_argument('--method', type=str, default="MemSAC")
    parser.add_argument('--knn_method', type=str, nargs='?', default='ranking', choices=['ranking', 'classic'])
    parser.add_argument('--ranking_k', type=int, default=4, help="use number of samples per class")
    parser.add_argument('--top_ranked_n', type=int, default=20,
                        help="these many target samples are used finally, 1/3 of batch")
    parser.add_argument('--k', type=int, default=5, help="k for knn")

    ## Memory network
    parser.add_argument('--queue_size', type=int, default=24000, help="size of queue")
    parser.add_argument('--momentum', type=float, default=0, help="momentum value")
    parser.add_argument('--tau', type=float, default=0.07, help="temperature value")

    ## Loss coeffecients
    parser.add_argument('--sim-coeff', type=float, default=0.1, help="coeff for similarity loss")
    parser.add_argument('--adv-coeff', type=float, default=1., help="Adversarial Loss")
    parser.add_argument('--lambda_cf', type=float, default=1., help="Classifier Loss")
    parser.add_argument('--lambda_cr', type=float, default=1., help="Embed Classifier Loss")
    parser.add_argument('--lambda_dr', type=float, default=1., help="Domain and SuperClass Classifier Loss")

    ## visualization
    parser.add_argument('--mode', type=str ,default="train",choices=['train', 'tsne'] , help="to visualize or not")

    args = parser.parse_args()
    out_dir = os.path.join("work_dirs", args.dataset, args.out_dir)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, "log.txt")
    log_acc = os.path.join(out_dir, "logAcc.txt")
    print("Writing log to", out_file)
    out_file = open(out_file, "w")
    best_file = os.path.join(out_dir, "best.txt")
    args.multi_gpu = bool(args.multi_gpu)
    print(args)

    ##### TensorBoard & Misc Setup #####
    writer_loc = os.path.join(out_dir, 'tensorboard_logs')
    writer = SummaryWriter(writer_loc)

    if args.dataset == "GeoImNet":
        file_path = {
            "asia": "./data_files/GeoImNet/asia_train.txt",
            "usa": "./data_files/GeoImNet/usa_train.txt",
        }
        dataset_source = file_path[args.source]
        dataset_target = file_path[args.target]
        dataset_test = file_path[args.target].replace("train", "test")

    elif args.dataset == "GeoPlaces":
        file_path = {
            "asia": "./data_files/GeoPlaces/asia_train.txt",
            "usa": "./data_files/GeoPlaces/usa_train.txt",
        }
        dataset_source = file_path[args.source]
        dataset_target = file_path[args.target]
        dataset_test = file_path[args.target].replace("train", "test")

    elif args.dataset == "DomainNet":
        file_path = {
            "real": "./data_files/DomainNet/real_train.txt" ,
            "sketch": "./data_files/DomainNet/sketch_train.txt" ,
            "painting": "./data_files/DomainNet/painting_train.txt" ,
            "clipart": "./data_files/DomainNet/clipart_train.txt" ,
            "quickdraw": "./data_files/DomainNet/quickdraw_train.txt"}
        dataset_source = file_path[args.source]
        dataset_target = file_path[args.target]
        dataset_test = file_path[args.target].replace("train" , "test")


    else:
        raise NotImplementedError

    super_class = []

    if args.dataset == "GeoImNet":
        f = open("./data_files/GeoImNet/SuperClass.txt")
        num_superclass = 20
    elif args.dataset == "GeoPlaces":
        f = open("./data_files/GeoPlaces/SuperClass.txt")
        num_superclass = 8
    elif args.dataset == "DomainNet":
        f = open("./data_files/DomainNet/SuperClass.txt")
        num_superclass = 24

    for line in f.readlines():
        super_class.append(int(line))
    f.close()

    print("Source:", args.source)
    print("Target:", args.target)

    batch_size = {"train": args.batch_size, "val": args.batch_size * 4}

    out_file.write('args = {}\n'.format(args))
    out_file.flush()

    dataset_loaders = {}
    print(dataset_source)

    dataset_list = ImageList(args.data_dir, open(dataset_source).readlines(),
                             transform=transforms.image_train(resize_size=256, crop_size=224))

    print(f"{len(dataset_list)} source samples")

    dataset_loaders["source"] = torch.utils.data.DataLoader(dataset_list, batch_size=batch_size['train'],
                                                            shuffle=True, num_workers=8,
                                                            drop_last=True)

    dataset_list = ImageList(args.data_dir, open(dataset_target).readlines(),
                             transform=transforms.image_train(resize_size=256, crop_size=224))
    dataset_loaders["target"] = torch.utils.data.DataLoader(dataset_list, batch_size=batch_size['train'], shuffle=True,
                                                            num_workers=8, drop_last=True)

    print(f"{len(dataset_list)} target samples")

    dataset_list = ImageList(args.data_dir, open(dataset_test).readlines(),
                             transform=transforms.image_test(resize_size=256, crop_size=224))
    dataset_loaders["test"] = torch.utils.data.DataLoader(dataset_list, batch_size=batch_size['val'], shuffle=False,
                                                          num_workers=8)
    print(f"{len(dataset_list)} target test samples")

    # network construction
    base_network = Feature_extractor(args.resnet)                       # backbone (F)
    base_network = base_network.cuda()

    classifier_o = Classifier(2048, args.total_classes)                 # object classifier (C_o^f)
    classifier_o.cuda()

    projector = Projector(args.bn_dim)                                  # projector (P)
    projector = projector.cuda()

    embed_classifier_o = Classifier(args.bn_dim, args.total_classes)    # object classifier (C_o^z)
    embed_classifier_o = embed_classifier_o.cuda()
    embed_classifier_d = Classifier(args.bn_dim, 2*num_superclass)      # domain classifier (C_d^z)
    embed_classifier_d = embed_classifier_d.cuda()

    discriminator = discriminator(args.bn_dim, args.total_classes)      # discriminator (G)
    discriminator = discriminator.cuda()
    discriminator.train(True)

    ## memory bank (M)
    memory_network = MemoryModule(args.bn_dim, K=args.queue_size, m=args.momentum, T=args.tau, knn=args.k,
                                  top_ranked_n=args.top_ranked_n, similarity_func=args.simi_func,
                                  batch_size=batch_size["train"], ranking_k=args.ranking_k)
    memory_network = memory_network.cuda()

    ## memory bank (M_s)
    memory_network_src = MemoryModule_rsf(args.bn_dim, K=36000, m=args.momentum, T=args.tau, knn=args.k,
                                          top_ranked_n=args.top_ranked_n, similarity_func=args.simi_func,
                                          batch_size=batch_size["train"], ranking_k=args.ranking_k)
    memory_network_src = memory_network_src.cuda()

    ## memory bank (M_t)
    memory_network_tgt = MemoryModule_rsf(args.bn_dim, K=36000, m=args.momentum, T=args.tau, knn=args.k,
                                          top_ranked_n=args.top_ranked_n, similarity_func=args.simi_func,
                                          batch_size=batch_size["train"], ranking_k=args.ranking_k)
    memory_network_tgt = memory_network_tgt.cuda()

    ## gradient reversal layer (GRL)
    grl = AdversarialLayer()

    ## Loss function
    criterion = {
        "classifier": nn.CrossEntropyLoss(),
        "adversarial": nn.BCEWithLogitsLoss(),
    }


    optimizer_dict = [
        {"params": filter(lambda p: p.requires_grad, base_network.model_fc.parameters()), "lr": 0.1},

        {"params": filter(lambda p: p.requires_grad, classifier_o.classifier.parameters()), "lr": 1},
        # {"params": filter(lambda p: p.requires_grad, encoder_o.projector_0.parameters()), "lr": 1},
        {"params": filter(lambda p: p.requires_grad, projector.projector_0.parameters()), "lr": 1},

        {"params": filter(lambda p: p.requires_grad, embed_classifier_o.classifier.parameters()), "lr": 1},
        # {"params": filter(lambda p: p.requires_grad, encoder_d.projector_0.parameters()), "lr": 1},
        {"params": filter(lambda p: p.requires_grad, embed_classifier_d.classifier.parameters()), "lr": 1},

        {"params": filter(lambda p: p.requires_grad, discriminator.parameters()), "lr": 1}  # ,
    ]

    optimizer = optim.SGD(optimizer_dict, momentum=0.9, weight_decay=0.0005)

    if args.multi_gpu:
        base_network = nn.DataParallel(base_network).cuda()

    param_lr = []
    for param_group in optimizer.param_groups:
        param_lr.append(param_group["lr"])

    len_source = len(dataset_loaders["source"]) - 1
    len_target = len(dataset_loaders["target"]) - 1
    iter_source = iter(dataset_loaders["source"])
    iter_target = iter(dataset_loaders["target"])

    with open(os.path.join(out_dir, "best.txt"), "a") as fh:
        fh.write("Best Accuracy file\n")

    start_iter = 1
    best_acc = 0

    if os.path.exists(os.path.join(out_dir, "checkpoint.pth.tar")):
        print("Loading from pretrained model ...")
        checkpoint = torch.load(os.path.join(out_dir, "checkpoint.pth.tar"))
        base_network.load_state_dict(checkpoint["state_dict"])
        classifier_o.load_state_dict(checkpoint["classifier_o_state_dict"])
        embed_classifier_o.load_state_dict(checkpoint["embed_classifier_o_state_dict"])
        embed_classifier_d.load_state_dict(checkpoint["embed_classifier_d_state_dict"])
        memory_network.load_state_dict(checkpoint["memory_state_dict"])
        memory_network_src.load_state_dict(checkpoint["memory_s_state_dict"])
        memory_network_tgt.load_state_dict(checkpoint["memory_o_state_dict"])
        start_iter = checkpoint["iter"]

    correct_count = 0
    total_count = 0
    mean_features_source = torch.zeros(args.bn_dim).repeat(num_superclass, 1).cuda()
    mean_features_target = torch.zeros(args.bn_dim).repeat(num_superclass, 1).cuda()


    ## visualization
    fs = None
    ft = None
    fts = None
    sls = None
    slt = None
    tsne = TSNE(n_components=2)


    for iter_num in range(start_iter, args.max_iteration + 1):

        optimizer = inv_lr_scheduler(param_lr, optimizer, iter_num, init_lr=args.lr, gamma=0.001, power=0.75)
        optimizer.zero_grad()
        print("Iter:", iter_num, end="\r")

        if iter_num % len_source == 0:
            iter_source = iter(dataset_loaders["source"])
        if iter_num % len_target == 0:
            iter_target = iter(dataset_loaders["target"])
        # data_source = iter_source.next()
        # data_target = iter_target.next()
        data_source = next(iter_source)
        data_target = next(iter_target)

        correct_count, total_count, superlabels_source, superlabels_target, fs, ft, fts, sls, slt = \
            train_base(data_source, data_target, correct_count, total_count, fs, ft, fts, sls, slt)

        # test
        test_interval = args.test_iter
        if iter_num % test_interval == 0:
            print()
            base_network.eval()
            classifier_o.eval()
            test_acc = test_target(dataset_loaders, base_network, classifier_o)
            writer.add_scalar("Acc/test", test_acc, iter_num)
            print_str1 = '\niter: {:05d}, test_acc:{:.4f}'.format(iter_num, test_acc)
            print(print_str1)
            label_acc = correct_count * 100 / total_count
            print('label accuracy: {:.2f}\n'.format(label_acc))


            correct_count = 0
            total_count = 0

            if test_acc > best_acc:
                best_acc = test_acc
                best_model = base_network.state_dict()
                best_obj_classifier = classifier_o.state_dict()
                with open(os.path.join(out_dir, "best.txt"), "a") as fh:
                    fh.write(
                        "Best Accuracy : {:.4f} Label Accuracy : {:.2f} at iter: {:05d}\n".format(best_acc, label_acc,
                                                                                                  iter_num))
                torch.save(best_model, os.path.join(out_dir, "best_model.pth.tar"))
                torch.save(best_obj_classifier, os.path.join(out_dir, "best_classifier_o.pth.tar"))

            checkpoint_dict = {
                "state_dict": base_network.state_dict(),
                "classifier_o_state_dict": classifier_o.state_dict(),
                "embed_classifier_o_state_dict": embed_classifier_o.state_dict(),
                "embed_classifier_d_state_dict": embed_classifier_d.state_dict(),
                "discriminator_state_dict": discriminator.state_dict(),
                "memory_state_dict": memory_network.state_dict(),
                "memory_s_state_dict": memory_network_src.state_dict(),
                "memory_o_state_dict": memory_network_tgt.state_dict(),
                "optimizer": optimizer.state_dict(),
                "iter": iter_num + 1,
                "args": args
            }
            torch.save(checkpoint_dict, os.path.join(out_dir, "checkpoint.pth.tar"))
