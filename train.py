from __future__ import division

from test import evaluate
from models import Darknet
from utils.utils import load_classes, weights_init_normal
from utils.datasets import ListDataset, resize
from utils.parse_config import parse_data_config

import os
import argparse
import random
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.autograd import Variable
import torch.optim as optim

TQDM_USE=True

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100, help="number of epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="size of each image batch")
    parser.add_argument("--gradient_accumulations", type=int, default=2, help="number of gradient accums before step")
    parser.add_argument("--model_def", type=str, default="config/yolov3.cfg", help="path to model definition file")
    parser.add_argument("--data_config", type=str, default="config/coco.data", help="path to data config file")
    parser.add_argument("--pretrained_weights", type=str, help="if specified starts from checkpoint model")
    parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
    parser.add_argument("--img_size", type=int, default=416, help="size of each image dimension")
    parser.add_argument("--print_interval", type=int, default=100, help="output the log in a batch")
    parser.add_argument("--checkpoint_interval", type=int, default=1, help="interval between saving model weights")
    parser.add_argument("--evaluation_interval", type=int, default=1, help="interval evaluations on validation set")
    parser.add_argument("--compute_map", default=False, help="if True computes mAP every tenth batch")
    parser.add_argument("--multiscale_training", default=True, help="allow for multi-scale training")
    opt = parser.parse_args()
    print(opt)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs("checkpoints", exist_ok=True)

    # Get data configuration
    data_config = parse_data_config(opt.data_config)
    train_path  = data_config["train"]
    valid_path  = data_config["valid"]
    class_names = load_classes(data_config["names"])

    # multi-sacle image size
    img_min_size = opt.img_size - 3*32
    img_max_size = opt.img_size + 3*32
    img_cur_size = img_max_size if opt.multiscale_training else opt.img_size

    # Get dataloader
    dataset = ListDataset(train_path, augment=True,
                          img_size=img_cur_size)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=opt.n_cpu,
        pin_memory=True,
        collate_fn=dataset.collate_fn,
    )

    # Initiate model
    model = Darknet(opt.model_def).to(device)
    model.apply(weights_init_normal)

    # If specified we start from checkpoint
    if opt.pretrained_weights:
        if opt.pretrained_weights.endswith(".pth"):
            model.load_state_dict(torch.load(opt.pretrained_weights))
        else:
            model.load_darknet_weights(opt.pretrained_weights)

    # Set up optimizer
    optimizer = torch.optim.Adam(model.parameters())

    ##=== main train ===
    for epoch in range(opt.epochs):
        model.train()
        if TQDM_USE: dataloader=tqdm(dataloader)
        for batch_i, (_, imgs, targets) in enumerate(dataloader):
            batches_done = len(dataloader) * epoch + batch_i

            # imgs.shape(batch_size, 3, img_size, img_size)
            # targets.shape(num_bboxes, 6_vals), 6_val=(idx, labels, x, y, w, h)

            ##=== multi-scale training ===
            # Select new image size every 10 batch
            if opt.multiscale_training and batch_i % 10 == 0:
                img_cur_size = random.choice(range(img_min_size, img_max_size + 1, 32))
            imgs = resize(imgs, img_cur_size)
 
            imgs    = Variable(imgs.to(device))
            targets = Variable(targets.to(device), requires_grad=False)

            loss, outputs = model(imgs, targets)
            loss.backward()

            if batches_done % opt.gradient_accumulations:
                # Accumulates gradient before each step
                optimizer.step()
                optimizer.zero_grad()

            model.seen += imgs.size(0)

            # === Log metrics at each YOLO layer ===
            # yolo.metrics: grid_size, loss, x, y, w, h, conf, cls,
            #               cls_acc, recall50, recall75, precision, conf_obj, conf_noobj
            log_str="---- [Epoch {}/{}, Batch {}/{}] ----\n".format(epoch, opt.epochs, batch_i, len(dataloader))
            metric_str = "{}-th YOLO: "\
                         "Grid size: {metrics[grid_size]}*{metrics[grid_size]},\t"\
                         "Loss: {metrics[loss]:.3f},\t Recall50: {metrics[recall50]:.3f},\t"\
                         "Recall75: {metrics[recall75]:.3f}\n"
            for yolo_i, yolo in enumerate(model.yolo_layers):
                log_str += metric_str.format(yolo_i, metrics=yolo.metrics)
            if not TQDM_USE: print(log_str)


        #=== Evaluate the model on the validation set ===
        if epoch % opt.evaluation_interval == 0:
            print("\n---- Evaluating Model ----")
            precision, recall, AP, f1, ap_class = evaluate(
                model,
                path=valid_path,
                iou_thres=0.5,
                conf_thres=0.5,
                nms_thres=0.5,
                img_size=opt.img_size,
                batch_size=8,
            )
            # Print class APs and mAP
            ap_table = [["Index", "Class name", "AP"]]
            for i, c in enumerate(ap_class):
                ap_table += [[c, class_names[c], "%.5f" % AP[i]]]
            print("---- mAP {}".format(AP.mean()))

        if epoch % opt.checkpoint_interval == 0:
            torch.save(model.state_dict(), f"checkpoints/yolov3_ckpt_%d.pth" % epoch)
