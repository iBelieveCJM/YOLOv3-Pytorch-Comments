from __future__ import division

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import NullLocator

from models import Darknet
from utils.utils import load_classes, rescale_boxes
from utils.datasets import ImageFolder
from utils.nms import non_max_suppression

import os
import random

import tqdm
import numpy as np
import argparse
from PIL import Image

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.autograd import Variable

TQDM_USE=True

def get_detections(model, dataloader, conf_thres, nms_thres, img_size, batch_size):
    model.eval()

    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
    if TQDM_USE: dataloader=tqdm.tqdm(dataloader)

    img_paths = []
    img_detections = []
    for batch_i, data in enumerate(dataloader):

        if len(data)==3:
            paths, imgs, target = data
        elif len(data)==2:
            paths, imgs = data

        imgs = Variable(imgs.type(Tensor), requires_grad=False)

        with torch.no_grad():
            outputs = model(imgs)
            outputs = non_max_suppression(outputs, conf_thres=conf_thres, nms_thres=nms_thres)

        # Save image path and detections
        img_paths.extend(paths)
        img_detections.extend(outputs)

    return img_paths, img_detections

def plot_detections(img_paths, img_detections, class_names, output_path):
    # bbox colors
    cmap = plt.get_cmap("tab20b")
    colors = [cmap(i) for i in np.linspace(0,1,len(class_names))]

    print("\nSaving images:")
    os.makedirs(output_path, exist_ok=True)

    # Iterate through images and save plot of detections
    for img_i, (path, detections) in enumerate(zip(img_paths, img_detections)):

        print("(%d) Image: '%s'" % (img_i, path))

        # Create plot
        img = np.array(Image.open(path))
        plt.figure()
        fig, ax = plt.subplots(1)
        ax.imshow(img)

        # Draw bounding boxes and labels of detections
        if detections is not None:

            # Rescale boxes to original image
            detections = rescale_boxes(detections, opt.img_size, img.shape[:2])

            unique_labels = detections[:, -1].cpu().unique()
            n_cls_preds = len(unique_labels)
            bbox_colors = random.sample(colors, n_cls_preds)

            for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:

                print("\t+ Label: %s, Conf: %.5f" % (class_names[int(cls_pred)], cls_conf.item()))

                box_w = x2 - x1
                box_h = y2 - y1

                color = bbox_colors[int(np.where(unique_labels == int(cls_pred))[0])]
                # Create a Rectangle patch
                bbox = patches.Rectangle((x1, y1), box_w, box_h, linewidth=2, edgecolor=color, facecolor="none")
                # Add the bbox to the plot
                ax.add_patch(bbox)
                # Add label
                plt.text(
                    x1,
                    y1,
                    s=class_names[int(cls_pred)],
                    color="white",
                    verticalalignment="top",
                    bbox={"color": color, "pad": 0},
                )

        # Save generated image with detections
        plt.axis("off")
        plt.gca().xaxis.set_major_locator(NullLocator())
        plt.gca().yaxis.set_major_locator(NullLocator())
        filename = path.split("/")[-1].split(".")[0]
        fig.savefig("{}/{}.png".format(output_path, filename),
                    bbox_inches="tight", pad_inches=0.0)
        plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_folder", type=str, default="data/samples", help="path to dataset")
    parser.add_argument("--model_def", type=str, default="config/yolov3.cfg", help="path to model definition file")
    parser.add_argument("--weights_path", type=str, default="weights/yolov3.weights", help="path to weights file")
    parser.add_argument("--class_path", type=str, default="data/coco.names", help="path to class label file")
    parser.add_argument("--conf_thres", type=float, default=0.8, help="object confidence threshold")
    parser.add_argument("--nms_thres", type=float, default=0.4, help="iou thresshold for non-maximum suppression")
    parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
    parser.add_argument("--n_cpu", type=int, default=0, help="number of cpu threads to use during batch generation")
    parser.add_argument("--img_size", type=int, default=416, help="size of each image dimension")
    parser.add_argument("--checkpoint_model", type=str, help="path to checkpoint model")
    parser.add_argument("--output_path", type=str, default="detect_outputs", help="path to output image")
    opt = parser.parse_args()
    print(opt)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    class_names = load_classes(opt.class_path)  # Extracts class labels from file

    # Initiate model
    model = Darknet(opt.model_def).to(device)
    if opt.weights_path.endswith(".weights"):
        # Load darknet weights
        model.load_darknet_weights(opt.weights_path)
    else:
        # Load checkpoint weights
        model.load_state_dict(torch.load(opt.weights_path))
 
    ## Get dataloader
    dataloader = DataLoader(
        ImageFolder(opt.image_folder, img_size=opt.img_size),
        batch_size=opt.batch_size,
        shuffle=False,
        num_workers=opt.n_cpu,
    )


    print("Start plot.")

    paths, detections = get_detections(
        model,
        dataloader,
        conf_thres=opt.conf_thres,
        nms_thres=opt.nms_thres,
        img_size=opt.img_size,
        batch_size=8
    )
    plot_detections(paths, detections, class_names, opt.output_path)

    print("Plot done.")
