import argparse
import logging
import os

import numpy as np
import scipy.io as sio
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
import cv2 

from unet import UNet, NestedUNet
from utils.data_vis import plot_img_and_mask
from utils.dataset import BasicDataset

modelPath = 'checkpoints3/CP_epoch150.pth' # Your model's path

device_id = 1

def predict_img(net,
                full_img,
                device,
                scale_factor=1,
                out_threshold=0.5,
                input_index=0):
    net.eval()
    
    img = torch.from_numpy(BasicDataset.preprocess(transforms.Resize((224,224))(full_img), scale_factor))

    img = img.unsqueeze(0)
    img = img.cuda(device_id)#.to(device=device, dtype=torch.float32)
    img = img.float()
    with torch.no_grad():
        output = net(img)
        output = F.interpolate(output, size=300)
        if net.n_classes > 1:
            probs = F.softmax(output, dim=1)
        else:
            probs = torch.sigmoid(output)

        probs = probs.squeeze(0)
        save_array = probs.cpu().numpy()
        mat_prob=np.reshape(save_array,[300,300])
        mat_prob = mat_prob.astype('float32')
        save_fn = 'prob.mat'
        print(args.input[input_index])
        img_name_num = args.input[input_index].split('/')[2].split("_")
        img_category_id = img_name_num[0] + "_" + img_name_num[1]
    
        try:
            os.mkdir(os.path.join("ROC2/ROC_input",img_category_id))
        except:
            pass
        
        print(os.path.join(os.path.join("ROC2/ROC_input",img_category_id), save_fn))
        sio.savemat(os.path.join("ROC2/ROC_input",img_category_id) + "/"+save_fn, {'array' : mat_prob})
        tf = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize(full_img.size[1]),
                transforms.ToTensor()
            ]
        )

        probs = tf(probs.cpu())
        full_mask = probs.squeeze().cpu().numpy()

    return full_mask > out_threshold, os.path.join("ROC2/ROC_input",img_category_id)


def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model', '-m', default=modelPath,
                        metavar='FILE',
                        help="Specify the file in which the model is stored")
    parser.add_argument('--input', '-i', metavar='INPUT', nargs='+',
                        help='filenames of input images')#, required=True)

    parser.add_argument('--output', '-o', metavar='INPUT', nargs='+',
                        help='Filenames of ouput images')
    parser.add_argument('--viz', '-v', action='store_true',
                        help="Visualize the images as they are processed",
                        default=False)
    parser.add_argument('--no-save', '-n', action='store_true',
                        help="Do not save the output masks",
                        default=False)
    parser.add_argument('--mask-threshold', '-t', type=float,
                        help="Minimum probability value to consider a mask pixel white",
                        default=0.5)
    parser.add_argument('--scale', '-s', type=float,
                        help="Scale factor for the input images",
                        default=1)

    return parser.parse_args()


def get_output_filenames(args):
    in_files = args.input
    out_files = []

    if not args.output:
        for f in in_files:
            pathsplit = os.path.splitext(f)
            out_files.append("{}_OUT{}".format(pathsplit[0], pathsplit[1]))
    elif len(in_files) != len(args.output):
        logging.error("Input files and output files are not of the same length")
        raise SystemExit()
    else:
        out_files = args.output

    return out_files


def mask_to_image(mask):
    return Image.fromarray((mask * 255).astype(np.uint8))


if __name__ == "__main__":
    args = get_args()
    in_files = args.input
    
    in_files = []
    
    for i in os.listdir("data/img_b"):
    #for i in os.listdir("private_data_10/imgs/"):
        last_id = i.split(".")[0].split("_")[2] #os.path.join("data/img_b",i)
        if last_id == "1":
            in_files.append(os.path.join("data/img_b",i))
            #in_files.append(os.path.join("private_data_10/imgs/",i))
    print(in_files)
    
    args.input = in_files
    out_files = get_output_filenames(args)
    
    net = UNet(n_channels=3, n_classes=1,bilinear=True)
    #net = NestedUNet(num_classes=1, input_channels=3)
    logging.info("Loading model {}".format(args.model))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')
    
    net.load_state_dict(torch.load(args.model))#, map_location=device))
    net.cuda(device_id)#.to(device=device)
    
    
    logging.info("Model loaded !")

    for i, fn in enumerate(in_files):
        logging.info("\nPredicting image {} ...".format(fn))

        img = Image.open(fn)
        img=img.convert(mode='RGB')

        mask, img_save_path = predict_img(net=net,
                           full_img=img,
                           scale_factor=args.scale,
                           out_threshold=args.mask_threshold,
                           device=device,
                           input_index=i)
        
        if not args.no_save:
            out_fn = out_files[i]
            result = mask_to_image(mask)
            #result.save(out_files[i])
            result.save(img_save_path + "/" +args.input[i].split('/')[2])
            print(img_save_path + "/" +args.input[i].split('/')[2])
            logging.info("Mask saved to {}".format(out_files[i]))

        if args.viz:
            logging.info("Visualizing results for image {}, close to continue ...".format(fn))
            plot_img_and_mask(img, mask)
