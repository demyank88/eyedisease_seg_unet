import os
from labelme.util import *
import argparse
import pandas as pd

dataset_path = "/mnt/d/business/alphado/dataset-003"

BASE_FOLDER = f"{dataset_path}/refined_dataset/keratitis/"
IMAGE_FOLDER = os.path.join(BASE_FOLDER,'img')
SEG_FOLDER = os.path.join(BASE_FOLDER,'labelme')
BOX_FOLDER = os.path.join(BASE_FOLDER,'labelimg')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate a new image dataset by cropping and producing segmentation images.')
    # parser.add_argument('--input_image_dir', '-i', default=f"{dataset_path}/refined_dataset/keratitis/img/",
    parser.add_argument('--input_image_dir', '-i', default=f"{dataset_path}/scanner_capture/eye/corneal_ulcer/corneal_ulcer_images/",
                        type=str, help='an input folder to original dataset')
    parser.add_argument('--input_json_dir', '-j', default=f"{dataset_path}/scanner_capture/eye/corneal_ulcer/corneal_ulcer_labelme/",
                        type=str, help='an input folder including json files of segmentation')
    parser.add_argument('--output_dir', '-o', default=f"{dataset_path}/train/",type=str, help='an input folder for refined data to be saved')
    args = parser.parse_args()
    if args.input_image_dir is None:
        dict_name2=dict()
        name2pixelValue = pd.read_csv("labelme/name_list_with_pixel_value.csv", index_col=0, skiprows=0).T.to_dict()
        original_image, label_image = draw_label_png(args.input_json_seg_file,name2pixelValue)
        cropped_ori_img, cropped_lbl_img = crop_label_png(args.input_xml_bbox_file, original_image, label_image)
        show_img(cropped_ori_img)
        show_img(cropped_lbl_img)
    else:
        # list_xml_files = os.listdir(args.input_xml_dir)
        list_json_files = os.listdir(args.input_json_dir)
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        i=0
        for jsonfilename in list_json_files:
            # full_path2xml = os.path.join(args.input_xml_dir,afile)
            full_path2json = os.path.join(args.input_json_dir,jsonfilename)

            name2pixelValue = pd.read_csv(BASE_DIR+"/../labelme/name_list_with_pixel_value.csv", index_col=0,skiprows=0).T.to_dict()
            original_image, label_image = draw_label_png(full_path2json, name2pixelValue)
            # list_cropped_ori_img, list_cropped_lbl_img = crop_label_png(full_path2xml, original_image, label_image)
            # for cropped_ori_img, cropped_lbl_img in zip(list_cropped_ori_img, list_cropped_lbl_img):
            path2save_orig_img = args.output_dir + "/img/"
            path2save_seg_img = args.output_dir + "/seg/"
            if not os.path.isdir(path2save_orig_img):
                os.makedirs(path2save_orig_img,0o777)
            if not os.path.isdir(path2save_seg_img):
                os.makedirs(path2save_seg_img,0o777)
            dst_path2save_orig_img = path2save_orig_img + '{:08d}.png'.format(i)
            dst_path2save_seg_img = path2save_seg_img + '{:08d}.png'.format(i)
            while(os.path.isfile(dst_path2save_orig_img)):
                i += 1
                dst_path2save_orig_img = path2save_orig_img + '{:08d}.png'.format(i)
                dst_path2save_seg_img = path2save_seg_img + '{:08d}.png'.format(i)

            save_img(dst_path2save_orig_img,original_image)
            save_img(dst_path2save_seg_img,label_image)
            i+=1

            # show_img(cropped_ori_img)
            # show_img(cropped_lbl_img)





