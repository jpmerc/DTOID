import sys
import argparse
import os
import cv2
import yaml
from PIL import Image
from importlib.machinery import SourceFileLoader
import torch
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import pandas
import numpy


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Evaluation on Linemod dataset')
    parser.add_argument('--no_filter_z', help="no post-filtering", action="store_true")
    parser.add_argument('--obj_id', help="Index of the object to test", action="store", default=1, type=int)
    parser.add_argument('--all_obj', help="Test all objects", action="store_true")

    arguments = parser.parse_args()
    no_filter_z = arguments.no_filter_z
    obj_id = arguments.obj_id
    all_obj = arguments.all_obj

    # Objects
    linemod_objects = []
    if all_obj:
        linemod_objects = ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12", "13", "14", "15"]
    else:
        if obj_id > 0 and obj_id <= 15:
            linemod_objects.append(str(obj_id).zfill(2))
        else:
            linemod_objects.append("01")

    # evaluation paths
    model_directory = "./templates"
    dataset_path = "./linemod"
    backend = "cuda"

    # Load Network
    network_module = SourceFileLoader(".", "network.py").load_module()
    model = network_module.Network()
    model.eval()
    checkpoint = torch.load("model.pth.tar", map_location=lambda storage, loc: storage)
 

    model.load_state_dict(checkpoint["state_dict"])
    if backend == "cuda":
        model = model.cuda()
    preprocess = network_module.PREPROCESS



    print("Process templates")
    for linemod_model in linemod_objects:
        # Ground Truth
        scene_id = linemod_model
        scene = os.path.join(dataset_path, scene_id)
        gt_path = os.path.join(scene, "gt.yml")
        gt = yaml.load(open(gt_path, "r"))

        # RGB image paths
        files = [os.path.join(scene, "rgb", x) for x in os.listdir(os.path.join(scene, "rgb")) if ".png" in x]
        files.sort()

        model_name = "hinterstoisser_" + linemod_model
        template_dir = os.path.join(model_directory, model_name)
        output_file = "{}.yml".format(model_name)

        #load text file
        pose_file = os.path.join(template_dir, "poses.txt")
        pose_file_np = pandas.read_csv(pose_file, sep=" ", header=None).values
        pose_z_values = pose_file_np[:, 11]

        # Template
        global_template_list = []
        template_paths = [x for x in os.listdir(template_dir) if len(x) == 12 and "_a.png" in x]
        template_paths.sort()
        preprocessed_templates = []

        # features for all templates (240)
        template_list = []
        template_global_list = []
        template_ratios_list = []

        batch_size = 10
        temp_batch_local = []
        temp_batch_global = []
        temp_batch_ratios = []
        iteration = 0

        for t in tqdm(template_paths):
            # open template and template mask
            template_im = cv2.imread(os.path.join(template_dir, t))[:, :, ::-1]
            template = Image.fromarray(template_im)

            template_mask = cv2.imread(os.path.join(template_dir, t.replace("_a", "_m")))[:, :, 0]
            template_mask = Image.fromarray(template_mask)

            # preprocess and concatenate
            template = preprocess[1](template)
            template_mask = preprocess[2](template_mask)
            template = torch.cat([template, template_mask], dim=0)

            if backend == "cuda":
                template = template.cuda()

            template_feature = model.compute_template_local(template.unsqueeze(0))

            # Create mini-batches of templates
            if iteration == 0:
                temp_batch_local = template_feature

                template_feature_global = model.compute_template_global(template.unsqueeze(0))
                template_global_list.append(template_feature_global)

            elif iteration % (batch_size) == 0:
                template_list.append(temp_batch_local)
                temp_batch_local = template_feature

            elif iteration == (len(template_paths) - 1):
                temp_batch_local = torch.cat([temp_batch_local, template_feature], dim=0)
                template_list.append(temp_batch_local)

            else:
                temp_batch_local= torch.cat([temp_batch_local, template_feature], dim=0)

            iteration += 1


        # ==== eval ====

        results = {}
        results_corr = []
        results_depth = {}


        all_gt_preds = []

        good_preds = []
        bad_preds = []

        for i, file in tqdm(enumerate(files)):
            img_numpy = cv2.imread(file)
            img_h, img_w, img_c = img_numpy.shape

            img = Image.fromarray(img_numpy[:, :, ::-1])
            img = preprocess[0](img)

            network_h = img.size(1)
            network_w = img.size(2)
            if backend == "cuda":
                img = img.cuda()

            top_k_num = 500
            top_k_scores, top_k_bboxes, top_k_template_ids = model.forward_all_templates(
                img.unsqueeze(0), template_list, template_global_list, topk=top_k_num)


            pred_scores_np = top_k_scores.cpu().numpy()
            pred_bbox_np = top_k_bboxes.cpu().numpy()
            pred_template_ids = top_k_template_ids[:, 0].long().cpu().numpy()
            template_z_values = pose_z_values[pred_template_ids]

            
            if not no_filter_z:
            
                pred_w_np = pred_bbox_np[:, 2] - pred_bbox_np[:, 0]
                pred_h_np = pred_bbox_np[:, 3] - pred_bbox_np[:, 1]
                pred_max_dim_np = np.stack([pred_w_np, pred_h_np]).transpose().max(axis=1)
                pred_z = (124 / pred_max_dim_np) * -template_z_values

                # Filter based on predicted Z values
                pred_z_conds = (pred_z > 0.4) & (pred_z < 2)
                pred_z_conds_ids = numpy.where(pred_z_conds)[0]

                pred_scores_np = pred_scores_np[pred_z_conds_ids]
                pred_bbox_np = pred_bbox_np[pred_z_conds_ids]
                pred_template_ids = pred_template_ids[pred_z_conds_ids]
                pred_z = pred_z[pred_z_conds_ids]


            # Keep top 1 (eval)
            pred_scores_np = pred_scores_np[:1]
            pred_bbox_np = pred_bbox_np[:1]
            pred_template_ids = pred_template_ids[:1]
            pred_z = pred_z[:1]



            # Show prediction
            if len(pred_bbox_np) > 0:
                x1, y1, x2, y2 = pred_bbox_np[0]
                temp_score = pred_scores_np[0]

                x1 = int(x1 / network_w * img_w)
                x2 = int(x2 / network_w * img_w)
                y1 = int(y1 / network_h * img_h)
                y2 = int(y2 / network_h * img_h)

                rec_color = (0, 255, 255)
                cv2.rectangle(img_numpy,
                              (x1, y1),
                              (x2, y2),
                              rec_color,2)



            im_id = (os.path.basename(file).split('.')[0]).zfill(6)
            for bb in gt[int(im_id)]:
                if bb['obj_id'] == int(linemod_model):
                    x1, y1, w, h = bb["obj_bb"]
                    cv2.rectangle(img_numpy,
                                  (x1, y1),
                                  (x1 + w, y1 + h),
                                  (0, 0, 255), 2)


            plt.subplot(1, 1, 1)
            plt.imshow(img_numpy[:, :, ::-1])
            plt.show()







