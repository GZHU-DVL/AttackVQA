import torch
import h5py
import torchvision.transforms as transforms
import scipy.stats
import numpy as np
import os
import argparse
from CNNfeature_Spatial import ResNet50
# from CNNfeature_Spatial_TiVQA import ResNet50  #For TiVQA only
from VQAmodel_VSFA import VSFA
# from VQAmodel_TiVQA import TiVQA       #For TiVQA only
# from VQAmodel_MDTVSFA import MDTVSFA       #For MDTVSFA only
# from VQAmodel_BVQA_2022 import BVQA_2022         #For BVQA_2022 only
from VQAdataset import VQADataset
# from CNNfeature_Spatial_BVQA_2022 import CNNModel_Spatial      #For BVQA_2022 only
# from CNNfeature_Motion_BVQA_2022  import CNNModel_Motion       #For BVQA_2022 only
import skvideo.io
import torch.nn as nn
import torch.nn.functional as F
import csv
import time
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
torch.set_num_threads(3)
seed = 20230517
torch.manual_seed(seed)
np.random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
use_cuda = True

def get_Lsrb_value(model,video_data,label,extractor,length,q_hat_original_min,q_hat_original_max,median,config):  #Compute the Score-Reversed Boundary loss
    if label>=median: #Compute the boundary (disturbed quality score)
        boundary = torch.from_numpy(np.array(q_hat_original_min)).to('cuda')
    else:
        boundary = torch.from_numpy(np.array(q_hat_original_max)).to('cuda')
    fake_score =  quality_prediction(extractor,model, config.quality_model,video_data, length)  #Compute the estimated quality score of adversarial video
    Lsrb = F.l1_loss(fake_score,boundary)
    return Lsrb, fake_score, boundary

def patch_based(extractor,video_data, model,length,label,q_hat_original_min,q_hat_original_max, median,config):
    Z_patches = max(int(video_data.shape[1] *(video_data.shape[2]  // config.patch_size) * (video_data.shape[3] // config.patch_size) / config.query),1) #Select Z patches to perturb in one frame.
    pn = video_data.shape[1] * (video_data.shape[2] // config.patch_size)  * (video_data.shape[3] // config.patch_size)
    patch_heigth = (video_data.shape[2] // config.patch_size)  #The number of patches in the vertical direction
    patch_width = (video_data.shape[3] // config.patch_size)   #The number of patches in the horizontal direction
    perm = torch.randperm(pn)  #Encode information of the positions of the selected patches
    last_prob, last_score, boundary = get_Lsrb_value(model,video_data, label,extractor,length,q_hat_original_min,q_hat_original_max,median,config)  #Original quality score (estimated quality score)
    init_prob = last_prob
    init_score = last_score
    for n in range(0,pn,Z_patches):  # One round contains N queries.
        if pn - n < Z_patches:
            Z_patches = pn - n
        diff = torch.zeros_like(video_data)
        m_n = (2 * np.random.randint(0, 2, size=(config.patch_size*config.patch_size)) - 1)  #Generate universal perturbation map from a discrete set {−γ, +γ}
        for num_patch in range(Z_patches):  #Compute the position of the selected patch in the frame
            channel = (perm[n+num_patch] // patch_heigth // patch_width)
            heigth_begin = (perm[n+num_patch] - (patch_heigth * patch_width * channel)) // patch_width
            width_begin = (perm[n+num_patch] - (patch_heigth * patch_width * channel)) - (patch_width * heigth_begin)
            diff[:,channel,(heigth_begin * config.patch_size):(heigth_begin * config.patch_size)+config.patch_size,(width_begin * config.patch_size)
            :(width_begin * config.patch_size)+config.patch_size] = config.gamma * torch.from_numpy(m_n.reshape(1,config.patch_size,config.patch_size)).to('cuda')
        left_prob, left_score, boundary = get_Lsrb_value(model,(video_data-diff).clamp(0,1), label,extractor,length,q_hat_original_min,q_hat_original_max,median,config)  #Evaluate the attack effect.
        if left_prob < last_prob:
            video_data = (video_data - diff).clamp(0,1) #Update the adversarial video.
            last_prob = left_prob
            last_score = left_score
        else:
            right_prob, right_score, boundary = get_Lsrb_value(model,(video_data+diff).clamp(0,1), label,extractor,length,q_hat_original_min,q_hat_original_max,median,config) #  Evaluate the attack effect.
            if right_prob < last_prob:
                video_data = (video_data + diff).clamp(0,1) # Update the adversarial video.
                last_prob = right_prob
                last_score = right_score
    print(last_prob - init_prob)
    return video_data.squeeze(), boundary

def quantize(x):
    quantizer = transforms.ToPILImage()
    x = quantizer(x.squeeze().cpu())
    return x

test_transform = transforms.Compose([
    transforms.ToTensor(),
])

def quality_prediction(spatial_extractor,model, quality_model,video_data,length, motion_extractor = None):
    if quality_model == 1:
        s = model(spatial_extractor,video_data,length)
    elif quality_model == 2:
        relative_score, mapped_score, aligned_score = model(spatial_extractor, video_data, length)
    elif quality_model == 3:
        s = model(spatial_extractor,video_data,length)
    elif quality_model == 4:
        relative_score, mapped_score, aligned_score = model(spatial_extractor, motion_extractor, video_data, length)
    return s       # Or return relative_score, mapped_score, aligned_score

def query_method(quality_model):
    if quality_model == 1:
        method = 'VSFA'
    elif quality_model == 2:
        method = 'MDTVSFA'
    elif quality_model == 3:
        method = 'TiVQA'
    elif quality_model == 4:
        method = 'BVQA-2022'
    return method

def do_attack(config, model):
    q_mos = []
    q_hat_original = []
    q_hat_adv = []
    l2 = []
    R_value_list = []
    index_list = []
    if config.trained_datasets[0] == 'K':
        videos_dir = 'KoNViD-1k_video/'  # videos dir
        features_dir = './Features/VSFA_K_features/'  # features dir
        datainfo = 'data/KoNViD-1kinfo.mat'
    if config.trained_datasets[0] == 'N':
        videos_dir = 'LIVE-VQC/'
        features_dir = './Features/VSFA_N_features/'
        datainfo = 'data/LIVE-VQCinfo.mat'
    if config.trained_datasets[0] == 'Y':
        videos_dir = 'Youtube-UGC/'
        features_dir = './Features/VSFA_Y_features/'
        datainfo = 'data/YOUTUBE_UGC_metadata.csv'
    if config.trained_datasets[0] == 'Q':
        videos_dir = 'LSVQ/'
        features_dir = './Features/VSFA_Q_features/'
        datainfo1 = 'data/labels_train_test.csv'
        datainfo2 = 'data/labels_test_1080p.csv'

    Info = h5py.File(datainfo, 'r')
    video_names = [Info[Info['video_names'][0, :][i]][()].tobytes()[::2].decode() for i in
                   range(len(Info['video_names'][0, :]))]

    for idx in range(len(video_names)):
        if video_names[idx][11] == '_':
            video_names[idx] = video_names[idx][0:11] + '.mp4'
        elif video_names[idx][10] == '_':
            video_names[idx] = video_names[idx][0:10] + '.mp4'
        else:
            video_names[idx] = video_names[idx][0:9] + '.mp4'
    index = Info['index']
    index = index[1100:1150]
    index = index[:, 0 % index.shape[1]]
    ref_ids = Info['ref_ids'][0, :]  #
    max_len = int(Info['max_len'][0])
    testindex = index
    test_index = []
    for i in range(len(ref_ids)):
        if i in testindex:
            test_index.append(i)

    mos_max = Info['scores'][0, :].max()
    train_dataset = VQADataset(features_dir, test_index, max_len, scale=mos_max)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=config.batch_size, shuffle=True,
                                               drop_last=True)
    cnt = 0
    device = torch.device("cuda" if torch.cuda.is_available() and use_cuda else "cpu")
    extractor = ResNet50().to('cuda')
    extractor.eval()
    # motion_extractor = CNNModel_Motion(model='MotionExtractor').to(device)      # For BVQA-2022 only
    # motion_extractor.eval()
    # spatial_extractor = CNNModel_Spatial(model='SpatialExtractor').to(device)
    # spatial_extractor.eval()


    for i, (length, label, index) in enumerate(train_loader):
        q_mos.append(label[0][0])
        median = np.median(np.array(q_mos))  # The threshold to decide whether a video is of high quality or low quality
        video_data = skvideo.io.vread(os.path.join(videos_dir, video_names[index[0]]))  # Original video
        video_data = video_data.transpose(0, 3, 1, 2) / 255
        length[0][0] = video_data.shape[0]
        with torch.no_grad():
            sa = quality_prediction(extractor, model, config.quality_model, torch.from_numpy(video_data),length)  #Original quality score (estimated quality score)
        q_hat_original.append(sa.item())
        index_list.append(index)

    q_hat_original_min = (((0) - np.mean(q_mos)) / np.std(q_mos)) * np.std(q_hat_original) + np.mean(q_hat_original)  # Compute the boundary according to the distribution quality scores estimated by target NR-VQA models.
    q_hat_original_max = (((1) - np.mean(q_mos)) / np.std(q_mos)) * np.std(q_hat_original) + np.mean(q_hat_original)

    for video_index in range(len(q_mos)):
        start = time.time()
        video_data = skvideo.io.vread(os.path.join(videos_dir, video_names[index_list[video_index]]))
        video_data = video_data.transpose(0, 3, 1, 2) / 255
        video_data_intermediate = video_data
        video_data_intermediate = torch.from_numpy(video_data_intermediate).float()
        perturbations = torch.zeros_like(video_data_intermediate)
        adversarial_video = torch.zeros_like(video_data_intermediate)  # Adversarial video

        if config.attack_trigger == 1:
            for index in range(0,video_data_intermediate.shape[0],1): #The optimization process contains X/T rounds
                video_data_round = video_data_intermediate[index:index + 1]
                video_data_round = video_data_round.to(device)
                length[0][0] = video_data_round.shape[0]
                adv_round, boundary = patch_based(extractor,video_data_round, model, length,q_mos[video_index],q_hat_original_min,q_hat_original_max,median,config)
                adv_round = torch.round(adv_round * 255) / 255
                adversarial_video[index:index + 1] = adv_round.cpu().detach() #Adversarial frames in one round
                perturbations[index:index + 1] = adv_round.cpu().detach() - video_data_round.cpu().detach()
            with torch.no_grad():
                length[0][0] = adversarial_video.shape[0]
                sa_adv = quality_prediction(extractor, model, config.quality_model, adversarial_video, length)  # Compute the estimated quality score of adversarial video
                R_value_list.append(np.log((abs((q_hat_original[video_index] - boundary).cpu())) / (
                            abs((sa_adv - q_hat_original[video_index]).cpu()) + 1e-12)))  # Compute the R_value
                l2.append((((torch.mean(torch.norm((perturbations).contiguous().view((perturbations).shape[0], -1), dim=1),dim=0))**2)
                           /(adversarial_video.shape[1]*adversarial_video.shape[2]*adversarial_video.shape[3]))**0.5)  #Compute the pixel-level L2 norm
                q_hat_adv.append(sa_adv.item())

        method_folder = query_method(config.quality_model)
        attack_folder = os.path.join(config.attack_folder, method_folder, file_name)
        print("attack_folder", attack_folder)
        if not os.path.exists(attack_folder):
            os.makedirs(attack_folder)

        original_folder = os.path.join(config.original_folder, method_folder, file_name)
        print("original_folder", original_folder)
        if not os.path.exists(original_folder):
            os.makedirs(original_folder)

        if config.save_original:  # Save the original video frames
            for frame_index in range(video_data_intermediate.shape[0]):
                if frame_index % 100 == 0:
                    save_name = str(video_names[video_index] + '_') + str(cnt) + '.png'
                    path = os.path.join(original_folder, save_name)
                    original_index = quantize(video_data_intermediate[frame_index])
                    original_index.save(path)

        if config.save_attack & config.attack_trigger:  # Save the adversarial video frames
            for frame_index in range(adversarial_video.shape[0]):
                if frame_index % 100 == 0:
                    save_name = str(video_names[video_index] + '_') + str(cnt) + '.png'
                    cnt = cnt + 1
                    path = os.path.join(attack_folder, save_name)
                    adv_index = quantize(adversarial_video[frame_index])
                    adv_index.save(path)

        end = time.time()
        print('{} seconds'.format(end - start))

    l2 = torch.mean(torch.Tensor(l2))
    print("l2", l2)

    f = open(str(attack_folder + 'score'), 'w', encoding='utf-8', newline='' "")
    csv_writer = csv.writer(f)
    csv_writer.writerow(["MOS", "Original_scores", "Fake_scores"])
    for i in range(len(q_hat_original)):
        result = [q_mos[i].item(), q_hat_original[i], q_hat_adv[i]]
        csv_writer.writerow(result)

    R_value = torch.mean(torch.Tensor(R_value_list))
    print("R_value", R_value)

    srcc_original = scipy.stats.mstats.spearmanr(x=q_mos, y=q_hat_original)[0]
    print("srcc_original",srcc_original)

    krcc_original = scipy.stats.kendalltau(x=q_mos, y=q_hat_original)[0]
    print("krcc_original", krcc_original)

    plcc_original = scipy.stats.mstats.pearsonr(x=q_mos, y=q_hat_original)[0]
    print("plcc_original", plcc_original)

    rmse_original = np.sqrt(((np.array(q_mos) - np.array(q_hat_original)) ** 2).mean())
    print("rmse_original", rmse_original)

    srcc_adv = scipy.stats.mstats.spearmanr(x=q_mos, y=q_hat_adv)[0]
    print("srcc_adv", srcc_adv)

    krcc_adv = scipy.stats.kendalltau(x=q_mos, y=q_hat_adv)[0]
    print("krcc_adv", krcc_adv)

    plcc_adv = scipy.stats.mstats.pearsonr(x=q_mos, y=q_hat_adv)[0]
    print("plcc_adv", plcc_adv)

    rmse_adv = np.sqrt(((np.array(q_mos) - np.array(q_hat_adv)) ** 2).mean())
    print("rmse_adv", rmse_adv)

def parse_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quality_model", type=int, default=1)  ## 1:VSFA  2: MDTVSFA  3: TiVQA  4:BVQA-2022
    parser.add_argument("--query", type=int, default=300)
    parser.add_argument("--gamma", type=float, default=5/255)
    parser.add_argument("--patch_size", type=int, default=56)
    parser.add_argument("--attack_folder", type=str, default="./counterexample")
    parser.add_argument("--original_folder", type=str, default="./original")
    parser.add_argument("--attack_trigger", type=int, default=1)
    parser.add_argument("--save_attack", type=bool, default=True)
    parser.add_argument("--save_original", type=bool, default=True)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument('--trained_datasets', nargs='+', type=str, default=['K'], # K: KoNViD-1k  N: LIVE-VQC  Y: YouTube-UGC  Q: LSVQ
                        help="trained datasets (default: ['K', 'N', 'Y', 'Q'])")
    return parser.parse_args()

def main():
    config = parse_config()
    device = torch.device("cuda" if torch.cuda.is_available() and use_cuda else "cpu")
    if config.quality_model == 1:
        BVQA = VSFA()
        pretrained_model = './models/VSFA_K'
        BVQA.load_state_dict(torch.load(pretrained_model))
    elif config.quality_model == 2:
        BVQA = MDTVSFA()
        pretrained_model = './models/MDTVSFA-KoNViD-1k'
        BVQA.load_state_dict(torch.load(pretrained_model))
    elif config.quality_model == 3:
        BVQA = TiVQA()
        pretrained_model = './models/TiVQA-KoNViD-1k'
        BVQA.load_state_dict(torch.load(pretrained_model))
    elif config.quality_model == 4:
        BVQA = BVQA_2022()
        pretrained_model = "./models/BVQA-2022-KoNViD-1k"
        BVQA.load_state_dict(torch.load(pretrained_model))
    BVQA = BVQA.to(device)
    BVQA.train()
    for k, v in BVQA.named_parameters():
        v.requires_grad = False
        for name, module in BVQA.named_modules():
            if isinstance(module, nn.BatchNorm2d):
                module.training = False
    do_attack(config, BVQA)

config = parse_config()
method_folder = query_method(config.quality_model)
file_name = os.path.join(method_folder + '_black/')


if __name__ == "__main__":
    main()
