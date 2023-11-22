import torch
import numpy as np
import os
from utils.reranking import re_ranking
import numpy as np
import numpy as np
import matplotlib.pyplot as plt
from sklearn import manifold
# from sklearn import manifold


def euclidean_distance(qf, gf):
    m = qf.shape[0]
    n = gf.shape[0]
    dist_mat = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
               torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    dist_mat.addmm_(qf, gf.t(), beta=1, alpha=-2)
    return dist_mat.cpu().numpy()


def cosine_similarity(qf, gf):
    epsilon = 0.00001
    dist_mat = qf.mm(gf.t())
    qf_norm = torch.norm(qf, p=2, dim=1, keepdim=True)  # mx1
    gf_norm = torch.norm(gf, p=2, dim=1, keepdim=True)  # nx1
    qg_normdot = qf_norm.mm(gf_norm.t())

    dist_mat = dist_mat.mul(1 / qg_normdot).cpu().numpy()
    dist_mat = np.clip(dist_mat, -1 + epsilon, 1 - epsilon)
    dist_mat = np.arccos(dist_mat)
    return dist_mat


def eval_func_sysu(distmat, q_pids, g_pids, q_camids, g_camids, max_rank=20):
    """Evaluation with sysu metric
    Key: for each query identity, its gallery images from the same camera view are discarded. "Following the original setting in ite dataset"
    """
    num_q, num_g = distmat.shape
    if num_g < max_rank:
        max_rank = num_g
        print("Note: number of gallery samples is quite small, got {}".format(num_g))
    indices = np.argsort(distmat, axis=1)
    pred_label = g_pids[indices]
    matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)

    # compute cmc curve for each query
    new_all_cmc = []
    all_cmc = []
    all_AP = []
    all_INP = []
    num_valid_q = 0.  # number of valid query
    for q_idx in range(num_q):
        # get query pid and camid
        q_pid = q_pids[q_idx]
        q_camid = q_camids[q_idx]

        # remove gallery samples that have the same pid and camid with query
        order = indices[q_idx]
        remove = (q_camid == 2) & (g_camids[order] == 1)
        keep = np.invert(remove)

        # compute cmc curve
        # the cmc calculation is different from standard protocol
        # we follow the protocol of the author's released code
        new_cmc = pred_label[q_idx][keep]
        new_index = np.unique(new_cmc, return_index=True)[1]
        new_cmc = [new_cmc[index] for index in sorted(new_index)]

        new_match = (new_cmc == q_pid).astype(np.int32)
        new_cmc = new_match.cumsum()
        new_all_cmc.append(new_cmc[:max_rank])

        orig_cmc = matches[q_idx][keep]  # binary vector, positions with value 1 are correct matches
        if not np.any(orig_cmc):
            # this condition is true when query identity does not appear in gallery
            continue

        cmc = orig_cmc.cumsum()

        # compute mINP
        # refernece Deep Learning for Person Re-identification: A Survey and Outlook
        pos_idx = np.where(orig_cmc == 1)
        pos_max_idx = np.max(pos_idx)
        inp = cmc[pos_max_idx] / (pos_max_idx + 1.0)
        all_INP.append(inp)

        cmc[cmc > 1] = 1

        all_cmc.append(cmc[:max_rank])
        num_valid_q += 1.

        # compute average precision
        # reference: https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Average_precision
        num_rel = orig_cmc.sum()
        tmp_cmc = orig_cmc.cumsum()
        tmp_cmc = [x / (i + 1.) for i, x in enumerate(tmp_cmc)]
        tmp_cmc = np.asarray(tmp_cmc) * orig_cmc
        AP = tmp_cmc.sum() / num_rel
        all_AP.append(AP)

    assert num_valid_q > 0, "Error: all query identities do not appear in gallery"

    all_cmc = np.asarray(all_cmc).astype(np.float32)
    all_cmc = all_cmc.sum(0) / num_valid_q  # standard CMC

    new_all_cmc = np.asarray(new_all_cmc).astype(np.float32)
    new_all_cmc = new_all_cmc.sum(0) / num_valid_q
    mAP = np.mean(all_AP)
    mINP = np.mean(all_INP)
    return new_all_cmc, mAP, mINP


def eval_func_msrv(distmat, q_pids, g_pids, q_camids, g_camids, q_sceneids, g_sceneids, max_rank=50):
    """Evaluation with market1501 metric
        Key: for each query identity, its gallery images from the same camera view are discarded.
        """
    num_q, num_g = distmat.shape
    if num_g < max_rank:
        max_rank = num_g
        print("Note: number of gallery samples is quite small, got {}".format(num_g))
    indices = np.argsort(distmat, axis=1)

    query_arg = np.argsort(q_pids, axis=0)
    result = g_pids[indices]
    gall_re = result[query_arg]
    gall_re = gall_re.astype(np.str)
    # pdb.set_trace()

    result = gall_re[:, :100]

    # with open("re.txt", 'w') as file_obj:
    #     for li in result:
    #         for j in range(len(li)):
    #             if j == len(li) - 1:
    #                 file_obj.write(li[j] + "\n")
    #             else:
    #                 file_obj.write(li[j] + " ")
    with open('re.txt', 'w') as f:
        f.write('rank list file\n')

    # pdb.set_trace()
    matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)

    # compute cmc curve for each query
    all_cmc = []
    all_AP = []
    num_valid_q = 0.  # number of valid query
    for q_idx in range(num_q):
        # get query pid and camid
        q_pid = q_pids[q_idx]
        q_camid = q_camids[q_idx]

        q_sceneid = q_sceneids[q_idx]

        # remove gallery samples that have the same pid and camid with query
        order = indices[q_idx]
        # original protocol in RGBNT100 or RGBN300
        # remove = (g_pids[order] == q_pid) & (g_camids[order] == q_camid)

        # for each query sample, its gallery samples from same scene with same or neighbour view are discarded # added by zxp
        # symmetrical_cam = (8 - q_camid) % 8
        # remove = (g_pids[order] == q_pid) & ( # same id
        #              (g_sceneids[order] == q_sceneid) & # same scene
        #              ((g_camids[order] == q_camid) | (g_camids[order] == (q_camid + 1)%8) | (g_camids[order] == (q_camid - 1)%8) | # neighbour cam with q_cam
        #              (g_camids[order] == symmetrical_cam) | (g_camids[order] == (symmetrical_cam + 1)%8) | (g_camids[order] == (symmetrical_cam - 1)%8)) # nerighboour cam with symmetrical cam
        #          )
        # new protocol in MSVR310
        remove = (g_pids[order] == q_pid) & (g_sceneids[order] == q_sceneid)
        keep = np.invert(remove)

        with open('re.txt', 'a') as f:
            f.write('{}_s{}_v{}:\n'.format(q_pid, q_sceneid, q_camid))
            v_ids = g_pids[order][keep][:max_rank]
            v_cams = g_camids[order][keep][:max_rank]
            v_scenes = g_sceneids[order][keep][:max_rank]
            for vid, vcam, vscene in zip(v_ids, v_cams, v_scenes):
                f.write('{}_s{}_v{}  '.format(vid, vscene, vcam))
            f.write('\n')

        # compute cmc curve
        # binary vector, positions with value 1 are correct matches
        orig_cmc = matches[q_idx][keep]
        if not np.any(orig_cmc):
            # this condition is true when query identity does not appear in gallery
            continue

        cmc = orig_cmc.cumsum()
        cmc[cmc > 1] = 1

        all_cmc.append(cmc[:max_rank])
        num_valid_q += 1.

        # compute average precision
        # reference: https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Average_precision
        num_rel = orig_cmc.sum()
        tmp_cmc = orig_cmc.cumsum()
        tmp_cmc = [x / (i + 1.) for i, x in enumerate(tmp_cmc)]
        tmp_cmc = np.asarray(tmp_cmc) * orig_cmc
        AP = tmp_cmc.sum() / num_rel
        all_AP.append(AP)

    assert num_valid_q > 0, "Error: all query identities do not appear in gallery"

    all_cmc = np.asarray(all_cmc).astype(np.float32)
    all_cmc = all_cmc.sum(0) / num_valid_q
    mAP = np.mean(all_AP)

    return all_cmc, mAP


def eval_func(distmat, q_pids, g_pids, q_camids, g_camids, max_rank=50):
    """Evaluation with market1501 metric
        Key: for each query identity, its gallery images from the same camera view are discarded.
        """
    num_q, num_g = distmat.shape
    if num_g < max_rank:
        max_rank = num_g
        print("Note: number of gallery samples is quite small, got {}".format(num_g))
    indices = np.argsort(distmat, axis=1)
    matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)

    # compute cmc curve for each query
    all_cmc = []
    all_AP = []
    all_INP = []
    num_valid_q = 0.  # number of valid query
    for q_idx in range(num_q):
        # get query pid and camid
        q_pid = q_pids[q_idx]
        q_camid = q_camids[q_idx]

        # remove gallery samples that have the same pid and camid with query
        order = indices[q_idx]
        remove = (g_pids[order] == q_pid) & (g_camids[order] == q_camid)
        keep = np.invert(remove)

        # compute cmc curve
        # binary vector, positions with value 1 are correct matches
        orig_cmc = matches[q_idx][keep]
        if not np.any(orig_cmc):
            # this condition is true when query identity does not appear in gallery
            continue

        cmc = orig_cmc.cumsum()
        cmc[cmc > 1] = 1

        all_cmc.append(cmc[:max_rank])
        num_valid_q += 1.

        # compute average precision
        # reference: https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Average_precision
        num_rel = orig_cmc.sum()
        tmp_cmc = orig_cmc.cumsum()
        # tmp_cmc = [x / (i + 1.) for i, x in enumerate(tmp_cmc)]
        y = np.arange(1, tmp_cmc.shape[0] + 1) * 1.0
        tmp_cmc = tmp_cmc / y
        tmp_cmc = np.asarray(tmp_cmc) * orig_cmc
        AP = tmp_cmc.sum() / num_rel
        all_AP.append(AP)

    assert num_valid_q > 0, "Error: all query identities do not appear in gallery"

    all_cmc = np.asarray(all_cmc).astype(np.float32)
    all_cmc = all_cmc.sum(0) / num_valid_q
    mAP = np.mean(all_AP)

    return all_cmc, mAP


class R1_mAP():
    def __init__(self, num_query, max_rank=50, feat_norm='yes'):
        super(R1_mAP, self).__init__()
        self.num_query = num_query
        self.max_rank = max_rank
        self.feat_norm = feat_norm

    def reset(self):
        self.feats = []
        self.pids = []
        self.camids = []
        self.sceneids = []
        self.img_path = []

    def update(self, output):
        feat, pid, camid, sceneid, img_path = output
        self.feats.append(feat)
        self.pids.extend(np.asarray(pid))
        self.camids.extend(np.asarray(camid))
        self.sceneids.extend(np.asarray(sceneid.cpu()))
        self.img_path.extend(img_path)

    def compute(self,cfg):
        feats = torch.cat(self.feats, dim=0)
        if self.feat_norm == 'yes':
            print("The test feature is normalized")
            feats = torch.nn.functional.normalize(feats, dim=1, p=2)
        # query
        qf = feats[:self.num_query]
        q_pids = np.asarray(self.pids[:self.num_query])
        q_camids = np.asarray(self.camids[:self.num_query])

        q_sceneids = np.asarray(self.sceneids[:self.num_query])  # zxp
        # gallery
        gf = feats[self.num_query:]
        g_pids = np.asarray(self.pids[self.num_query:])
        g_camids = np.asarray(self.camids[self.num_query:])

        g_sceneids = np.asarray(self.sceneids[self.num_query:])  # zxp

        m, n = qf.shape[0], gf.shape[0]
        distmat = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
                  torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
        distmat.addmm_(1, -2, qf, gf.t())
        distmat = distmat.cpu().numpy()
        cmc, mAP = eval_func_msrv(distmat, q_pids, g_pids, q_camids, g_camids, q_sceneids, g_sceneids)
        return cmc, mAP, distmat, self.pids, self.camids, qf, gf


class R1_mAP_eval():
    def __init__(self, num_query, max_rank=20, feat_norm=True, reranking=False):
        super(R1_mAP_eval, self).__init__()
        self.num_query = num_query
        self.max_rank = max_rank
        self.feat_norm = feat_norm
        self.reranking = reranking

    def reset(self):
        self.feats = []
        self.pids = []
        self.camids = []

    def update(self, output):  # called once for each batch
        feat, pid, camid = output
        self.feats.append(feat.cpu())
        self.pids.extend(np.asarray(pid))
        self.camids.extend(np.asarray(camid))

    def showPointMultiModal(self,features, real_label, draw_label, save_path='/13559197192/wyh/UNIReID/pic'):
        id_show = 25
        num_ids = len(np.unique(real_label))
        save_path = os.path.join(save_path, str(draw_label) + ".pdf")
        print("Draw points of features to {}".format(save_path))
        indices = find_label_indices(real_label, draw_label, max_indices_per_label=id_show)
        feat = features[indices]
        tsne = manifold.TSNE(n_components=2, init='pca', random_state=1,learning_rate=100, perplexity=60)
        features_tsne = tsne.fit_transform(feat)
        colors =  ['#1f78b4', '#33a02c', '#e31a1c', '#ff7f00', '#6a3d9a', '#b15928', '#a6cee3', '#b2df8a', '#fb9a99', '#fdbf6f', '#cab2d6', '#ffff99']
        MARKS = ['*']
        plt.figure(figsize=(10, 10))
        for i in range(features_tsne.shape[0]):
            plt.scatter(features_tsne[i, 0], features_tsne[i, 1], s=300,color=colors[i//id_show],marker=MARKS[0],
                        alpha=0.4)
        plt.title("t-SNE Visualization of Different IDs")
        plt.xlabel("t-SNE Dimension 1")
        plt.ylabel("t-SNE Dimension 2")
        # plt.legend()
        plt.savefig(save_path)
        plt.show()
        plt.close()
        # f_R = features[indices, 0:768]
        # f_N = features[indices, 768:1536]
        # f_T = features[indices, 1536:2304]
        #
        # tsne = manifold.TSNE(n_components=2, init='pca', random_state=501,learning_rate=0.001,perplexity=30)
        # features_R_tsne = tsne.fit_transform(f_R)
        # features_N_tsne = tsne.fit_transform(f_N)
        # features_T_tsne = tsne.fit_transform(f_T)
        # COLORS = ['darkorange', 'limegreen', 'royalblue', 'red', 'darkviolet', 'black']
        # MARKS = ['*', 'o', '^']
        # features_R_min, features_R_max = features_R_tsne.min(0), features_R_tsne.max(0)
        # features_R_norm = (features_R_tsne - features_R_min) / (features_R_max - features_R_min)
        # features_N_min, features_N_max = features_N_tsne.min(0), features_N_tsne.max(0)
        # features_N_norm = (features_N_tsne - features_N_min) / (features_N_max - features_N_min)
        # features_T_min, features_T_max = features_T_tsne.min(0), features_T_tsne.max(0)
        # features_T_norm = (features_T_tsne - features_T_min) / (features_T_max - features_T_min)
        # plt.figure(figsize=(20, 20))
        # for i in range(features_R_norm.shape[0]):
        #     plt.scatter(features_R_norm[i, 0], features_R_norm[i, 1], s=300,color=COLORS[i//id_show],marker=MARKS[0],
        #                 alpha=0.4, label='RGB')
        #     plt.scatter(features_N_norm[i, 0], features_N_norm[i, 1], s=300, color=COLORS[i//id_show], marker=MARKS[1],
        #                 alpha=0.4, label='NIR')
        #     plt.scatter(features_T_norm[i, 0], features_T_norm[i, 1], s=400, color=COLORS[i//id_show], marker=MARKS[2],
        #                 alpha=0.4, label='TIR')


    def compute(self, vis=0):  # called after each epoch
        feats = torch.cat(self.feats, dim=0)
        if self.feat_norm:
            print("The test feature is normalized")
            feats = torch.nn.functional.normalize(feats, dim=1, p=2)  # along channel
        # query
        qf = feats[:self.num_query]
        q_pids = np.asarray(self.pids[:self.num_query])
        q_camids = np.asarray(self.camids[:self.num_query])
        # gallery
        gf = feats[self.num_query:]
        g_pids = np.asarray(self.pids[self.num_query:])

        g_camids = np.asarray(self.camids[self.num_query:])
        if self.reranking:
            print('=> Enter reranking')
            # distmat = re_ranking(qf, gf, k1=20, k2=6, lambda_value=0.3)
            distmat = re_ranking(qf, gf, k1=50, k2=15, lambda_value=0.3)

        else:
            print('=> Computing DistMat with euclidean_distance')
            distmat = euclidean_distance(qf, gf)
        cmc, mAP = eval_func(distmat, q_pids, g_pids, q_camids, g_camids)
        # if vis:
        #     self.showPointMultiModal(feats, real_label= self.pids, draw_label=[258,260,269,271,273,280,282,284,285,286,287,289])
        return cmc, mAP, distmat, self.pids, self.camids, qf, gf


def eval_func_regdb(distmat, q_pids, g_pids, qu_path, ga_path, epoch, max_rank=10, rank_vis=False):
    num_q, num_g = distmat.shape
    if num_g < max_rank:
        max_rank = num_g
        print("Note: number of gallery samples is quite small, got {}".format(num_g))
    indices = np.argsort(distmat, axis=1)
    if rank_vis:
        end_index = 10
        for index in range(0, 10):
            output_file_path = f"/13559197192/wyh/UNIReID/RegDB/n2r_{epoch}.txt"  # 指定输出文件路径
            rank_show = indices[index, :end_index]
            with open(output_file_path, "a") as output_file:
                output_file.write('query:' + qu_path[index].replace("_v_", "_t_") + "\n")
                for idx in rank_show:
                    image_name = ga_path[idx]
                    output_file.write(image_name + "\n")  # 写入文件名并换行
        # for index in range(0, 10, 2):
        #     output_file_path = f"/13559197192/wyh/UNIReID/RegDB/r2n.txt"  # 指定输出文件路径
        #     rank_show = indices[index, :max_rank]
        #     with open(output_file_path, "a") as output_file:
        #         output_file.write('query:' + qu_path[index] + "\n")
        #         for idx in rank_show:
        #             if output_file_path.split('/')[-1].split('.')[0][0] == 'r':
        #                 image_name = ga_path[idx].replace("_v_", "_t_")
        #             else:
        #                 image_name = ga_path[idx]
        #             output_file.write(image_name + "\n")  # 写入文件名并换行
    matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)

    # compute cmc curve for each query
    all_cmc = []
    all_AP = []
    all_INP = []
    num_valid_q = 0.  # number of valid query

    # only two cameras
    q_camids = np.ones(num_q).astype(np.int32)
    g_camids = 2 * np.ones(num_g).astype(np.int32)

    for q_idx in range(num_q):
        # get query pid and camid
        q_pid = q_pids[q_idx]
        q_camid = q_camids[q_idx]

        # remove gallery samples that have the same pid and camid with query
        order = indices[q_idx]
        remove = (g_pids[order] == q_pid) & (g_camids[order] == q_camid)
        keep = np.invert(remove)

        # compute cmc curve
        raw_cmc = matches[q_idx][keep]  # binary vector, positions with value 1 are correct matches
        if not np.any(raw_cmc):
            # this condition is true when query identity does not appear in gallery
            continue

        cmc = raw_cmc.cumsum()

        # compute mINP
        # refernece Deep Learning for Person Re-identification: A Survey and Outlook
        pos_idx = np.where(raw_cmc == 1)
        pos_max_idx = np.max(pos_idx)
        inp = cmc[pos_max_idx] / (pos_max_idx + 1.0)
        all_INP.append(inp)

        cmc[cmc > 1] = 1

        all_cmc.append(cmc[:max_rank])
        num_valid_q += 1.

        # compute average precision
        # reference: https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Average_precision
        num_rel = raw_cmc.sum()
        tmp_cmc = raw_cmc.cumsum()
        tmp_cmc = [x / (i + 1.) for i, x in enumerate(tmp_cmc)]
        tmp_cmc = np.asarray(tmp_cmc) * raw_cmc
        AP = tmp_cmc.sum() / num_rel
        all_AP.append(AP)
    assert num_valid_q > 0, "Error: all query identities do not appear in gallery"

    all_cmc = np.asarray(all_cmc).astype(np.float32)
    all_cmc = all_cmc.sum(0) / num_valid_q
    mAP = np.mean(all_AP)
    return all_cmc, mAP


class Cross_R1_mAP_eval():
    def __init__(self, num_query, max_rank=50, feat_norm=True, reranking=False):
        super(Cross_R1_mAP_eval, self).__init__()
        self.num_query = num_query
        self.max_rank = max_rank
        self.feat_norm = feat_norm
        self.reranking = reranking

    def reset(self):
        self.query_feats = []
        self.query_pids = []
        self.query_camids = []
        self.query_imgpath = []
        self.gallery_feats = []
        self.gallery_pids = []
        self.gallery_camids = []
        self.gallery_imgpath = []

    def update_query(self, output):  # called once for each batch
        feat, pid, camid, imgpath = output
        self.query_feats.append(feat.cpu())
        self.query_pids.extend(np.asarray(pid))
        self.query_camids.extend(np.asarray(camid))
        self.query_imgpath.extend(imgpath)

    def update_gallery(self, output):  # called once for each batch
        feat, pid, camid, imgpath = output
        self.gallery_feats.append(feat.cpu())
        self.gallery_pids.extend(np.asarray(pid))
        self.gallery_camids.extend(np.asarray(camid))
        self.gallery_imgpath.extend(imgpath)

    def compute(self, cfg, epoch=0, sysu_mode='all'):  # called after each epoch
        query_feats = torch.cat(self.query_feats, dim=0)
        gallery_feats = torch.cat(self.gallery_feats, dim=0)

        if self.feat_norm:
            print("The test feature is normalized")
            query_feats = torch.nn.functional.normalize(query_feats, dim=1, p=2)  # along channel
            gallery_feats = torch.nn.functional.normalize(gallery_feats, dim=1, p=2)  # along channel
        # query
        qf = query_feats[:self.num_query]
        q_pids = np.asarray(self.query_pids[:self.num_query])
        q_camids = np.asarray(self.query_camids[:self.num_query])

        # gallery
        gf = gallery_feats[:self.num_query]
        g_pids = np.asarray(self.gallery_pids[:self.num_query])
        g_camids = np.asarray(self.gallery_camids[:self.num_query])
        if cfg.DATASETS.NAMES == 'SYSU':
            if sysu_mode != 'all':
                condition = g_camids in [0, 1]
                gf = gf[condition]
                g_pids = g_pids[condition]
                g_camids = g_camids[condition]

        if cfg.DATASETS.NAMES == 'RGBNT201':
            condition_q = q_camids == 0
            condition_g = g_camids == 1
            qf = qf[condition_q]
            gf = gf[condition_g]
            q_pids = q_pids[condition_q]
            g_pids = g_pids[condition_g]
            q_camids = q_camids[condition_q]
            g_camids = g_camids[condition_g]
            print('RGBNT201 Cross Here!!!')

        if self.reranking:
            print('=> Enter reranking')
            # distmat = re_ranking(qf, gf, k1=20, k2=6, lambda_value=0.3)
            distmat = re_ranking(qf, gf, k1=50, k2=15, lambda_value=0.3)
            distmat2 = re_ranking(gf, qf, k1=50, k2=15, lambda_value=0.3)
        else:
            print('=> Computing DistMat with euclidean_distance')
            distmat = euclidean_distance(qf, gf)
            distmat2 = euclidean_distance(gf, qf)
        if cfg.DATASETS.NAMES == 'RegDB':
            cmc, mAP = eval_func_regdb(distmat, q_pids, g_pids, qu_path=self.query_imgpath,
                                       ga_path=self.gallery_imgpath, epoch=epoch)
            cmc2, mAP2 = eval_func_regdb(distmat2, g_pids, q_pids, qu_path=self.gallery_imgpath,
                                         ga_path=self.query_imgpath, epoch=epoch)
            # visualize_tsne(qf, gf, q_pids, cfg.OUTPUT_DIR, epoch=epoch)
        elif cfg.DATASETS.NAMES == 'SYSU':
            cmc, mAP = eval_func_sysu(distmat, q_pids, g_pids, q_camids, g_camids)
        else:
            # cmc, mAP = eval_func_regdb(distmat, q_pids, g_pids, qu_path=self.query_imgpath,
            #                            ga_path=self.gallery_imgpath, epoch=epoch)
            # cmc2, mAP2 = eval_func_regdb(distmat2, g_pids, q_pids, qu_path=self.gallery_imgpath,  # gallery as query
            #                              ga_path=self.query_imgpath, epoch=epoch)
            # visualize_tsne(qf, gf, q_pids, cfg.OUTPUT_DIR, epoch=epoch)
            cmc, mAP = eval_func(distmat, q_pids, g_pids, q_camids, g_camids)
            cmc2, mAP2 = eval_func(distmat2, g_pids, q_pids, g_camids, q_camids)

        return cmc, mAP, cmc2, mAP2


def find_label_indices(label_list, target_labels, max_indices_per_label=1):
    indices = []
    counts = {label: 0 for label in target_labels}
    for index, label in enumerate(label_list):
        if label in target_labels and counts[label] < max_indices_per_label:
            indices.append(index)
            counts[label] += 1
    sorted_indices = sorted(indices, key=lambda index: (label_list[index], index))
    return sorted_indices

#
# def visualize_tsne(rgb_features, tir_features, pids, save_path, epoch):
#     """
#     可视化两类特征的分布
#     参数：
#     - rgb_features: RGB特征的numpy数组，维度为(batch_size, D)
#     - tir_features: TIR特征的numpy数组，维度为(batch_size, D)
#     - ids: 特征对应的ID，长度为batch_size
#     """
#     ids = pids[44:56]
#     id_show = 10  # 每个ID最多显示的特征点数
#     if not os.path.exists(save_path):
#         # 如果路径不存在，创建它
#         os.makedirs(save_path)
#     print("Draw points of features to {}".format(save_path))
#     indices = find_label_indices(pids, ids, max_indices_per_label=id_show)
#     # 将RGB和TIR特征合并成一个特征矩阵
#     rgb_features = rgb_features[indices, :]
#     tir_features = tir_features[indices, :]
#     assert rgb_features.shape[0] == int(id_show * len(ids))  # 每个ID最多显示id_show个特征点
#     combined_features = np.vstack((rgb_features, tir_features))
#
#     # 使用t-SNE进行降维
#     tsne = manifold.TSNE(n_components=2, random_state=42)
#     tsne_features = tsne.fit_transform(combined_features)
#     features_min, features_max = tsne_features.min(0), tsne_features.max(0)
#     features_norm = (tsne_features - features_min) / (features_max - features_min)
#     tsne_features = features_norm
#     COLORS = ['darkorange', 'limegreen', 'royalblue', 'red', 'darkviolet', 'black', 'gold', 'green', 'blue', 'purple',
#               'pink', 'gray',
#               'darkorange', 'limegreen', 'royalblue', 'red', 'darkviolet', 'black', 'gold', 'green', 'blue', 'purple',
#               'pink', 'gray']
#     MARKS = ['*', 'o']
#     # 创建一个新的图形
#     plt.figure(figsize=(10, 10))
#
#     # 绘制RGB特征的散点图
#     for i in range(tsne_features.shape[0]):
#         plt.scatter(tsne_features[i, 0], tsne_features[i, 1], c=COLORS[i // id_show],
#                     marker=MARKS[i // int(id_show * len(ids))])
#
#     # 添加标题和轴标签
#     plt.title('t-SNE Visualization of Features')
#     plt.xlabel('t-SNE Dimension 1')
#     plt.ylabel('t-SNE Dimension 2')
#
#     # 保存图像
#     plt.savefig(save_path + '/' + f'epoch_{epoch}__' + '_'.join(map(str, ids)) + '.pdf')
#     plt.savefig(save_path + '/' + f'epoch_{epoch}__' + '_'.join(map(str, ids)) + '.jpg')
#     # 显示图像
#     plt.show()

#
# def visualize_tsne_3d(rgb_features, tir_features, pids, save_path, epoch):
#     ids = pids[44:50]
#     id_show = 10  # Maximum number of points to show per ID
#     if not os.path.exists(save_path):
#         os.makedirs(save_path)
#     print("Drawing 3D t-SNE plot and saving it to {}".format(save_path))
#     indices = find_label_indices(pids, ids, max_indices_per_label=id_show)
#
#     # Combine RGB and TIR features into one feature matrix
#     rgb_features = rgb_features[indices, :]
#     tir_features = tir_features[indices, :]
#     assert rgb_features.shape[0] == int(id_show * len(ids))
#
#     combined_features = np.vstack((rgb_features, tir_features))
#
#     # Use t-SNE to reduce dimensionality to 3D
#     tsne = manifold.TSNE(n_components=3, random_state=42)
#     tsne_features = tsne.fit_transform(combined_features)
#
#     features_min, features_max = tsne_features.min(0), tsne_features.max(0)
#     features_norm = (tsne_features - features_min) / (features_max - features_min)
#     tsne_features = features_norm
#
#     COLORS = ['darkorange', 'limegreen', 'royalblue', 'red', 'darkviolet', 'black',
#               'darkorange', 'limegreen', 'royalblue', 'red', 'darkviolet', 'black']
#     MARKS = ['*', 'o']
#
#     # Create a new 3D figure
#     fig = plt.figure(figsize=(12, 8))
#     ax = fig.add_subplot(111, projection='3d')
#
#     # Plot RGB features in 3D
#     for i in range(tsne_features.shape[0]):
#         ax.scatter(tsne_features[i, 0], tsne_features[i, 1], tsne_features[i, 2],
#                    c=COLORS[i // id_show], marker=MARKS[i // int(id_show * len(ids))])
#
#     # Add title and labels
#     ax.set_title('3D t-SNE Visualization of Features')
#     ax.set_xlabel('t-SNE Dimension 1')
#     ax.set_ylabel('t-SNE Dimension 2')
#     ax.set_zlabel('t-SNE Dimension 3')
#
#     # Save the 3D plot as a PDF and JPG
#     plt.savefig(save_path + '/' + f'epoch_{epoch}__' + '_'.join(map(str, ids)) + '_3D.pdf')
#     plt.savefig(save_path + '/' + f'epoch_{epoch}__' + '_'.join(map(str, ids)) + '_3D.jpg')
#
#     # Show the 3D plot
#     plt.show()
