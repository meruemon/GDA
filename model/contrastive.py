import torch.nn as nn
import torch.nn.functional as F
import torch


def cdist(x1, x2):
    x1_norm = x1.pow(2).sum(dim=-1, keepdim=True)
    x2_norm = x2.pow(2).sum(dim=-1, keepdim=True)
    res = torch.addmm(x2_norm.transpose(-2, -1), x1, x2.transpose(-2, -1), alpha=-2).add_(x1_norm)
    res = res.clamp_min_(1e-30).sqrt_()
    return res


class MSCLoss(nn.Module):
    def __init__(self, config_data):
        super().__init__()
        self.ranking_method = 'sim_ratio'
        self.ranking_k = config_data['ranking_k']
        self.top_ranked_n = config_data['top_ranked_n']
        self.eps = 1e-9
        self.similarity_func = config_data['similarity_func']  # euclidean dist, cosine
        self.top_n_sim = config_data['top_n_sim']  # k for knn
        self.knn_method = config_data['knn_method']  # if 'ranking' use filtering, if 'classic' no filtering
        self.batch_size = config_data["batch_size"]
        self.conf_ind = None
        self.tau = config_data['tau']


    def __get_sim_matrix(self, out_src, out_tar):
        matrix = None
        if (self.similarity_func == 'euclidean'):  ## Inverse Euclidean Distance
            matrix = cdist(out_src, out_tar)
            matrix = matrix + 1.0
            matrix = 1.0 / matrix

        elif (self.similarity_func == 'gaussian'):  ## exponential Gaussian Distance
            matrix = cdist(out_src, out_tar)
            matrix = torch.exp(-1 * matrix)

        elif (self.similarity_func == 'cosine'):  ## Cosine Similarity
            out_src = F.normalize(out_src, dim=1, p=2)
            out_tar = F.normalize(out_tar, dim=1, p=2)
            matrix = torch.matmul(out_src, out_tar.T)

        else:
            raise NotImplementedError

        return matrix

    ## assign target labels by KNN
    def assign_labels_KNN(self, sim_matrix, src_labels, tgt_labels):

        ind = torch.sort(sim_matrix, descending=True, dim=0).indices
        k_orderedNeighbors = src_labels[ind[:self.top_n_sim]]
        assigned_target_labels = torch.mode(k_orderedNeighbors, dim=0).values

        correct = torch.where(assigned_target_labels == tgt_labels, 1, 0)
        num_correct = torch.count_nonzero(correct)
        num_correct = num_correct.item()

        return assigned_target_labels, ind, num_correct

    def calc_loss(self, confident_sim_matrix, src_labels, confident_tgt_labels):
        n_src = src_labels.shape[0]
        n_tgt = confident_tgt_labels.shape[0]

        vr_src = src_labels.unsqueeze(-1).repeat(1, n_tgt)
        hr_tgt = confident_tgt_labels.unsqueeze(-2).repeat(n_src, 1)

        mask_sim = (vr_src == hr_tgt).float()

        expScores = torch.softmax(confident_sim_matrix / self.tau, dim=0)
        contrastiveMatrix = (expScores * mask_sim).sum(0) / (expScores.sum(0))
        MSC_loss = -1 * torch.mean(torch.log(contrastiveMatrix + 1e-6))

        return MSC_loss

    # def forward(self, source_features, source_labels, target_features, target_labels):
    def forward(self, source_features, source_labels, target_features, target_features_0, target_labels):

        n_tgt = len(target_features)

        sim_matrix = self.__get_sim_matrix(source_features, target_features)
        sim_matrix_0 = self.__get_sim_matrix(source_features, target_features_0)
        flat_src_labels = source_labels.squeeze()

        assigned_tgt_labels, sorted_indices, num_correct = self.assign_labels_KNN(sim_matrix, source_labels, target_labels)

        ranking_score_list = []

        for i in range(0, n_tgt):  # nln: nearest like neighbours, nun: nearest unlike neighbours
            nln_mask = (flat_src_labels == assigned_tgt_labels[i]).float()
            sorted_nln_mask = nln_mask[sorted_indices[:, i]].bool()
            # nln_sim_r = sim_matrix[:, i][sorted_indices[:, i][sorted_nln_mask]][:self.ranking_k]
            nln_sim_r = sim_matrix_0[:, i][sorted_indices[:, i][sorted_nln_mask]][:self.ranking_k]

            nun_mask = ~(flat_src_labels == assigned_tgt_labels[i])
            nun_mask = nun_mask.float()
            sorted_nun_mask = nun_mask[sorted_indices[:, i]].bool()
            # nun_sim_r = sim_matrix[:, i][sorted_indices[:, i][sorted_nun_mask]][:self.ranking_k]
            nun_sim_r = sim_matrix_0[:, i][sorted_indices[:, i][sorted_nun_mask]][:self.ranking_k]

            pred_conf_score = (
                        1.0 * torch.sum(nln_sim_r) / torch.sum(nun_sim_r)).detach()  # sim ratio : confidence score
            ranking_score_list.append(pred_conf_score)

        top_n_tgt_ind = torch.topk(torch.tensor(ranking_score_list), self.top_ranked_n)[1]
        # confident_sim_matrix = sim_matrix[:, top_n_tgt_ind]
        confident_sim_matrix = sim_matrix_0[:, top_n_tgt_ind]
        confident_tgt_labels = assigned_tgt_labels[top_n_tgt_ind]  # filtered tgt labels
        loss_targetAnch = self.calc_loss(confident_sim_matrix, source_labels, confident_tgt_labels)

        return loss_targetAnch, num_correct
