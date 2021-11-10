from torch import nn
import torch

PI = torch.tensor(3.141592653589793238462643383279502884197169, dtype=torch.float32)
class IMSE(nn.Module):
    def __init__(self, neighbor_k=3, kernel_size=1, sigma=0.8):
        super(IMSE, self).__init__()
        self.neighbor_k = neighbor_k
        self.kernel_size = kernel_size
        self.sigma = sigma

    def get_gauss_kernel(self, device):
        row = 2 * self.kernel_size + 1
        col = 2 * self.kernel_size + 1
        A = []
        sigma = torch.tensor(self.sigma, dtype=torch.float64)
        for i in range(row):
            r = []
            for j in range(col):
                fenzi = (i + 1 - self.kernel_size - 1) ** 2 + (j + 1 - self.kernel_size - 1) ** 2
                r.append(torch.exp(-fenzi / (2 * sigma)) / (2 * PI * sigma))
            A.append(torch.stack(r, 0))
        A = torch.stack(A, 0)
        A = A / A.sum()
        gauss_kernel = A.view(1, 1, 1, 2 * self.kernel_size + 1, 2 * self.kernel_size + 1).type(torch.float32).to(device)
        return gauss_kernel

    def cal_cosinesimilarity(self, query_x, support_x):
        self.gauss_kernel = self.get_gauss_kernel(query_x.device)
        Similarity_list = []
        query_x = query_x.permute(0, 2, 3, 1)
        qm, qh, qw, C = query_x.size()
        q_feature_size = qh * qw
        query_x = torch.reshape(query_x, [qm, q_feature_size, C])
        cov_list = []
        for i in range(len(support_x)):
            prototype_x = support_x[i].permute(0, 2, 3, 1)
            pm, ph, pw, C = prototype_x.size()
            p_feature_size = ph * pw
            prototype_x = torch.reshape(prototype_x, [pm, p_feature_size, C])
            # get the cosine similarity matrix
            res_q2s = self.get_lds(prototype_x, query_x)
            # get sparse similarity matrix
            q2s = self.compose_ldV2(res_q2s).permute(0, 1, 3, 2)
            res = torch.reshape(q2s, [qm*pm, 1, qh * qw, ph, pw])
            # do Guassian filter and max-polling 2 times
            for i in range(2):
                res = torch.nn.functional.conv3d(res, self.gauss_kernel.detach(), stride=[1, 1, 1],
                                                 padding=0)
                res = torch.nn.functional.max_pool3d(res, kernel_size=[1, 2, 2],
                                                     stride=[1, 2, 2])
            cov_list.append(res.view(-1, ph*pw))
            score_vec = res.view(qm, -1).sum(-1)
            Similarity_list.append(score_vec)
        Similarity_list = torch.stack(Similarity_list, 0).t()
        return {"logits": Similarity_list, "cov_list":cov_list}

    def get_cov(self, x):
        mean = x.mean(0)
        va = x - mean.expand_as(x)
        cov = torch.matmul(va.t(), va)/(x.size(0)-1)
        return cov

    def get_lds(self, p, q):
        innerproduct = torch.einsum('ijk,spk->isjp', q, p)
        q2 = torch.sqrt(torch.einsum('ijk,ijk->ij', q, q))
        p2 = torch.sqrt(torch.einsum('ijk,ijk->ij', p, p))
        q2p2 = torch.einsum('ij,sp->isjp', q2, p2)
        res = innerproduct / q2p2
        return res

    def compose_ld(self, qlds):
        qm, pm, q_feature_size, p_feature_size = qlds.size()
        qlds = qlds.permute(0, 2, 1, 3)
        qlds = qlds.reshape([qm, q_feature_size, -1])
        q2s_topk, q2s_idx = torch.topk(qlds, k=self.neighbor_k,  dim=-1)
        q2s_idx, idxs = self.handel_topidx(q2s_idx, qm, pm, q_feature_size, self.neighbor_k)
        q2s_topk = torch.squeeze(torch.reshape(q2s_topk, [1, -1])).cuda()
        q2s = torch.sparse.FloatTensor(q2s_idx.t(), q2s_topk,
                                       torch.Size((qm, q_feature_size, pm * p_feature_size))).to_dense()
        q2s = q2s.reshape([qm, q_feature_size, pm, p_feature_size])
        q2s = q2s.permute(0, 2, 1, 3)

        return q2s


    def compose_ldV2(self, qlds):
        qm, pm, q_feature_size, p_feature_size = qlds.size()
        qlds = qlds.view(-1, p_feature_size)
        # select the top-k similar points in support spatial-wise (HXW)
        q2s_topk, q2s_idx = torch.topk(qlds, k=self.neighbor_k,  dim=-1, sorted=False)
        q2s_sparse = torch.zeros_like(qlds)
        # set the no top-k similar points to 0
        q2s_sparse = q2s_sparse.scatter(-1, q2s_idx, q2s_topk).view(qm, pm, q_feature_size, p_feature_size)
        return q2s_sparse

    def handel_topidx(self, idxs, qm, pm, m, k):
        all_num = qm * m * k
        idxs = torch.reshape(idxs, [qm, m, k])
        idxs = torch.reshape(idxs, [qm * m * k])

        q_idx = torch.tensor(list(range(qm)), dtype=torch.int64).cuda()
        q_idx = torch.reshape(q_idx, [qm, 1])
        q_idx = torch.squeeze(torch.reshape(q_idx.repeat([1, int(all_num / qm)]), [1, all_num]))

        m_idx = torch.tensor(list(range(m)), dtype=torch.int64).cuda()
        m_idx = torch.reshape(m_idx, [m, 1])
        m_idx = m_idx.repeat([1, k])
        m_idx = torch.reshape(m_idx, [1, int(m * k)])
        m_idx = m_idx.repeat([1, qm])
        m_idx = torch.squeeze(torch.reshape(m_idx, [all_num, 1]))
        sparse_idx = torch.reshape(torch.transpose(torch.stack([q_idx, m_idx, idxs], dim=0), dim0=1, dim1=0),
                                   [all_num, -1])
        return sparse_idx, idxs

    def forward(self, x1, x2):
        Similarity_list = self.cal_cosinesimilarity(x1, x2)
        return Similarity_list





















