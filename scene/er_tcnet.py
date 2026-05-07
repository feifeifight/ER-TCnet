import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def index_points(points, idx):
    batch_size = points.shape[0]
    view_shape = [batch_size] + [1] * (idx.ndim - 1)
    batch_idx = torch.arange(batch_size, device=points.device).view(*view_shape).expand_as(idx)
    return points[batch_idx, idx]


def farthest_point_sample(xyz, npoint):
    device = xyz.device
    batch_size, num_points, _ = xyz.shape
    npoint = min(int(npoint), int(num_points))

    xyz_sq = (xyz ** 2).sum(dim=-1)
    centroids = torch.zeros(batch_size, npoint, dtype=torch.long, device=device)
    distance = torch.full((batch_size, num_points), 1e10, device=device)
    farthest = torch.randint(0, num_points, (batch_size,), dtype=torch.long, device=device)
    batch_indices = torch.arange(batch_size, dtype=torch.long, device=device)

    for i in range(npoint):
        centroids[:, i] = farthest
        centroid_sq = xyz_sq[batch_indices, farthest]
        centroid_dot = torch.einsum('bd,bnd->bn',
                                     xyz[batch_indices, farthest], xyz)
        dist = centroid_sq.unsqueeze(1) - 2 * centroid_dot + xyz_sq
        torch.minimum(distance, dist, out=distance)
        farthest = torch.max(distance, dim=-1).indices

    return centroids


def random_sample(xyz, npoint):
    batch_size, num_points, _ = xyz.shape
    npoint = min(int(npoint), int(num_points))
    # Vectorised: generate all random indices in one op
    noise = torch.rand(batch_size, num_points, device=xyz.device)
    return noise.topk(npoint, dim=-1).indices


def compute_inflation_index(num_neighbors, n=10, m=1):
    safe_neighbors = torch.clamp(num_neighbors.float(), max=20.0)
    d_i = (float(n) / (1.0 + torch.exp(safe_neighbors))).long()
    d_i = torch.clamp(d_i * int(m), min=1, max=int(n))
    return d_i


class PointCloudHead(nn.Module):
    def __init__(self, in_dim=128, n_points=256, hidden_dim=256, feat_dim=64):
        super().__init__()
        self.n_points = n_points
        self.feat_dim = feat_dim

        self.mlp_xyz = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, n_points * 3),
        )
        self.mlp_feat = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, n_points * feat_dim),
        )

    def forward(self, f_global):
        batch_size = f_global.shape[0]
        xyz = self.mlp_xyz(f_global).view(batch_size, self.n_points, 3)
        feat = self.mlp_feat(f_global).view(batch_size, self.n_points, self.feat_dim)
        xyz = torch.tanh(xyz)
        return xyz, feat


class DGRB(nn.Module):
    def __init__(
        self,
        K=64,
        radius=0.1,
        n=10,
        m=1,
        k=8,
        in_dim=64,
        out_dim=128,
        fast_sampling=True,
    ):
        super().__init__()
        self.K = K
        self.radius = radius
        self.n = n
        self.m = m
        self.k = k
        self.max_neighbors = max(int(k * n), 1)
        self.fast_sampling = fast_sampling

        self.conv_local = nn.Sequential(
            nn.Conv2d(in_dim, 64, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, out_dim, 1),
        )
        self.pos_mlp = nn.Sequential(
            nn.Conv1d(3, out_dim, 1),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(inplace=True),
        )

    def density_aware_grouping(self, P0, F_p0, centers):
        if F_p0 is None:
            F_p0 = P0

        max_neighbors = min(self.max_neighbors, P0.shape[1])

        # Single distance computation — used for both KNN and radius check
        dist = torch.cdist(centers, P0)  # [B, K, N]

        knn_idx = torch.topk(dist, k=max_neighbors, dim=-1, largest=False).indices
        grouped_feat = index_points(F_p0, knn_idx)

        # Reuse the same dist matrix for radius check
        num_neighbors = (dist <= self.radius).sum(dim=-1)
        d_i = compute_inflation_index(num_neighbors, n=self.n, m=self.m)
        target_len = torch.clamp(d_i * self.k, min=1, max=max_neighbors)

        order = torch.arange(max_neighbors, device=P0.device).view(1, 1, max_neighbors)
        mask = (order < target_len.unsqueeze(-1)).to(grouped_feat.dtype)
        grouped_feat = grouped_feat * mask.unsqueeze(-1)

        return grouped_feat, mask

    def forward(self, P0, F_p0=None):
        if self.fast_sampling and self.training:
            idx = random_sample(P0, self.K)
        else:
            idx = farthest_point_sample(P0, self.K)
        centers = index_points(P0, idx)

        grouped_feat, mask = self.density_aware_grouping(P0, F_p0, centers)

        x = grouped_feat.permute(0, 3, 2, 1).contiguous()
        x = self.conv_local(x)
        x = x * mask.permute(0, 2, 1).unsqueeze(1)
        x = torch.max(x, dim=2).values.permute(0, 2, 1).contiguous()

        pos = self.pos_mlp(centers.permute(0, 2, 1)).permute(0, 2, 1).contiguous()
        F_embed = x + pos
        return F_embed, centers, idx


class CCM(nn.Module):
    def __init__(self, img_dim=128, cap_dim=64):
        super().__init__()
        self.W_j = nn.Linear(cap_dim, 1)
        self.W_i = nn.Linear(img_dim, 1)
        self.b_p = nn.Parameter(torch.zeros(1))
        self.img_proj = nn.Linear(img_dim, cap_dim) if img_dim != cap_dim else nn.Identity()

    def forward(self, F_image, v_p):
        gate = torch.sigmoid(self.W_j(v_p) + self.W_i(F_image) + self.b_p)
        F_image_proj = self.img_proj(F_image)
        U_c = gate * F_image_proj + (1.0 - gate) * v_p
        return U_c


class SelfAttnRouting(nn.Module):
    def __init__(self, cap_dim=64, num_heads=4):
        super().__init__()
        self.attn = nn.MultiheadAttention(cap_dim, num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(cap_dim)
        self.norm2 = nn.LayerNorm(cap_dim)
        self.mlp = nn.Sequential(
            nn.Linear(cap_dim, cap_dim * 2),
            nn.ReLU(inplace=True),
            nn.Linear(cap_dim * 2, cap_dim),
        )

    def forward(self, U_c):
        attn_out, _ = self.attn(U_c, U_c, U_c)
        v_c = self.norm1(U_c + attn_out)
        v_c = v_c + self.mlp(self.norm2(v_c))
        return v_c


class OffsetAttention(nn.Module):
    def __init__(self, q_dim, k_dim, v_dim, out_dim, num_heads=4):
        super().__init__()
        if out_dim % num_heads != 0:
            raise ValueError("out_dim must be divisible by num_heads")

        self.num_heads = num_heads
        self.head_dim = out_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.to_q = nn.Linear(q_dim, out_dim)
        self.to_k = nn.Linear(k_dim, out_dim)
        self.to_v = nn.Linear(v_dim, out_dim)

        self.lbr_linear = nn.Linear(out_dim, out_dim)
        self.lbr_bn = nn.BatchNorm1d(out_dim)
        self.lbr_act = nn.ReLU(inplace=True)

        self.norm1 = nn.LayerNorm(out_dim)
        self.norm2 = nn.LayerNorm(out_dim)
        self.mlp = nn.Sequential(
            nn.Linear(out_dim, out_dim * 2),
            nn.ReLU(inplace=True),
            nn.Linear(out_dim * 2, out_dim),
        )

    def forward(self, F_image, v_c, F_embed):
        batch_size, num_caps, _ = F_image.shape

        q = self.to_q(F_image).view(batch_size, num_caps, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.to_k(v_c).view(batch_size, num_caps, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.to_v(F_embed).view(batch_size, num_caps, self.num_heads, self.head_dim).transpose(1, 2)

        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        A = torch.softmax(attn, dim=-1)

        attn_out = torch.matmul(A, v)
        offset = v - attn_out

        offset = offset.transpose(1, 2).reshape(batch_size, num_caps, -1)
        offset = self.lbr_linear(offset)
        offset = self.lbr_bn(offset.reshape(batch_size * num_caps, -1)).reshape(batch_size, num_caps, -1)
        offset = self.lbr_act(offset)

        v_flat = v.transpose(1, 2).reshape(batch_size, num_caps, -1)
        F_oa = self.norm1(offset + v_flat)
        F_pc = F_oa + self.mlp(self.norm2(F_oa))
        return F_pc


class ERRouting(nn.Module):
    def __init__(self, pose_dim=64, lambda_reg=0.05, n_iter=3, delta=1e-5):
        super().__init__()
        self.pose_dim = pose_dim
        self.lambda_reg = lambda_reg
        self.n_iter = n_iter
        self.delta = delta
        self.activation_conv = nn.Conv1d(pose_dim, 1, 1)
        for param in self.activation_conv.parameters():
            param.requires_grad = False

    def forward(self, Q, P_init):
        _, _, pose_dim = Q.shape
        logits = torch.bmm(Q, P_init.transpose(1, 2)) / math.sqrt(float(pose_dim))
        C = torch.softmax(logits, dim=1)

        for _ in range(max(self.n_iter - 1, 0)):
            entropy_grad = torch.log(C.clamp_min(self.delta))
            C = torch.softmax(logits - self.lambda_reg * entropy_grad, dim=1)

        P_hat = torch.bmm(C.transpose(1, 2), Q)
        a_L = torch.sigmoid(self.activation_conv(P_hat.permute(0, 2, 1)).squeeze(1))
        return P_hat, C, a_L


class ERTCNet(nn.Module):
    def __init__(
        self,
        img_global_dim,
        n_points=256,
        sample_points=64,
        point_hidden_dim=256,
        point_feat_dim=64,
        dgrb_radius=0.1,
        dgrb_n=10,
        dgrb_m=1,
        dgrb_k=8,
        embed_dim=128,
        cap_dim=64,
        num_heads=4,
        lambda_reg=0.05,
        n_iter=3,
        fast_sampling=True,
    ):
        super().__init__()
        self.point_head = PointCloudHead(
            in_dim=img_global_dim,
            n_points=n_points,
            hidden_dim=point_hidden_dim,
            feat_dim=point_feat_dim,
        )
        self.dgrb = DGRB(
            K=sample_points,
            radius=dgrb_radius,
            n=dgrb_n,
            m=dgrb_m,
            k=dgrb_k,
            in_dim=point_feat_dim,
            out_dim=embed_dim,
            fast_sampling=fast_sampling,
        )

        self.image_token_proj = nn.Linear(img_global_dim, embed_dim)
        self.primary_cap_proj = nn.Linear(embed_dim, cap_dim)

        self.ccm = CCM(img_dim=embed_dim, cap_dim=cap_dim)
        self.self_attn_routing = SelfAttnRouting(cap_dim=cap_dim, num_heads=num_heads)
        self.offset_attention = OffsetAttention(
            q_dim=embed_dim,
            k_dim=cap_dim,
            v_dim=embed_dim,
            out_dim=embed_dim,
            num_heads=num_heads,
        )

        self.pose_proj = nn.Linear(embed_dim, cap_dim)
        self.er_routing = ERRouting(
            pose_dim=cap_dim,
            lambda_reg=lambda_reg,
            n_iter=n_iter,
        )

    def forward(self, f_global):
        P0, F_p0 = self.point_head(f_global)
        F_embed, centers, center_idx = self.dgrb(P0, F_p0)

        batch_size, num_caps, _ = F_embed.shape
        F_image = self.image_token_proj(f_global).unsqueeze(1).expand(batch_size, num_caps, -1)

        v_p = self.primary_cap_proj(F_embed)
        U_c = self.ccm(F_image, v_p)
        v_c = self.self_attn_routing(U_c)

        F_pc = self.offset_attention(F_image, v_c, F_embed)
        P_hat, C, a_L = self.er_routing(v_c, self.pose_proj(F_pc))

        return {
            "P0": P0,
            "F_p0": F_p0,
            "centers": centers,
            "center_idx": center_idx,
            "F_embed": F_embed,
            "F_image": F_image,
            "v_p": v_p,
            "U_c": U_c,
            "v_c": v_c,
            "F_pc": F_pc,
            "P_hat": P_hat,
            "C": C,
            "a_L": a_L,
        }


def entropy_regularization_loss(C, delta=1e-5):
    entropy = -(C * torch.log(C.clamp_min(delta))).sum(dim=-1).mean()
    return entropy


def point_cloud_regularization_loss(P0):
    return P0.pow(2).mean()
