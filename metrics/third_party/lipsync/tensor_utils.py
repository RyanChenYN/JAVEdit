import torch


def img2tensor(img, device, non_blocking=False):
    """BGR numpy (H, W, 3) -> float tensor (1, 3, H, W) in [-1, 1]"""
    data = (torch.from_numpy(img[:, :, ::-1].copy())
            .to(device, non_blocking=non_blocking)
            .permute(2, 0, 1).unsqueeze(0).float() / 127.5 - 1.0)
    return data


def tensor2imgs(tensor):
    """float tensor (B, 3, H, W) in [-1, 1] -> list of BGR numpy (H, W, 3)"""
    data = ((tensor.detach() + 1.0) * 127.5 + 0.5).clamp(0, 255).type(torch.uint8).permute(0, 2, 3, 1)
    data = data.cpu().numpy()[:, :, :, ::-1].copy()
    return [data[i, :] for i in range(data.shape[0])]


def calc_pdist_cos(feat1, feat2, vshift=10):
    """计算偏移窗口内的余弦距离，返回 similarities list"""
    win_size = vshift * 2 + 1
    feat2p = torch.nn.functional.pad(feat2, (0, 0, vshift, vshift))

    similarities = []
    for i in range(len(feat1)):
        similarity = torch.nn.functional.cosine_similarity(
            feat1[[i], :], feat2p[i: i + win_size, :], dim=1)
        similarity[torch.isnan(similarity)] = 0.
        similarities.append(1 - similarity)

    return similarities
