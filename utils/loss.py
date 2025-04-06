import torch
import torch.nn.functional as F
import torch.nn as nn


def sdf_hessian_loss(pred, coord, hessian, surface_mask, hessian_norm_scale=7e-9, hessian_loss_scale=7e-12):
    hessian_norm = torch.linalg.norm(hessian[surface_mask], ord=2, dim=(-2, -1))
    hessian_mean = hessian_norm.mean()*hessian_norm_scale

    # Step 2: Compute Laplacian (Trace of Hessian)
    laplacian = hessian[:, 0, 0] + hessian[:, 1, 1] + hessian[:, 2, 2]

    # Compute Bi-Laplacian (Fourth-order derivative)
    N = pred.shape[0]
    bi_laplacian = torch.zeros(pred.shape[0], device = coord.device)
    grid_spacing = 1.0
    
    # Limit the range to inner elements to avoid boundary errors
    inner_range = slice(0, pred.shape[0])
    
    # Second derivative in the x-direction (finite difference)
    d2_laplacian_dx2 = (laplacian[inner_range][2:] - 2 * laplacian[inner_range][1:-1] + laplacian[inner_range][:-2]) / grid_spacing**2
    
    # Second derivative in the y-direction
    d2_laplacian_dy2 = (laplacian[inner_range][2:] - 2 * laplacian[inner_range][1:-1] + laplacian[inner_range][:-2]) / grid_spacing**2
    
    # Second derivative in the z-direction
    d2_laplacian_dz2 = (laplacian[inner_range][2:] - 2 * laplacian[inner_range][1:-1] + laplacian[inner_range][:-2]) / grid_spacing**2

    # Mixed second derivatives
    d2_laplacian_dxdy = (laplacian[inner_range][2:] - laplacian[inner_range][1:-1]) / (2 * grid_spacing)
    d2_laplacian_dxdz = (laplacian[inner_range][2:] - laplacian[inner_range][1:-1]) / (2 * grid_spacing)
    d2_laplacian_dydz = (laplacian[inner_range][2:] - laplacian[inner_range][1:-1]) / (2 * grid_spacing)

    # Create tensors to hold the full output size and initialize them with zeros
    d2_laplacian_dx2_full = torch.zeros(N, device=coord.device)
    d2_laplacian_dy2_full = torch.zeros(N, device=coord.device)
    d2_laplacian_dz2_full = torch.zeros(N, device=coord.device)
    d2_laplacian_dxdy_full = torch.zeros(N, device=coord.device)
    d2_laplacian_dxdz_full = torch.zeros(N, device=coord.device)
    d2_laplacian_dydz_full = torch.zeros(N, device=coord.device)

    # Insert the computed second derivatives in the inner region (ignoring the boundaries)
    d2_laplacian_dx2_full[1:-1] = d2_laplacian_dx2
    d2_laplacian_dy2_full[1:-1] = d2_laplacian_dy2
    d2_laplacian_dz2_full[1:-1] = d2_laplacian_dz2
    d2_laplacian_dxdy_full[1:-1] = d2_laplacian_dxdy
    d2_laplacian_dxdz_full[1:-1] = d2_laplacian_dxdz
    d2_laplacian_dydz_full[1:-1] = d2_laplacian_dydz

    # Sum to get the bilaplacian
    bi_laplacian += d2_laplacian_dx2_full + d2_laplacian_dy2_full + d2_laplacian_dz2_full
    bi_laplacian += 2 * (d2_laplacian_dxdy_full + d2_laplacian_dxdz_full + d2_laplacian_dydz_full)

    # Compute Frobenius norm of Bi-Laplacian
    #bi_laplacian_norm = torch.linalg.norm(bi_laplacian[surface_mask], ord='fro', dim=(-2, -1))
    bi_laplacian_norm = torch.linalg.norm(bi_laplacian[surface_mask], ord=2, dim=-1)

    # Integrate over surface (approximate with mean)
    hessian_energy = bi_laplacian_norm.mean()
    hessian_loss = hessian_energy *hessian_loss_scale + hessian_mean
    return hessian_loss



def sdf_diff_loss(pred, label, weight, scale, l2_loss=True):
    count = pred.shape[0]
    diff = pred - label
    diff_m = diff / scale # so it's still in m unit
    if l2_loss:
        loss = (weight * (diff_m**2)).sum() / count  # l2 loss
    else:
        loss = (weight * abs(diff_m)).sum() / count  # l1 loss
    return loss


def sdf_bce_loss(pred, label, sigma, weight, weighted=False, bce_reduction = "mean"):
    if weighted:
        loss_bce = nn.BCEWithLogitsLoss(reduction=bce_reduction, weight=weight)
    else:
        loss_bce = nn.BCEWithLogitsLoss(reduction=bce_reduction)
    label_op = torch.sigmoid(label / sigma)  # occupancy prob
    loss = loss_bce(pred, label_op)
    return loss


def ray_estimation_loss(x, y, d_meas):  # for each ray
    # x as depth
    # y as sdf prediction
    # d_meas as measured depth

    # print(x.shape, y.shape, d_meas.shape)

    # regard each sample as a ray
    mat_A = torch.vstack((x, torch.ones_like(x))).transpose(0, 1)
    vec_b = y.view(-1, 1)

    # print(mat_A.shape, vec_b.shape)

    least_square_estimate = torch.linalg.lstsq(mat_A, vec_b).solution

    a = least_square_estimate[0]  # -> -1 (added in ekional loss term)
    b = least_square_estimate[1]

    # d_estimate = -b/a
    d_estimate = torch.clamp(-b / a, min=1.0, max=40.0)  # -> d

    # d_error = (d_estimate-d_meas)**2

    d_error = torch.abs(d_estimate - d_meas)

    # print(mat_A.shape, vec_b.shape, least_square_estimate.shape)
    # print(d_estimate.item(), d_meas.item(), d_error.item())

    return d_error


def ray_rendering_loss(x, y, d_meas):  # for each ray [should run in batch]
    # x as depth
    # y as occ.prob. prediction
    x = x.squeeze(1)
    sort_x, indices = torch.sort(x)
    sort_y = y[indices]

    w = torch.ones_like(y)
    for i in range(sort_x.shape[0]):
        w[i] = sort_y[i]
        for j in range(i):
            w[i] *= 1.0 - sort_y[j]

    d_render = (w * x).sum()

    d_error = torch.abs(d_render - d_meas)

    # print(x.shape, y.shape, d_meas.shape)
    # print(mat_A.shape, vec_b.shape, least_square_estimate.shape)
    # print(d_render.item(), d_meas.item(), d_error.item())

    return d_error


def batch_ray_rendering_loss(x, y, d_meas, neus_on=True):  # for all rays in a batch
    # x as depth [ray number * sample number]
    # y as prediction (the alpha in volume rendering) [ray number * sample number]
    # d_meas as measured depth [ray number]
    # w as the raywise weight [ray number]
    # neus_on determine if using the loss defined in NEUS

    # print(x.shape, y.shape, d_meas.shape, w.shape)

    sort_x, indices = torch.sort(x, 1)  # for each row
    sort_y = torch.gather(y, 1, indices)  # for each row

    if neus_on:
        neus_alpha = (sort_y[:, 1:] - sort_y[:, 0:-1]) / ( 1. - sort_y[:, 0:-1] + 1e-10)
        # avoid dividing by 0 (nan)
        # print(neus_alpha)
        alpha = torch.clamp(neus_alpha, min=0.0, max=1.0)
    else:
        alpha = sort_y

    one_minus_alpha = torch.ones_like(alpha) - alpha + 1e-10

    cum_mat = torch.cumprod(one_minus_alpha, 1)

    weights = cum_mat / one_minus_alpha * alpha

    weights_x = weights * sort_x[:, 0 : alpha.shape[1]]

    d_render = torch.sum(weights_x, 1)

    d_error = torch.abs(d_render - d_meas)

    # d_error = torch.abs(d_render - d_meas) * w # times ray-wise weight

    d_error_mean = torch.mean(d_error)

    return d_error_mean
