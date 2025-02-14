import torch
import math
from utils import gradient


def minsurf_loss(normal, pred, coef_alpha):
    loss_minsurf = (normal.norm(2, dim=-1, keepdim=True) * (1 - torch.tanh(coef_alpha * pred) ** 2)).mean()
    # loss_minsurf = torch.sigmoid(coef_alpha * pred).mean()
    return loss_minsurf


def eikonal_loss(normal, loss_type='e_l1'):
    """
    Gropp, A., Yariv, L., Haim, N., Atzmon, M., Lipman, Y., 2020. Implicit geometric regularization for learning shapes,
    in: Proceedings of the 37th International Conference on Machine Learning, pp. 3789–1379.
    https://arxiv.org/pdf/2002.10099
    """
    if loss_type == 'e_l1':
        eikonal = normal.norm(2, dim=-1, keepdim=True) - 1
        loss_eikonal = eikonal.abs().mean()
    elif loss_type == 'e_l2':
        eikonal = normal.norm(2, dim=-1, keepdim=True) - 1
        loss_eikonal = eikonal.square().mean()
    elif loss_type == 're_l1':
        eikonal = torch.relu(-(normal.norm(2, dim=-1, keepdim=True) - 0.8))
        loss_eikonal = eikonal.abs().mean()
    elif loss_type == 're_l2':
        eikonal = torch.relu(-(normal.norm(2, dim=-1, keepdim=True) - 0.8))
        loss_eikonal = eikonal.square().mean()
    else:
        raise ValueError('unexpected eikonal type')
    return eikonal, loss_eikonal


def viscosity_loss(eikonal, laplacian, coef_vis, loss_type='div_l1'):
    if loss_type == 'div_l1':
        loss_vis = (eikonal.squeeze() - coef_vis * laplacian.sum(dim=-1)).abs().mean()
    elif loss_type == 'div_l2':
        loss_vis = (eikonal.squeeze() - coef_vis * laplacian.sum(dim=-1)).square().mean()
    else:
        raise ValueError('unexpected vis type')
    return loss_vis


def laplacian_loss(lapalcian, loss_type='div_l1'):
    """
    Ben-Shabat, Y., Koneputugodage, C.H., Gould, S., 2022. Digs: Divergence guided shape implicit neural representation
    for unoriented point clouds, in: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition,
    pp.19323–19332.
    https://openaccess.thecvf.com/content/CVPR2022/papers/Ben-Shabat_DiGS_Divergence_Guided_Shape_Implicit_Neural_Representation_for_Unoriented_Point_CVPR_2022_paper.pdf
    """
    if loss_type == 'div_l1':
        loss_la = lapalcian.sum(dim=-1).abs().mean()
    elif loss_type == 'div_l2':
        loss_la = lapalcian.sum(dim=-1).square().mean()
    else:
        raise ValueError('unexpected vis type')
    return loss_la


def hessian_loss(hessian, loss_type='div_l2'):
    """
    div_l1:
    Zhang, J., Yao, Y., Li, S., Fang, T., McKinnon, D., Tsin, Y., Quan, L., 2022. Critical regularizations for neural
    surface reconstruction in the wild, in: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern
    Recognition, pp.6270–6279.
    https://openaccess.thecvf.com/content/CVPR2022/papers/Zhang_Critical_Regularizations_for_Neural_Surface_Reconstruction_in_the_Wild_CVPR_2022_paper.pdf
    """
    if loss_type == 'div_l1':
        hessian_norm = torch.linalg.matrix_norm(hessian, ord=1, dim=(-2, -1))
        # hessian_norm = torch.linalg.matrix_norm(hessian, ord='nuc', dim=(-2, -1))
        # hessian_norm = torch.linalg.matrix_norm(hessian, ord='inf', dim=(-2, -1))
    elif loss_type == 'div_l2':
        hessian_norm = torch.linalg.matrix_norm(hessian, ord='fro', dim=(-2, -1)) ** 2
        # hessian_norm = torch.linalg.matrix_norm(hessian, ord='2', dim=(-2, -1))
    else:
        raise ValueError('unexpected hessian type')
    loss_h = hessian_norm.mean()
    return loss_h


def morse_loss(hessian, loss_type='div_l1'):
    """
    Wang, Z., Zhang, Y., Xu, R., Zhang, F., Wang, P.S., Chen, S., Xin, S., Wang, W., Tu, C., 2023. Neural-singular-hessian:
    Implicit neural representation of unoriented point clouds by enforcing singular hessian. ACM Transactions on Graphics
    (TOG) 42, 1–14.
    https://dl.acm.org/doi/abs/10.1145/3618311
    """
    if loss_type == 'div_l1':
        loss_m = torch.linalg.det(hessian).abs().mean()
    elif loss_type == 'div_l2':
        loss_m = torch.linalg.det(hessian).square().mean()
    else:
        raise ValueError('unexpected vis type')
    return loss_m


def alignGH_loss(points, normal, loss_type='div_l2'):
    """
    Wang, R., Wang, Z., Zhang, Y., Chen, S., Xin, S., Tu, C., Wang, W.: Aligning gradient and hessian for neural signed
    distance function. In: NeurIPS (2023)
    https://proceedings.neurips.cc/paper_files/paper/2023/hash/c87bd5843849884e9430f1693b018d71-Abstract-Conference.html
    """
    normal_norm = normal.norm(2, dim=-1, keepdim=True)
    alignGH = gradient(points, normal_norm)
    if loss_type == 'div_l1':
        loss_a = alignGH.norm(1, dim=-1, keepdim=True).mean()
    elif loss_type == 'div_l2':
        loss_a = (alignGH.norm(2, dim=-1, keepdim=True) ** 2).mean()
    else:
        raise ValueError('unexpected vis type')
    return loss_a


def directional_div_loss(points, normal, loss_type='div_l1'):
    """
    Yang, H., Sun, Y., Sundaramoorthi, G., Yezzi, A., 2023. Steik: stabilizing the optimization of neural signed distance
    functions and finer shape representation, in: Proceedings of the 37th International Conference on Neural Information
    Processing Systems, pp. 13993–14004.
    https://arxiv.org/pdf/2305.18414
    """
    dot_grad = (normal * normal).sum(dim=-1, keepdim=True)
    hvp_mnfld = 0.5 * gradient(points, dot_grad)
    div_mnfld = (normal * hvp_mnfld).sum(dim=-1) / ((normal ** 2).sum(dim=-1) + 1e-5)
    if loss_type == 'div_l1':
        loss_ddiv = div_mnfld.abs().mean()
    elif loss_type == 'div_l2':
        loss_ddiv = div_mnfld.square().mean()
    else:
        raise ValueError('unexpected div_type')
    return loss_ddiv


def project_loss(points, pred, normal, model, points_target, loss_type='div_l2', project_type=None):
    from pytorch3d.loss import chamfer_distance
    """
    de:
    Ma, B., Han, Z., Liu, Y.S., Zwicker, M., 2021. Neural-pull: Learning signed distance function from point clouds by learning
    to pull space onto surface, in: Meila, M., Zhang, T. (Eds.), Proceedings of the 38th International Conference on Machine
    Learning, PMLR. pp. 7246–7257.
    https://arxiv.org/pdf/2011.13495

    cd:
    Chao Chen, Zhizhong Han, Yu-Shen Liu, and Matthias Zwicker. Unsupervised learning of fine structure generation for
    3D point clouds by 2D projections matching. In Proceedings of the ieee/cvf international conference on computer vision,
    pages 12466–12477, 2021.
    https://openaccess.thecvf.com/content/CVPR2023/papers/Chen_Unsupervised_Inference_of_Signed_Distance_Functions_From_Single_Sparse_Point_CVPR_2023_paper.pdf
    """
    if project_type is None:
        project_type = ['cd']
    loss_p = torch.zeros(1, device=points.device)
    points_project = points - pred * normal
    if 'zero' in project_type:
        pred_p = model(points_project)
        if loss_type == 'div_l1':
            loss_p += pred_p.abs().mean()
        elif loss_type == 'div_l2':
            loss_p += pred_p.square().mean()
        else:
            raise ValueError('unexpected div_type')
    if 'de' in project_type:
        if loss_type == 'div_l1':
            loss_p += (points_project - points_target).norm(1, dim=-1).mean()
        elif loss_type == 'div_l2':
            loss_p += (points_project - points_target).norm(2, dim=-1).square().mean()
        else:
            raise ValueError('unexpected div_type')
    if 'dde' in project_type:
        pred_project = model(points_project)
        project_grad = gradient(points, pred_project)
        if loss_type == 'div_l1':
            loss_p += project_grad.norm(1, dim=-1).mean()
        elif loss_type == 'div_l2':
            loss_p += project_grad.norm(2, dim=-1).square().mean()
        else:
            raise ValueError('unexpected div_type')
    if 'cd' in project_type:
        loss_p += chamfer_distance(points_project.unsqueeze(0), points_target.unsqueeze(0))[0]
    return loss_p
