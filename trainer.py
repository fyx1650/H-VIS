import numpy as np
import torch
import trimesh

from utils import *
from loss import *


class Trainer(object):
    def __init__(self,
                 name,  # name of this experiment
                 model,  # network
                 device,
                 net_para=None,
                 criterion=None,  # loss function, if None, assume inline implementation in train_step
                 optimizer=None,  # optimizer
                 ema_decay=0.95,  # if use EMA, set the decay
                 lr_type='remain',
                 use_siren=False,
                 local_rank=0,  # which GPU am I
                 mute=False,  # whether to mute all print
                 fp16=False,  # amp optimize level
                 eval_interval=1,  # eval once every $ epoch
                 max_keep_ckpt=2,  # max num of saved ckpts in disk
                 workspace='workspace',  # workspace to save logs & ckpts
                 best_mode='min',  # the smaller/larger result, the better
                 use_checkpoint="latest",  # which ckpt to use at init time
                 use_tensorboardX=True,  # whether to use tensorboard for logging
                 provide_sdf=False,  # whether to provide ground truth sdf values
                 minsurf=False,  # whether to use the minimal surface term as the regularization term
                 eikonal=False,  # whether to use the eikonal term as the regularization term
                 second_order='vis_hessian',
                 dimension='3d',
                 max_epochs=10
                 ):

        self.name = name
        self.net_para = net_para
        self.mute = mute
        self.local_rank = local_rank
        self.workspace = workspace
        self.ema_decay = ema_decay
        self.clip_grad_norm = self.net_para["Loss"]["clip_grad_norm"]
        self.fp16 = fp16
        self.best_mode = best_mode
        self.max_keep_ckpt = max_keep_ckpt
        self.eval_interval = eval_interval
        self.use_checkpoint = use_checkpoint
        self.use_tensorboardX = use_tensorboardX
        self.time_stamp = time.strftime("%Y-%m-%d_%H-%M-%S")
        self.lr_type = lr_type
        self.use_siren = use_siren
        self.device = device
        self.console = Console()
        self.provide_sdf = provide_sdf
        self.with_normal = self.net_para["Loss"]["with_normal"]
        self.minsurf = minsurf
        self.eikonal = eikonal
        self.second_order = second_order
        self.dimension = dimension
        self.max_epochs = max_epochs
        self.eps = 1e-5

        self.minsurf_mnfld = self.net_para["Loss"]["minsurf_mnfld"]
        self.minsurf_near = self.net_para["Loss"]["minsurf_near"]
        self.minsurf_far = self.net_para["Loss"]["minsurf_far"]
        self.E_contain_mnfld = self.net_para["Loss"]["E_contain_mnfld"]
        self.E_contain_near = self.net_para["Loss"]["E_contain_near"]
        self.E_contain_far = self.net_para["Loss"]["E_contain_far"]
        self.div_contain_mnfld = self.net_para["Loss"]["div_contain_mnfld"]
        self.div_contain_near = self.net_para["Loss"]["div_contain_near"]
        self.div_contain_far = self.net_para["Loss"]["div_contain_far"]

        matplotlib.use('Agg')

        model.to(self.device)
        self.model = model

        if isinstance(criterion, nn.Module):
            criterion.to(self.device)
        self.criterion = criterion

        if optimizer is None:
            self.optimizer = optim.Adam(self.model.parameters(), lr=0.001, weight_decay=0, betas=(0.9, 0.99), eps=1e-15)  # naive adam
        else:
            self.optimizer = optimizer(self.model)

        if ema_decay is not None:
            self.ema = ExponentialMovingAverage(self.model.parameters(), decay=ema_decay)
        else:
            self.ema = None

        self.scaler = torch.cuda.amp.GradScaler(enabled=self.fp16)

        # variable init
        self.epoch = 0
        self.global_step = 0
        self.local_step = 0
        self.stats = {
            "loss": [],
            "valid_loss": [],
            "results": [],  # metrics[0], or valid_loss
            "checkpoints": [],  # record path of saved ckpt, to automatically remove old ckpt
            "best_result": None,
        }

        # workspace prepare
        self.log_ptr = None
        if self.workspace is not None:
            os.makedirs(self.workspace, exist_ok=True)
            self.log_path = os.path.join(workspace, f"log_{self.name}.txt")
            self.log_ptr = open(self.log_path, "a+")

            self.ckpt_path = os.path.join(self.workspace, 'checkpoints')
            self.best_path = f"{self.ckpt_path}/{self.name}.pth.tar"
            os.makedirs(self.ckpt_path, exist_ok=True)

        self.log(
            f'[INFO] Trainer: {self.name} | {self.time_stamp} | {self.device} | {"fp16" if self.fp16 else "fp32"} | {self.workspace}')
        self.log(f'[INFO] #parameters: {sum([p.numel() for p in model.parameters() if p.requires_grad])}')

        if self.workspace is not None:
            if self.use_checkpoint == "scratch":
                self.log("[INFO] Training from scratch ...")
            elif self.use_checkpoint == "latest":
                self.log("[INFO] Loading latest checkpoint ...")
                self.load_checkpoint()
            elif self.use_checkpoint == "best":
                if os.path.exists(self.best_path):
                    self.log("[INFO] Loading best checkpoint ...")
                    self.load_checkpoint(self.best_path)
                else:
                    self.log(f"[INFO] {self.best_path} not found, loading latest ...")
                    self.load_checkpoint()
            else:  # path to ckpt
                self.log(f"[INFO] Loading {self.use_checkpoint} ...")
                self.load_checkpoint(self.use_checkpoint)

        if self.use_tensorboardX:
            self.writer = tensorboardX.SummaryWriter(os.path.join(self.workspace, "run", self.name))

        self.loss_dict = {'loss_mnfld': torch.zeros(1, device=self.device),
                          'loss_nonmnfld': torch.zeros(1, device=self.device),
                          'loss_near': torch.zeros(1, device=self.device),
                          'loss_minsurf': torch.zeros(1, device=self.device),
                          'loss_minsurf_mnfld': torch.zeros(1, device=self.device),
                          'loss_minsurf_nonmnfld': torch.zeros(1, device=self.device),
                          'loss_minsurf_near': torch.zeros(1, device=self.device),
                          'loss_normal': torch.zeros(1, device=self.device),
                          'loss_eikonal': torch.zeros(1, device=self.device),
                          'loss_eikonal_mnfld': torch.zeros(1, device=self.device),
                          'loss_eikonal_nonmnfld': torch.zeros(1, device=self.device),
                          'loss_eikonal_near': torch.zeros(1, device=self.device),
                          'loss_vis': torch.zeros(1, device=self.device),
                          'loss_vis_mnfld': torch.zeros(1, device=self.device),
                          'loss_vis_nonmnfld': torch.zeros(1, device=self.device),
                          'loss_vis_near': torch.zeros(1, device=self.device),
                          'loss_div': torch.zeros(1, device=self.device),
                          'loss_div_mnfld': torch.zeros(1, device=self.device),
                          'loss_div_nonmnfld': torch.zeros(1, device=self.device),
                          'loss_div_near': torch.zeros(1, device=self.device),
                          'loss_project': torch.zeros(1, device=self.device),
                          'loss_project_mnfld': torch.zeros(1, device=self.device),
                          'loss_project_nonmnfld': torch.zeros(1, device=self.device),
                          'loss_project_near': torch.zeros(1, device=self.device)
                          }

        for index, (key, w_variable) in enumerate(self.model.named_parameters()):
            self.log('{}|{}'.format(key, w_variable))

    def __del__(self):
        if self.log_ptr:
            self.log_ptr.close()

    def log(self, *args, **kwargs):
        if not self.mute:
            # print(*args)
            self.console.print(*args, **kwargs)
        if self.log_ptr:
            print(*args, file=self.log_ptr)
            self.log_ptr.flush()  # write immediately to file

    # ----------------------------UPDATING---------------------------- #

    def update_learning_rate(self, batch_size):
        warm_up = batch_size // 5
        max_iter = batch_size * self.max_epochs // self.net_para["BasicInfo"]["iter"]
        init_lr = self.net_para["BasicInfo"]["init_lr"]
        gamma = 0.9
        curr_iter = self.global_step // self.net_para["BasicInfo"]["iter"]
        if self.lr_type == 'remain':
            lr = init_lr
        elif self.lr_type == 'pow':
            lr = max(5e-7, init_lr * gamma ** (self.epoch - 1))
        elif self.lr_type == 'cos':
            lr = init_lr * (1 - 1e-2) * (0.5 * (math.cos(curr_iter / max_iter * math.pi) + 1)) + init_lr * 1e-2
            # lr = init_lr * (0.5 * (math.cos(curr_iter / max_iter * math.pi) + 1 + 2e-2))
        elif self.lr_type == 'anneal_cos':
            lr = init_lr * (curr_iter / warm_up) if curr_iter < warm_up else init_lr * 0.5 * (
                        math.cos((curr_iter - warm_up) / (max_iter - warm_up) * math.pi) + 1)
        else:
            raise ValueError('unexpected lr type')
        for g in self.optimizer.param_groups:
            g['lr'] = lr

    def update_coefficient(self, i=0, batch_size=200):
        coefficient = {}

        # the coefficient of data term and the first order regularization term usually no need to change
        coefficient['coef_mnfld'] = self.net_para["Loss"]["coef_mnfld"]
        coefficient['coef_nonmnfld'] = self.net_para["Loss"]["coef_nonmnfld"]
        coefficient['coef_near'] = self.net_para["Loss"]["coef_near"]
        coefficient['coef_alpha'] = self.net_para["Loss"]["coef_alpha"]
        coefficient['coef_ms'] = self.net_para["Loss"]["coef_ms"]
        coefficient['coef_normal'] = self.net_para["Loss"]["coef_normal"]
        coefficient['coef_eikonal'] = self.net_para["Loss"]["coef_eikonal"]
        coefficient['coef_proportion'] = self.net_para["Loss"]["coef_proportion"]
        digs_para = self.net_para["Loss"]["digs_para"]

        # the coefficient of the second order regularization term usually down to a small value to avoid oversmooth
        init_coef_vis = self.net_para["Loss"]["coef_vis"]
        init_coef_div = self.net_para["Loss"]["coef_div"]
        warm_up = batch_size // 5
        max_iter = self.max_epochs * batch_size // self.net_para["BasicInfo"]["iter"]
        curr_iter = self.global_step // self.net_para["BasicInfo"]["iter"]
        coef_vis_down_type = self.net_para["Loss"]["coef_vis_down_type"]
        coef_div_down_type = self.net_para["Loss"]["coef_div_down_type"]  # choose from [remain, digs, pow_full]

        coefficient['coef_project'] = self.net_para["Loss"]["coef_project"]

        if coef_div_down_type == 'remain':
            coefficient['coef_div'] = init_coef_div
        elif coef_div_down_type == 'digs':
            if curr_iter / max_iter <= 0.2:
                coefficient['coef_div'] = digs_para[0]
            elif (curr_iter / max_iter > 0.2) and (curr_iter / max_iter <= 0.4):
                coefficient['coef_div'] = digs_para[0] - (curr_iter - max_iter * 0.2) / (
                        max_iter * 0.2) * (digs_para[0] - digs_para[1])
            else:
                coefficient['coef_div'] = digs_para[1] - (curr_iter - max_iter * 0.4) / (
                        max_iter * 0.6) * (digs_para[1] - digs_para[2])

            # if curr_iter / max_iter <= 0.2:
            #     coefficient['coef_div'] = digs_para[0]
            # elif (curr_iter / max_iter > 0.2) and (curr_iter / max_iter <= 0.4):
            #     coefficient['coef_div'] = digs_para[0] - (curr_iter - max_iter * 0.2) / (
            #             max_iter * 0.2) * (digs_para[0] - digs_para[1])
            # elif (curr_iter / max_iter > 0.4) and (curr_iter / max_iter <= 0.6):
            #     coefficient['coef_div'] = digs_para[1]
            # elif (curr_iter / max_iter > 0.6) and (curr_iter / max_iter <= 0.8):
            #     coefficient['coef_div'] = digs_para[1] - (curr_iter - max_iter * 0.6) / (
            #             max_iter * 0.2) * (digs_para[1] - digs_para[2])
            # else:
            #     coefficient['coef_div'] = digs_para[2]

            # if curr_iter / max_iter <= 1/10:
            #     coefficient['coef_div'] = digs_para[0]
            # elif (curr_iter / max_iter > 1/10) and (curr_iter / max_iter <= 2/10):
            #     coefficient['coef_div'] = digs_para[0] - (curr_iter - max_iter * 1/10) / (
            #             max_iter * 1/10) * (digs_para[0] - digs_para[1])
            # elif (curr_iter / max_iter > 2/10) and (curr_iter / max_iter <= 3/10):
            #     coefficient['coef_div'] = digs_para[1]
            # elif (curr_iter / max_iter > 3 / 10) and (curr_iter / max_iter <= 4 / 10):
            #     coefficient['coef_div'] = digs_para[1] - (curr_iter - max_iter * 3 / 10) / (
            #             max_iter * 1 / 10) * (digs_para[1] - digs_para[2])
            # elif (curr_iter / max_iter > 4 / 10) and (curr_iter / max_iter <= 5 / 10):
            #     coefficient['coef_div'] = digs_para[2]
            # elif (curr_iter / max_iter > 5 / 10) and (curr_iter / max_iter <= 6 / 10):
            #     coefficient['coef_div'] = digs_para[2] - (curr_iter - max_iter * 5 / 10) / (
            #             max_iter * 1 / 10) * (digs_para[2] - digs_para[3])
            # else:
            #     coefficient['coef_div'] = digs_para[3]

            # if curr_iter / max_iter <= 1/10:
            #     coefficient['coef_div'] = digs_para[0]
            # elif (curr_iter / max_iter > 1/10) and (curr_iter / max_iter <= 2/10):
            #     coefficient['coef_div'] = digs_para[1]
            # elif (curr_iter / max_iter > 2/10) and (curr_iter / max_iter <= 3/10):
            #     coefficient['coef_div'] = digs_para[2]
            # elif (curr_iter / max_iter > 3 / 10) and (curr_iter / max_iter <= 4 / 10):
            #     coefficient['coef_div'] = digs_para[3]
            # elif (curr_iter / max_iter > 4 / 10) and (curr_iter / max_iter <= 5 / 10):
            #     coefficient['coef_div'] = digs_para[4]
            # else:
            #     coefficient['coef_div'] = digs_para[5]
        elif coef_div_down_type == 'pow_full':
            # from 1 to 10^-7
            coefficient['coef_div'] = 10 ** (math.log10(init_coef_div) +
                                             curr_iter * (-math.log10(init_coef_div)-7) / max_iter)
        else:
            raise ValueError('unexpected type')

        if coef_vis_down_type == 'remain':
            coefficient['coef_vis'] = init_coef_vis
        elif coef_vis_down_type == 'digs':
            if curr_iter / max_iter <= 0.2:
                coefficient['coef_vis'] = digs_para[0]
            elif (curr_iter / max_iter > 0.2) and (curr_iter / max_iter <= 0.4):
                coefficient['coef_vis'] = digs_para[0] - (curr_iter - max_iter * 0.2) / (
                        max_iter * 0.2) * (digs_para[0] - digs_para[1])
            else:
                coefficient['coef_vis'] = digs_para[1] - (curr_iter - max_iter * 0.4) / (
                        max_iter * 0.6) * (digs_para[1] - digs_para[2])

            # if curr_iter / max_iter <= 0.2:
            #     coefficient['coef_vis'] = digs_para[0]
            # elif (curr_iter / max_iter > 0.2) and (curr_iter / max_iter <= 0.4):
            #     coefficient['coef_vis'] = digs_para[0] - (curr_iter - max_iter * 0.2) / (
            #             max_iter * 0.2) * (digs_para[0] - digs_para[1])
            # elif (curr_iter / max_iter > 0.4) and (curr_iter / max_iter <= 0.6):
            #     coefficient['coef_vis'] = digs_para[1]
            # elif (curr_iter / max_iter > 0.6) and (curr_iter / max_iter <= 0.8):
            #     coefficient['coef_vis'] = digs_para[1] - (curr_iter - max_iter * 0.6) / (
            #             max_iter * 0.2) * (digs_para[1] - digs_para[2])
            # else:
            #     coefficient['coef_vis'] = digs_para[2]

            # if curr_iter / max_iter <= 1/10:
            #     coefficient['coef_vis'] = digs_para[0]
            # elif (curr_iter / max_iter > 1/10) and (curr_iter / max_iter <= 2/10):
            #     coefficient['coef_vis'] = digs_para[0] - (curr_iter - max_iter * 1/10) / (
            #             max_iter * 1/10) * (digs_para[0] - digs_para[1])
            # elif (curr_iter / max_iter > 2/10) and (curr_iter / max_iter <= 3/10):
            #     coefficient['coef_vis'] = digs_para[1]
            # elif (curr_iter / max_iter > 3 / 10) and (curr_iter / max_iter <= 4 / 10):
            #     coefficient['coef_vis'] = digs_para[1] - (curr_iter - max_iter * 3 / 10) / (
            #             max_iter * 1 / 10) * (digs_para[1] - digs_para[2])
            # elif (curr_iter / max_iter > 4 / 10) and (curr_iter / max_iter <= 5 / 10):
            #     coefficient['coef_vis'] = digs_para[2]
            # elif (curr_iter / max_iter > 5 / 10) and (curr_iter / max_iter <= 6 / 10):
            #     coefficient['coef_vis'] = digs_para[2] - (curr_iter - max_iter * 5 / 10) / (
            #             max_iter * 1 / 10) * (digs_para[2] - digs_para[3])
            # else:
            #     coefficient['coef_vis'] = digs_para[3]

            # if curr_iter / max_iter <= 1/10:
            #     coefficient['coef_vis'] = digs_para[0]
            # elif (curr_iter / max_iter > 1/10) and (curr_iter / max_iter <= 2/10):
            #     coefficient['coef_vis'] = digs_para[1]
            # elif (curr_iter / max_iter > 2/10) and (curr_iter / max_iter <= 3/10):
            #     coefficient['coef_vis'] = digs_para[2]
            # elif (curr_iter / max_iter > 3 / 10) and (curr_iter / max_iter <= 4 / 10):
            #     coefficient['coef_vis'] = digs_para[3]
            # elif (curr_iter / max_iter > 4 / 10) and (curr_iter / max_iter <= 5 / 10):
            #     coefficient['coef_vis'] = digs_para[4]
            # else:
            #     coefficient['coef_vis'] = digs_para[5]
        elif coef_vis_down_type == 'pow_full':
            # from 1 to 10^-7
            coefficient['coef_vis'] = 10 ** (math.log10(init_coef_vis) +
                                             curr_iter * (-math.log10(init_coef_vis)-7) / max_iter)
        else:
            raise ValueError('unexpected type')

        return coefficient

    # gradually add the level of the encoder, but we don't use this in our experiment
    def update_encoder_levels(self, i=0, batch_size=400):
        num_levels = self.model.encoder.num_levels
        active_level = min(self.epoch - 1 + self.model.encoder.init_active_level, num_levels - 1)
        self.model.encoder.set_active_level(active_level)

        # step = math.ceil(batch_size / (2 * num_levels + 1))
        # curr_status = self.global_step // step
        # for i in range(num_levels):
        #     # if i < abs(num_levels - curr_status):
        #     #     self.model.encoder.embeddings[num_levels - 1 - i].requires_grad = True
        #     # else:
        #     #     self.model.encoder.embeddings[num_levels - 1 - i].requires_grad = False
        #
        #     if i < abs(num_levels - curr_status):
        #         self.model.encoder.embeddings[i].requires_grad = True
        #     else:
        #         self.model.encoder.embeddings[i].requires_grad = False
        # if self.epoch > 1:
        #     for i in range(self.epoch-1):
        #         self.model.encoder.embeddings[self.model.encoder.num_levels - 1 - i].requires_grad = False
        return active_level

    # ----------------------------VISUALIZATION---------------------------- #

    def draw_implicit_function_1d(self, bbox, a, b):
        save_path = os.path.join(self.workspace, 'validation', f'{self.name}_{self.epoch}_result.png')
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        save_path_1 = os.path.join(self.workspace, 'validation', f'{self.name}_{self.epoch}_result_grad.png')
        os.makedirs(os.path.dirname(save_path_1), exist_ok=True)
        save_path_2 = os.path.join(self.workspace, 'validation', f'{self.name}_{self.epoch}_result_local.png')
        os.makedirs(os.path.dirname(save_path_2), exist_ok=True)
        save_path_3 = os.path.join(self.workspace, 'validation', f'{self.name}_{self.epoch}_result_local_grad.png')
        os.makedirs(os.path.dirname(save_path_2), exist_ok=True)
        self.log(f"==> Saving result to {save_path} and {save_path_1} and {save_path_2}")

        # [-1,1] res:1024
        inputs_2 = torch.linspace(-2, 2, 1025).unsqueeze(1).to(self.device)
        inputs_2.requires_grad_()
        outputs_2 = self.model(inputs_2)
        gradients_2 = gradient(inputs_2, outputs_2)
        X_2 = inputs_2.detach().cpu().numpy()
        Y_2 = outputs_2.detach().cpu().numpy()
        G_2 = gradients_2.detach().cpu().numpy()

        # (a+b)/2 neighborhood res: 1025
        c = (a + 3 * b) / 4
        inputs_4 = torch.linspace(c - 16 / 64, c + 16 / 64, 1025).unsqueeze(1).to(self.device)
        inputs_4.requires_grad_()
        outputs_4 = self.model(inputs_4)
        gradients_4 = gradient(inputs_4, outputs_4)
        X_4 = inputs_4.detach().cpu().numpy()
        Y_4 = outputs_4.detach().cpu().numpy()
        G_4 = gradients_4.detach().cpu().numpy()

        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(20, 20))
        fig.tight_layout(h_pad=2)
        ax.cla()
        ax.spines['left'].set_position(('data', bbox[0, 0]))
        ax.spines['bottom'].set_position('center')
        ax.spines['right'].set_color('none')
        ax.spines['top'].set_color('none')
        ax.spines['left'].set_linewidth(10)
        ax.spines['bottom'].set_linewidth(10)
        ax.set_xlim(bbox[0, 0] - 0.4, bbox[1, 0])
        ax.set_ylim(-1.3, 1.3)
        # ax.set_aspect(1)
        ax.set_aspect(1.33)
        ax.plot(1, 0, ">k", transform=ax.get_yaxis_transform(), ms=40, clip_on=False)
        ax.plot(bbox[0, 0], 1, "^k", transform=ax.get_xaxis_transform(), ms=40, clip_on=False)

        ax.set_xlabel('Location', fontsize=75, color='black')
        ax.set_ylabel('Distance', fontsize=75, color='red')
        ax.xaxis.set_label_coords(0.9, 0.63)
        ax.yaxis.set_label_coords(0, 0.8)
        # ax.set_xticks([])
        ax.set_yticks([-1, -0.5, 0.5, 1])
        ax.tick_params(axis='both', labelsize=75, length=8, width=6)
        ax.plot(X_2, Y_2, 'r', linewidth=10)
        ax.scatter([-1, 1], [0, 0], c='b', s=1000)
        plt.subplots_adjust(top=0.9, bottom=0.1, left=0.1, right=0.9)
        plt.savefig(save_path)
        plt.close(fig)
        fig_g, ax_g = plt.subplots(nrows=1, ncols=1, figsize=(20, 20))
        fig_g.tight_layout(h_pad=2)
        ax_g.cla()
        ax_g.spines['left'].set_position(('data', bbox[0, 0]))
        ax_g.spines['bottom'].set_position('center')
        ax_g.spines['right'].set_color('none')
        ax_g.spines['top'].set_color('none')
        ax_g.spines['left'].set_linewidth(10)
        ax_g.spines['bottom'].set_linewidth(10)
        ax_g.set_xlim(bbox[0, 0] - 0.4, bbox[1, 0])
        ax_g.set_ylim(-1.3, 1.3)
        # ax_g.set_aspect(1)
        ax_g.set_aspect(1.33)
        ax_g.plot(1, 0, ">k", transform=ax.get_yaxis_transform(), ms=40, clip_on=False)
        ax_g.plot(bbox[0, 0], 1, "^k", transform=ax.get_xaxis_transform(), ms=40, clip_on=False)
        # ax_g.set_xlabel('Location', fontsize=75, color='black')
        # ax_g.set_ylabel('Gradient', fontsize=75, color='blue')
        ax_g.xaxis.set_label_coords(0.9, 0.63)
        ax_g.yaxis.set_label_coords(0, 0.8)
        # ax_g.set_xticks([])
        ax_g.set_yticks([-1, -0.5, 0.5, 1])
        ax_g.tick_params(axis='both', labelsize=75, length=8, width=6)
        cubic_sp = interpolate.interp1d(X_2[:, 0], G_2[:, 0], kind='cubic')
        X_2_test = np.linspace(-2, 2, 1025)
        ax_g.plot(X_2_test, cubic_sp(X_2_test), 'b', linewidth=10)
        plt.subplots_adjust(top=0.9, bottom=0.1, left=0.1, right=0.9)
        plt.savefig(save_path_1)
        plt.close(fig_g)

        fig_local, ax_local = plt.subplots(nrows=1, ncols=1, figsize=(20, 20))
        fig_local.tight_layout(h_pad=2)
        ax_local.cla()
        # ax_local.set_ylim(-0.0493, -0.0492)
        ax_local.set_aspect(1)
        ax_local.set_axis_off()
        ax_local.plot(X_4, Y_4, 'r', linewidth=50)
        plt.subplots_adjust(top=0.9, bottom=0.1, left=0.1, right=0.9)
        plt.savefig(save_path_2)
        plt.close(fig_local)
        fig_local_g, ax_local_g = plt.subplots(nrows=1, ncols=1, figsize=(20, 20))
        fig_local_g.tight_layout(h_pad=2)
        ax_local_g.cla()
        ax_local_g.set_axis_off()
        if self.epoch == 0:
            ax_local_g.set_ylim(-0.1, 0.1)
        else:
            ax_local_g.set_ylim(-1.1, -0.9)
        ax_local_g.plot(X_4, G_4, 'b', linewidth=50)
        plt.subplots_adjust(top=0.9, bottom=0.1, left=0.1, right=0.9)
        plt.savefig(save_path_3)
        plt.close(fig_local_g)

    def draw_implicit_function_2d(self, bbox, points_gt, res=512):
        save_path = os.path.join(self.workspace, 'validation', f'{self.name}_{self.epoch}_result.png')
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        save_path_1 = os.path.join(self.workspace, 'validation', f'{self.name}_{self.epoch}_result_project.png')
        os.makedirs(os.path.dirname(save_path_1), exist_ok=True)
        save_path_2 = os.path.join(self.workspace, 'validation', f'{self.name}_{self.epoch}_result_3d.html')
        os.makedirs(os.path.dirname(save_path_2), exist_ok=True)
        save_path_g = os.path.join(self.workspace, 'validation', f'{self.name}_{self.epoch}_result_g.png')
        os.makedirs(os.path.dirname(save_path_g), exist_ok=True)
        save_path_h = os.path.join(self.workspace, 'validation', f'{self.name}_{self.epoch}_result_h.png')
        os.makedirs(os.path.dirname(save_path_h), exist_ok=True)
        self.log(f"==> Saving result to {save_path} and {save_path_1} and {save_path_2}")

        x = torch.linspace(bbox[0, 0], bbox[1, 0], res + 1)
        y = torch.linspace(bbox[0, 1], bbox[1, 1], res + 1)
        xx, yy = torch.meshgrid([x, y], indexing='xy')
        points_all = torch.cat((xx.unsqueeze(-1), yy.unsqueeze(-1)), dim=-1)
        points_all = torch.reshape(points_all, [(res + 1) ** 2, 2]).to(self.device)
        points_all.requires_grad_()
        z = []
        g_norm = []
        h_norm = []
        points_new = []
        for points in torch.split(points_all, 128 ** 2, dim=0):
            z_temp = self.model(points)
            z.append(z_temp.detach().cpu())
            g_temp = gradient(points, z_temp)
            g_norm.append(g_temp.norm(2, dim=1, keepdim=True).detach().cpu())
            h_temp = torch.zeros(points.shape[0], 2, 2, device=self.device)
            h_temp[:, 0, :] = gradient(points, g_temp[:, 0])
            h_temp[:, 1, :] = gradient(points, g_temp[:, 1])
            h_norm.append((torch.linalg.matrix_norm(h_temp, ord='fro', dim=(-2, -1), keepdim=True).detach().cpu()) ** 2)
            points_new.append((points - z_temp * g_temp).detach().cpu())
            torch.cuda.empty_cache()
        z = torch.cat(z, dim=0).to(self.device)
        g_norm = torch.cat(g_norm, dim=0).to(self.device)
        h_norm = torch.cat(h_norm, dim=0).to(self.device)
        points_new = torch.cat(points_new, dim=0).to(self.device)
        z = z.transpose(0, 1)
        g_norm = g_norm.transpose(0, 1)
        h_norm = h_norm.transpose(0, 1)
        xn = x.detach().cpu().numpy()
        yn = y.detach().cpu().numpy()
        zn = z.detach().cpu().numpy()
        gn = g_norm.detach().cpu().numpy()
        hn = h_norm.detach().cpu().numpy()
        pointsn = points_all.detach().cpu().numpy()
        points_newn = points_new.detach().cpu().numpy()
        qx, qy = np.meshgrid(np.linspace(0, res, 32 + 1, dtype=np.int),
                             np.linspace(0, res, 32 + 1, dtype=np.int), indexing='xy')
        array_qui = qx.flatten() + qy.flatten() * res
        qui = ff.create_quiver(pointsn[array_qui, 0], pointsn[array_qui, 1],
                               points_newn[array_qui, 0] - pointsn[array_qui, 0],
                               points_newn[array_qui, 1] - pointsn[array_qui, 1], scale=1,
                               arrow_scale=0.05, line=dict(color='blue', width=0.2))
        zn = np.reshape(zn, (res + 1, res + 1))
        gn = np.reshape(gn, (res + 1, res + 1))
        hn = np.reshape(hn, (res + 1, res + 1))
        layout = go.Layout(width=2160, height=2160,
                           xaxis=dict(side="bottom", range=[bbox[0, 0], bbox[1, 0]], showticklabels=False),
                           yaxis=dict(side="left", range=[bbox[0, 1], bbox[1, 1]], showticklabels=False),
                           scene=dict(xaxis=dict(range=[bbox[0, 0], bbox[1, 0]], autorange=False),
                                      yaxis=dict(range=[bbox[0, 1], bbox[1, 1]], autorange=False),
                                      aspectratio=dict(x=1, y=1)),
                           showlegend=False,
                           # title=dict(text='distance map', y=0.95, x=0.5, xanchor='center', yanchor='middle',
                           #            font=dict(family='Serif', size=24, color='red'))
                           )
        traces = []
        traces_g = []
        traces_h = []
        # start_sdf = np.min(zn)
        # end_sdf = np.max(zn)
        traces.append(go.Contour(x=xn, y=yn, z=zn,
                                 colorscale='Geyser',
                                 colorbar=dict(tickfont=dict(size=60), thickness=20, tickwidth=3),
                                 # autocontour=True,
                                 contours=dict(
                                     start=-0.3,
                                     end=0.3,
                                     size=0.025,
                                 ),
                                 line=dict(width=3),
                                 showscale=True
                                 ))  # contour trace
        zero_pts = go.Scatter(x=points_gt[:, 0], y=points_gt[:, 1], mode='markers',
                              marker=dict(size=15, color='white'))
        zero_line = go.Contour(x=xn, y=yn, z=zn,
                               contours=dict(start=0, end=0, coloring='lines'),
                               line=dict(width=10),
                               showscale=False,
                               colorscale=[[0, 'rgb(0, 0, 0)'],
                                           [1, 'rgb(0, 0, 0)']])
        project_result = go.Scatter(x=points_newn[:, 0], y=points_newn[:, 1], mode='markers',
                                    marker=dict(color='red', size=5))
        traces.append(zero_line)  # black bold zero line
        traces.append(zero_pts)
        traces_g.append(go.Contour(x=xn, y=yn, z=gn,
                                   colorscale='Viridis',
                                   # autocontour=True,
                                   contours=dict(
                                       start=1 - 0.5,
                                       end=1 + 0.5,
                                       size=0.025 * 8
                                   ), showscale=True
                                   ))
        min_h = np.min(hn)
        max_h = np.max(hn)
        traces_h.append(go.Contour(x=xn, y=yn, z=hn,
                                   colorscale='GnBu',
                                   colorbar=dict(tickfont=dict(size=60), thickness=60, tickwidth=3),
                                   # autocontour=True,
                                   contours=dict(
                                       start=0,
                                       end=120000,
                                       size=120000 / 24
                                   ),
                                   line=dict(width=3),
                                   showscale=True
                                   ))

        fig = go.Figure(data=traces, layout=layout)
        fig1 = go.Figure(data=[zero_line, project_result, qui.data[0]], layout=layout)
        fig_g = go.Figure(data=traces_g, layout=layout)
        fig_h = go.Figure(data=traces_h, layout=layout)
        # fig.show()
        plotly.io.write_image(fig, save_path)
        plotly.io.write_image(fig1, save_path_1)
        fig2 = go.Figure(data=[go.Surface(x=xn, y=yn, z=zn)])
        fig2.update_layout(title="3D Surface",
                           autosize=False,
                           width=1920,
                           height=1680,
                           margin=dict(l=65, r=50, b=65, t=90)
                           )
        plotly.io.write_html(fig2, file=save_path_2, auto_play=False)
        # plotly.io.write_image(fig3, save_path_g)
        plotly.io.write_image(fig_h, save_path_h)
        # fig2.show()
        self.log(f"==> Finished saving result.")

    def draw_implicit_function_3d(self, bbox, center, scale, bbox_model=None, save_path=None, save_path_1=None,
                                  resolution=256, mesh_gt=None, normals_gt_all=None):
        if save_path is None:
            save_path = os.path.join(self.workspace, 'validation', f'{self.name}_{self.epoch}.ply')
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
        if save_path_1 is None:
            save_path_1 = os.path.join(self.workspace, 'validation', f'{self.name}_{self.epoch}_cut.png')
            os.makedirs(os.path.dirname(save_path_1), exist_ok=True)
        self.log(f"==> Saving mesh to {save_path} and {save_path_1}")

        def query_func(pts):
            pts = pts.to(self.device)
            with torch.no_grad():
                with torch.cuda.amp.autocast(enabled=self.fp16):
                    sdfs = -self.model(pts)
            return sdfs

        if bbox_model is None:
            bounds_min = torch.FloatTensor(bbox[0, :])
            bounds_max = torch.FloatTensor(bbox[1, :])
        else:
            bounds_min = torch.FloatTensor(bbox_model[0, :])
            bounds_max = torch.FloatTensor(bbox_model[1, :])

        vertices, triangles = extract_geometry(bounds_min, bounds_max, resolution=resolution,
                                               threshold=0.0,
                                               query_func=query_func)

        # points = torch.from_numpy(vertices).to(self.device).float()
        # points.requires_grad_()
        # cur = compute_curvature(points, self.model, c_type='gaussian')
        # v_color = map_curvature_to_color(cur, vertices)

        if vertices.shape[0] == 0:
            self.log(f"the result of epoch {self.epoch} is none")
            return 0
        # if points.shape[0] > 2 ** 23:
        #     self.log(f"unreliable result: the result of epoch {self.epoch} contains too much points")
        #     mesh = trimesh.Trimesh(vertices=vertices, faces=triangles, process=False)
        #     mesh.export(save_path)
        #     return 0

        vertices = vertices / self.net_para["TPBEncoder"]["model_scale"] * scale + center
        mesh = trimesh.Trimesh(vertices=vertices, faces=triangles, process=False)
        # mesh.visual.vertex_colors = v_color
        mesh.export(save_path)
        plot_cuts_iso(self.model, bbox, save_path=save_path_1, device=self.device)

        if mesh_gt is not None:
            vs_gt = mesh_gt.vertices
            vs = mesh.vertices

            # points_gt = torch.from_numpy(np.asarray((vs_gt - center) / scale)).to(self.device).float()
            # points_gt.requires_grad_()
            # cur_gt = compute_curvature(points_gt, self.model, c_type='gaussian')
            # v_color_gt = map_curvature_to_color(cur_gt, vs_gt)
            # mesh_new = mesh_gt.copy()
            # mesh_new.visual.vertex_colors = v_color_gt
            # save_path_new = os.path.join(self.workspace, 'validation', f'{self.name}_{self.epoch}_gt.ply')
            # os.makedirs(os.path.dirname(save_path_new), exist_ok=True)
            # mesh_new.export(save_path_new)

            v_center = mesh_gt.centroid
            vs_gt = vs_gt - v_center
            v_scale = np.abs(np.asarray(vs_gt)).max()
            vs_gt = vs_gt / v_scale * 0.5
            vs = (vs - v_center) / v_scale * 0.5

            mesh_gt.vertices = vs_gt
            mesh.vertices = vs

            l1_cd, l2_cd, normals_c, f_score_pts = eval_pts(mesh_gt, mesh, normals_gt_all=normals_gt_all,
                                                            num_samples=10 ** 5, miu=5e-3)
            if isinstance(mesh_gt, trimesh.Trimesh):
                f_score = calculate_f_score(mesh_gt, mesh)
            else:
                f_score = 0
            self.log(f"Chamfer distance L1 is {l1_cd:.12f}, L2 is {l2_cd:.12f}, normals_c is {normals_c:.12f}, "
                     f"f_score is {f_score:.12f}, f_score_pts is {f_score_pts:.12f}")

        self.log(f"==> Finished saving mesh.")

    # ----------------------------TRAINING---------------------------- #

    def train_step(self, data, coefficient):
        # torch.cuda.empty_cache()
        loss = torch.zeros(1, device=self.device)
        loss_mnfld = torch.zeros(1, device=self.device)
        loss_near = torch.zeros(1, device=self.device)
        loss_nonmnfld = torch.zeros(1, device=self.device)
        loss_data = torch.zeros(1, device=self.device)
        loss_minsurf = torch.zeros(1, device=self.device)
        loss_eikonal = torch.zeros(1, device=self.device)
        loss_vis = torch.zeros(1, device=self.device)
        loss_div = torch.zeros(1, device=self.device)
        loss_project = torch.zeros(1, device=self.device)

        # ---------origin inputs--------- #
        points_mnfld = data["points_mnfld"][0]  # [B, 3]
        points_nonmnfld = data["points_nonmnfld"][0]
        points_near = data["points_near"][0]
        points_target_near = data["points_target_near"][0]
        points_target_nonmnfld = data["points_target_nonmnfld"][0]
        sdfs_mnfld = data["sdfs_mnfld"][0]  # [B, 1]
        sdfs_nonmnfld = data["sdfs_nonmnfld"][0]
        sdfs_near = data["sdfs_near"][0]
        exist_normals = data['exist_normals'][0]
        sigmas1 = data["sigmas1"][0]
        bound = self.net_para["TPBEncoder"]["space_scale"]
        normals_mnfld_gt = data['normals_mnfld'][0]  # [B, 3]
        normals_nonmnfld_gt = data['normals_nonmnfld'][0]  # fake in 3d
        normals_near_gt = data['normals_near'][0]  # fake in 3d

        points_mnfld.requires_grad_()
        points_nonmnfld.requires_grad_()
        points_near.requires_grad_()

        # ----sdf value prediction---- #
        pred_mnfld = self.model(points_mnfld)
        pred_nonmnfld = self.model(points_nonmnfld)
        pred_near = self.model(points_near)

        # ----normal vector(grad) prediction---- #
        normals_mnfld = gradient(points_mnfld, pred_mnfld)
        normals_nonmnfld = gradient(points_nonmnfld, pred_nonmnfld)
        normals_near = gradient(points_near, pred_near)

        # ---------Data Fitting--------- #

        loss_mnfld += self.criterion(pred_mnfld, sdfs_mnfld)
        # loss_near += self.criterion(pred_near.abs(), sigmas1)
        # loss_nonmnfld += torch.exp(-coefficient['coef_alpha'] * pred_nonmnfld.abs()).mean()

        if self.provide_sdf:
            loss_nonmnfld += self.criterion(pred_nonmnfld, sdfs_nonmnfld)
            loss_near += self.criterion(pred_near, sdfs_near)

        loss_data += loss_mnfld + loss_near + loss_nonmnfld
        self.loss_dict['loss_mnfld'] = loss_mnfld.clone().detach()
        self.loss_dict['loss_near'] = loss_near.clone().detach()
        self.loss_dict['loss_nonmnfld'] = loss_nonmnfld.clone().detach()
        self.loss_dict['loss_data'] = loss_data.clone().detach()
        loss += coefficient['coef_mnfld'] * loss_mnfld + coefficient['coef_near'] * loss_near + \
                coefficient['coef_nonmnfld'] * loss_nonmnfld

        # ---------Minimal Surface--------- #

        if self.minsurf:
            if self.minsurf_mnfld:
                loss_minsurf_mnfld = minsurf_loss(normals_mnfld, pred_mnfld, coefficient['coef_alpha'])
                loss_minsurf += loss_minsurf_mnfld
                self.loss_dict['loss_minsurf_mnfld'] = loss_minsurf_mnfld.clone().detach()
            if self.minsurf_near:
                loss_minsurf_near = minsurf_loss(normals_near, pred_near, coefficient['coef_alpha'])
                loss_minsurf += loss_minsurf_near
                self.loss_dict['loss_minsurf_near'] = loss_minsurf_near.clone().detach()
            if self.minsurf_far:
                loss_minsurf_nonmnfld = minsurf_loss(normals_nonmnfld, pred_nonmnfld, coefficient['coef_alpha'])
                loss_minsurf += loss_minsurf_nonmnfld
                self.loss_dict['loss_minsurf_nonmnfld'] = loss_minsurf_nonmnfld.clone().detach()
            loss += coefficient['coef_ms'] * loss_minsurf
            self.loss_dict['loss_minsurf'] = loss_minsurf.clone().detach()

        # ---------1st order regularization(normal, eikonal)-------- #

        if self.with_normal and exist_normals:
            cos_sim = nn.CosineSimilarity(dim=-1, eps=1e-6)
            areas_mnfld = 1 - cos_sim(normals_mnfld, normals_mnfld_gt)
            loss_normal = areas_mnfld.abs().mean()
            loss += coefficient['coef_normal'] * loss_normal
            self.loss_dict['loss_normal'] = loss_normal.clone().detach()

        if self.E_contain_mnfld:
            eikonal_mnfld, loss_eikonal_mnfld = eikonal_loss(normals_mnfld)
            loss_eikonal += loss_eikonal_mnfld
            self.loss_dict['loss_eikonal_mnfld'] = loss_eikonal_mnfld.clone().detach()
        if self.E_contain_near:
            eikonal_near, loss_eikonal_near = eikonal_loss(normals_near)
            loss_eikonal += loss_eikonal_near
            self.loss_dict['loss_eikonal_near'] = loss_eikonal_near.clone().detach()
        if self.E_contain_far:
            eikonal_nonmnfld, loss_eikonal_nonmnfld = eikonal_loss(normals_nonmnfld)
            loss_eikonal += loss_eikonal_nonmnfld
            self.loss_dict['loss_eikonal_nonmnfld'] = loss_eikonal_nonmnfld.clone().detach()
        self.loss_dict['loss_eikonal'] = loss_eikonal.clone().detach()
        if self.eikonal:
            loss += coefficient['coef_eikonal'] * loss_eikonal

        # ---------2nd order regularization(except 'project')-------- #

        if 'Non' in self.second_order:
            pass
        else:
            # ----Hessian matrix and its diagonal---- #
            Hessian_mnfld = torch.zeros(points_mnfld.shape[0], points_mnfld.shape[1], points_mnfld.shape[1],
                                        dtype=points_mnfld.dtype, device=self.device)
            laplacian_mnfld = torch.zeros(points_mnfld.shape[0], points_mnfld.shape[1],
                                          dtype=points_mnfld.dtype, device=self.device)
            for i in range(normals_mnfld.shape[1]):
                Hessian_mnfld[:, i, :] += gradient(points_mnfld, normals_mnfld[:, i])
                laplacian_mnfld[:, i] += Hessian_mnfld[:, i, i]
            Hessian_near = torch.zeros(points_near.shape[0], points_near.shape[1], points_near.shape[1],
                                       dtype=points_near.dtype, device=self.device)
            laplacian_near = torch.zeros(points_near.shape[0], points_near.shape[1],
                                         dtype=points_near.dtype, device=self.device)
            for i in range(normals_near.shape[1]):
                Hessian_near[:, i, :] += gradient(points_near, normals_near[:, i])
                laplacian_near[:, i] += Hessian_near[:, i, i]
            Hessian_nonmnfld = torch.zeros(points_nonmnfld.shape[0], points_nonmnfld.shape[1], points_nonmnfld.shape[1],
                                           dtype=points_nonmnfld.dtype, device=self.device)
            laplacian_nonmnfld = torch.zeros(points_nonmnfld.shape[0], points_nonmnfld.shape[1],
                                             dtype=points_nonmnfld.dtype, device=self.device)
            for i in range(normals_nonmnfld.shape[1]):
                Hessian_nonmnfld[:, i, :] += gradient(points_nonmnfld, normals_nonmnfld[:, i])
                laplacian_nonmnfld[:, i] += Hessian_nonmnfld[:, i, i]

            # ----Viscosity + Hessian loss---- #
            if 'vis' in self.second_order:
                if self.div_contain_mnfld:
                    loss_vis_mnfld = viscosity_loss(eikonal_mnfld, laplacian_mnfld, coefficient['coef_vis'])
                    loss_vis += loss_vis_mnfld
                    self.loss_dict['loss_vis_mnfld'] = loss_vis_mnfld.clone().detach()
                    loss += coefficient['coef_proportion'] * coefficient['coef_eikonal'] * loss_vis_mnfld
                if self.div_contain_near:
                    loss_vis_near = viscosity_loss(eikonal_near, laplacian_near, coefficient['coef_vis'])
                    loss_vis += loss_vis_near
                    self.loss_dict['loss_vis_near'] = loss_vis_near.clone().detach()
                    loss += coefficient['coef_eikonal'] * loss_vis_near
                if self.div_contain_far:
                    loss_vis_nonmnfld = viscosity_loss(eikonal_nonmnfld, laplacian_nonmnfld, coefficient['coef_vis'])
                    loss_vis += loss_vis_nonmnfld
                    self.loss_dict['loss_vis_nonmnfld'] = loss_vis_nonmnfld.clone().detach()
                    loss += coefficient['coef_eikonal'] * loss_vis_nonmnfld
                self.loss_dict['loss_vis'] = loss_vis.clone().detach()

            # ----Laplacian loss---- #
            if 'laplacian' in self.second_order:
                if self.div_contain_mnfld:
                    loss_laplacian_mnfld = laplacian_loss(laplacian_mnfld)
                    loss_div += loss_laplacian_mnfld
                    self.loss_dict['loss_div_mnfld'] = loss_laplacian_mnfld.clone().detach()
                if self.div_contain_near:
                    loss_laplacian_near = laplacian_loss(laplacian_near)
                    loss_div += loss_laplacian_near
                    self.loss_dict['loss_div_near'] = loss_laplacian_near.clone().detach()
                if self.div_contain_far:
                    loss_laplacian_nonmnfld = laplacian_loss(laplacian_nonmnfld)
                    loss_div += loss_laplacian_nonmnfld
                    self.loss_dict['loss_div_nonmnfld'] = loss_laplacian_nonmnfld.clone().detach()
                loss += coefficient['coef_div'] * loss_div
                self.loss_dict['loss_div'] = loss_div.clone().detach()

            # ----Hessian loss---- #
            if 'hessian' in self.second_order:
                if self.div_contain_mnfld:
                    loss_hessian_mnfld = hessian_loss(Hessian_mnfld)
                    loss_div += loss_hessian_mnfld
                    self.loss_dict['loss_div_mnfld'] = loss_hessian_mnfld.clone().detach()
                if self.div_contain_near:
                    loss_hessian_near = hessian_loss(Hessian_near)
                    loss_div += loss_hessian_near
                    self.loss_dict['loss_div_near'] = loss_hessian_near.clone().detach()
                if self.div_contain_far:
                    loss_hessian_nonmnfld = hessian_loss(Hessian_nonmnfld)
                    loss_div += loss_hessian_nonmnfld
                    self.loss_dict['loss_div_nonmnfld'] = loss_hessian_nonmnfld.clone().detach()
                loss += coefficient['coef_div'] * loss_div
                self.loss_dict['loss_div'] = loss_div.clone().detach()

            # ----directional div Loss---- #
            if 'ddiv' in self.second_order:
                if self.div_contain_mnfld:
                    loss_ddiv_mnfld = directional_div_loss(points_mnfld, normals_mnfld)
                    loss_div += loss_ddiv_mnfld
                    self.loss_dict['loss_div_mnfld'] = loss_ddiv_mnfld.clone().detach()
                if self.div_contain_near:
                    loss_ddiv_near = directional_div_loss(points_near, normals_near)
                    loss_div += loss_ddiv_near
                    self.loss_dict['loss_div_near'] = loss_ddiv_near.clone().detach()
                if self.div_contain_far:
                    loss_ddiv_nonmnfld = directional_div_loss(points_nonmnfld, normals_nonmnfld)
                    loss_div += loss_ddiv_nonmnfld
                    self.loss_dict['loss_div_nonmnfld'] = loss_ddiv_nonmnfld.clone().detach()
                loss += coefficient['coef_div'] * loss_div
                self.loss_dict['loss_div'] = loss_div.clone().detach()

            # ----Align Gradient-Hessian Loss---- #
            if 'hvp' in self.second_order:
                if self.div_contain_mnfld:
                    loss_hvp_mnfld = alignGH_loss(points_mnfld, normals_mnfld)
                    loss_div += loss_hvp_mnfld
                    self.loss_dict['loss_div_mnfld'] = loss_hvp_mnfld.clone().detach()
                if self.div_contain_near:
                    loss_hvp_near = alignGH_loss(points_near, normals_near)
                    loss_div += loss_hvp_near
                    self.loss_dict['loss_div_near'] = loss_hvp_near.clone().detach()
                if self.div_contain_far:
                    loss_hvp_nonmnfld = alignGH_loss(points_nonmnfld, normals_nonmnfld)
                    loss_div += loss_hvp_nonmnfld
                    self.loss_dict['loss_div_nonmnfld'] = loss_hvp_nonmnfld.clone().detach()
                loss += coefficient['coef_div'] * loss_div
                self.loss_dict['loss_div'] = loss_div.clone().detach()

            # ---- Morse Loss---- #
            if 'morse' in self.second_order:
                if self.div_contain_mnfld:
                    loss_morse_mnfld = morse_loss(Hessian_mnfld)
                    loss_div += loss_morse_mnfld
                    self.loss_dict['loss_div_mnfld'] = loss_morse_mnfld.clone().detach()
                if self.div_contain_near:
                    loss_morse_near = morse_loss(Hessian_near)
                    loss_div += loss_morse_near
                    self.loss_dict['loss_div_near'] = loss_morse_near.clone().detach()
                if self.div_contain_far:
                    loss_morse_nonmnfld = morse_loss(Hessian_nonmnfld)
                    loss_div += loss_morse_nonmnfld
                    self.loss_dict['loss_div_nonmnfld'] = loss_morse_nonmnfld.clone().detach()
                loss += coefficient['coef_div'] * loss_div
                self.loss_dict['loss_div'] = loss_div.clone().detach()

            # ----project Loss---- #
            if 'project' in self.second_order:
                if self.div_contain_mnfld:
                    points_target_mnfld = points_mnfld.clone().detach()
                    loss_project_mnfld = project_loss(points_mnfld, pred_mnfld, normals_mnfld, self.model,
                                                      points_target_mnfld)
                    loss_project += loss_project_mnfld
                    self.loss_dict['loss_project_mnfld'] = loss_project_mnfld.clone().detach()
                if self.div_contain_near:
                    loss_project_near = project_loss(points_near, pred_near, normals_near, self.model,
                                                     points_target_near)
                    loss_project += loss_project_near
                    self.loss_dict['loss_project_near'] = loss_project_near.clone().detach()
                if self.div_contain_far:
                    loss_project_nonmnfld = project_loss(points_nonmnfld, pred_nonmnfld, normals_nonmnfld, self.model,
                                                         points_target_nonmnfld)
                    loss_project += loss_project_nonmnfld
                    self.loss_dict['loss_project_nonmnfld'] = loss_project_nonmnfld.clone().detach()
                loss += coefficient['coef_project'] * loss_project
                self.loss_dict['loss_project'] = loss_project.clone().detach()
        return loss

    def eval_step(self, data):
        points_mnfld = data["points_mnfld"][0]  # [B, 3]
        points_mnfld.requires_grad_()
        sdfs_mnfld = data["sdfs_mnfld"][0]  # [B, 1]
        pred_mnfld = self.model(points_mnfld)
        loss = torch.zeros(1, device=self.device)
        loss += self.criterion(pred_mnfld, sdfs_mnfld)
        return loss

    def prepare_data(self, data):
        if isinstance(data, list):
            for i, v in enumerate(data):
                if isinstance(v, np.ndarray):
                    data[i] = torch.from_numpy(v).to(self.device, non_blocking=True)
                if torch.is_tensor(v):
                    data[i] = v.to(self.device, non_blocking=True)
        elif isinstance(data, dict):
            for k, v in data.items():
                if isinstance(v, np.ndarray):
                    data[k] = torch.from_numpy(v).to(self.device, non_blocking=True)
                if torch.is_tensor(v):
                    data[k] = v.to(self.device, non_blocking=True)
        elif isinstance(data, np.ndarray):
            data = torch.from_numpy(data).to(self.device, non_blocking=True)
        else:  # is_tensor, or other similar objects that has `to`
            data = data.to(self.device, non_blocking=True)

        return data

    def train_one_epoch(self, loader):
        # if isinstance(self.model.encoder, nn.Module):
        #     # self.update_encoder_levels()
        #     active_level = self.model.encoder.active_level
        # else:
        #     active_level = -1
        self.log(f"==> Start Training Epoch {self.epoch}, lr={self.optimizer.param_groups[0]['lr']:.6f}")

        total_loss = 0

        self.model.train()

        from tqdm.auto import tqdm
        pbar = tqdm(total=len(loader) * loader.batch_size,
                    bar_format='{desc}: {percentage:3.0f}% {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')

        self.local_step = 0
        losses = []
        losses_dict = {'losses_mnfld': [], 'losses_nonmnfld': [], 'losses_near': [], 'losses_data': [],
                       'losses_minsurf': [], 'losses_minsurf_mnfld': [], 'losses_minsurf_near': [],
                       'losses_minsurf_nonmnfld': [],
                       'losses_normal': [],
                       'losses_eikonal': [], 'losses_eikonal_mnfld': [], 'losses_eikonal_nonmnfld': [],
                       'losses_eikonal_near': [],
                       'losses_vis': [], 'losses_vis_mnfld': [], 'losses_vis_nonmnfld': [], 'losses_vis_near': [],
                       'losses_div': [], 'losses_div_mnfld': [], 'losses_div_nonmnfld': [], 'losses_div_near': [],
                       'losses_project': [], 'losses_project_mnfld': [], 'losses_project_nonmnfld': [],
                       'losses_project_near': []}

        para_grads = []
        if isinstance(self.model.encoder, nn.Module):
            offsets = self.model.encoder.offsets
        else:
            offsets = None

        for i, data in enumerate(loader):
            if isinstance(self.model.encoder, nn.Module):
                self.update_encoder_levels(i, len(loader))
            self.update_learning_rate(len(loader))
            coefficient = self.update_coefficient(i, len(loader))
            # if isinstance(self.model.encoder, nn.Module):
            #     self.update_encoder_levels(i, len(loader))
            self.local_step += 1
            # self.global_step += 1
            self.global_step = (i + (self.epoch - 1) * len(loader))

            data = self.prepare_data(data)

            self.optimizer.zero_grad()

            if self.optimizer.__class__.__name__ == 'LBFGS':
                def closure():
                    loss = self.train_step(data, coefficient)
                    loss = loss / self.net_para["BasicInfo"]["iter"]
                    loss.backward()
                    return loss

                loss = closure()
                self.optimizer.step(closure)
            else:
                with torch.cuda.amp.autocast(enabled=self.fp16):
                    loss = self.train_step(data, coefficient)
                    loss = loss / self.net_para["BasicInfo"]["iter"]
                self.scaler.scale(loss).backward()
                if self.clip_grad_norm:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 10)
                if (i + 1) % self.net_para["BasicInfo"]["iter"] == 0:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()

            if isinstance(self.model.encoder, nn.Module):
                para_grad = []

                if self.model.encoder.__class__.__name__ == 'tpb_encoder_sp':
                    for j in self.model.encoder.embeddings:
                        para_grad.append(j.grad.mean().detach().cpu().numpy())
                else:
                    embeddings = self.model.encoder.embeddings
                    for j in range(len(offsets) - 1):
                        para_grad.append(embeddings.grad[offsets[j]:offsets[j + 1], :].mean().detach().cpu().numpy())

                para_grads.append(para_grad)

            if self.ema is not None:
                self.ema.update()

            loss_val = loss.item()
            losses.append(loss.item())
            total_loss += loss_val
            losses_dict['losses_mnfld'].append(self.loss_dict['loss_mnfld'].item())
            losses_dict['losses_nonmnfld'].append(self.loss_dict['loss_nonmnfld'].item())
            losses_dict['losses_near'].append(self.loss_dict['loss_near'].item())
            losses_dict['losses_data'].append(self.loss_dict['loss_data'].item())
            losses_dict['losses_minsurf'].append(self.loss_dict['loss_minsurf'].item())
            losses_dict['losses_minsurf_mnfld'].append(self.loss_dict['loss_minsurf_mnfld'].item())
            losses_dict['losses_minsurf_nonmnfld'].append(self.loss_dict['loss_minsurf_nonmnfld'].item())
            losses_dict['losses_minsurf_near'].append(self.loss_dict['loss_minsurf_near'].item())
            losses_dict['losses_normal'].append(self.loss_dict['loss_normal'].item())
            losses_dict['losses_eikonal'].append(self.loss_dict['loss_eikonal'].item())
            losses_dict['losses_eikonal_mnfld'].append(self.loss_dict['loss_eikonal_mnfld'].item())
            losses_dict['losses_eikonal_nonmnfld'].append(self.loss_dict['loss_eikonal_nonmnfld'].item())
            losses_dict['losses_eikonal_near'].append(self.loss_dict['loss_eikonal_near'].item())
            losses_dict['losses_vis'].append(self.loss_dict['loss_vis'].item())
            losses_dict['losses_vis_mnfld'].append(self.loss_dict['loss_vis_mnfld'].item())
            losses_dict['losses_vis_nonmnfld'].append(self.loss_dict['loss_vis_nonmnfld'].item())
            losses_dict['losses_vis_near'].append(self.loss_dict['loss_vis_near'].item())
            losses_dict['losses_div'].append(self.loss_dict['loss_div'].item())
            losses_dict['losses_div_mnfld'].append(self.loss_dict['loss_div_mnfld'].item())
            losses_dict['losses_div_nonmnfld'].append(self.loss_dict['loss_div_nonmnfld'].item())
            losses_dict['losses_div_near'].append(self.loss_dict['loss_div_near'].item())
            losses_dict['losses_project'].append(self.loss_dict['loss_project'].item())
            losses_dict['losses_project_mnfld'].append(self.loss_dict['loss_project_mnfld'].item())
            losses_dict['losses_project_nonmnfld'].append(self.loss_dict['loss_project_nonmnfld'].item())
            losses_dict['losses_project_near'].append(self.loss_dict['loss_project_near'].item())

            if self.use_tensorboardX:
                # ----Loss Global-- #
                self.writer.add_scalar("train/loss", loss_val, self.global_step)
                self.writer.add_scalar("train/normal", self.loss_dict['loss_normal'].item(), self.global_step)
                self.writer.add_scalars("train/loss_data", {'all': self.loss_dict['loss_data'].item(),
                                                            'mnfld': self.loss_dict['loss_mnfld'].item(),
                                                            'near': self.loss_dict['loss_near'].item(),
                                                            'far': self.loss_dict['loss_nonmnfld'].item()},
                                        self.global_step)
                self.writer.add_scalars("train/loss_minsurf", {'all': self.loss_dict['loss_minsurf'].item(),
                                                               'mnfld': self.loss_dict['loss_minsurf_mnfld'].item(),
                                                               'near': self.loss_dict['loss_minsurf_near'].item(),
                                                               'far': self.loss_dict['loss_minsurf_nonmnfld'].item()},
                                        self.global_step)
                self.writer.add_scalars("train/loss_eikonal", {'all': self.loss_dict['loss_eikonal'].item(),
                                                               'mnfld': self.loss_dict['loss_eikonal_mnfld'].item(),
                                                               'near': self.loss_dict['loss_eikonal_near'].item(),
                                                               'far': self.loss_dict['loss_eikonal_nonmnfld'].item()},
                                        self.global_step)
                self.writer.add_scalars("train/loss_vis", {'all': self.loss_dict['loss_vis'].item(),
                                                           'mnfld': self.loss_dict['loss_vis_mnfld'].item(),
                                                           'near': self.loss_dict['loss_vis_near'].item(),
                                                           'far': self.loss_dict['loss_vis_nonmnfld'].item()},
                                        self.global_step)
                self.writer.add_scalars("train/loss_div", {'all': self.loss_dict['loss_div'].item(),
                                                           'mnfld': self.loss_dict['loss_div_mnfld'].item(),
                                                           'near': self.loss_dict['loss_div_near'].item(),
                                                           'far': self.loss_dict['loss_div_nonmnfld'].item()},
                                        self.global_step)
                self.writer.add_scalars("train/loss_project", {'all': self.loss_dict['loss_project'].item(),
                                                               'mnfld': self.loss_dict['loss_project_mnfld'].item(),
                                                               'near': self.loss_dict['loss_project_near'].item(),
                                                               'far': self.loss_dict['loss_project_nonmnfld'].item()},
                                        self.global_step)
                self.writer.add_scalar("train/lr", self.optimizer.param_groups[0]['lr'], self.global_step)

            pbar.set_description(f"loss={loss_val:.4f} ({total_loss / self.local_step:.4f}), "
                                 f"loss_mnfld={self.loss_dict['loss_mnfld'].item():.4f}, "
                                 f"loss_ms={self.loss_dict['loss_minsurf'].item():.4f}, "
                                 # f"loss_project={self.loss_dict['loss_project'].item():.4f}, "
                                 f"loss_E={self.loss_dict['loss_eikonal'].item():.4f}, "
                                 f"loss_vis={self.loss_dict['loss_vis'].item():.4f}, "
                                 f"loss_div={self.loss_dict['loss_div'].item():.4f}, "
                                 f"lr={self.optimizer.param_groups[0]['lr']:.6f}, lr_type={self.lr_type}")
            pbar.update(loader.batch_size)

        average_loss = total_loss / self.local_step
        self.stats["loss"].append(average_loss)

        pbar.close()

        # draw loss reduction chart in each epoch
        loss_dict_save_path = os.path.join(self.workspace, 'validation', f'{self.name}_{self.epoch}_loss_dict.png')
        os.makedirs(os.path.dirname(loss_dict_save_path), exist_ok=True)
        y_loss = np.asarray(losses)
        y_loss_mnfld = np.asarray(losses_dict['losses_mnfld'])
        y_loss_nonmnfld = np.asarray(losses_dict['losses_nonmnfld'])
        y_loss_near = np.asarray(losses_dict['losses_near'])
        y_loss_data = np.asarray(losses_dict['losses_data'])
        y_loss_minsurf = np.asarray(losses_dict['losses_minsurf'])
        y_loss_minsurf_mnfld = np.asarray(losses_dict['losses_minsurf_mnfld'])
        y_loss_minsurf_nonmnfld = np.asarray(losses_dict['losses_minsurf_nonmnfld'])
        y_loss_minsurf_near = np.asarray(losses_dict['losses_minsurf_near'])
        y_loss_normal = np.asarray(losses_dict['losses_normal'])
        y_loss_eikonal = np.asarray(losses_dict['losses_eikonal'])
        y_loss_eikonal_mnfld = np.asarray(losses_dict['losses_eikonal_mnfld'])
        y_loss_eikonal_nonmnfld = np.asarray(losses_dict['losses_eikonal_nonmnfld'])
        y_loss_eikonal_near = np.asarray(losses_dict['losses_eikonal_near'])
        y_loss_vis = np.asarray(losses_dict['losses_vis'])
        y_loss_vis_mnfld = np.asarray(losses_dict['losses_vis_mnfld'])
        y_loss_vis_nonmnfld = np.asarray(losses_dict['losses_vis_nonmnfld'])
        y_loss_vis_near = np.asarray(losses_dict['losses_vis_near'])
        y_loss_div = np.asarray(losses_dict['losses_div'])
        y_loss_div_mnfld = np.asarray(losses_dict['losses_div_mnfld'])
        y_loss_div_nonmnfld = np.asarray(losses_dict['losses_div_nonmnfld'])
        y_loss_div_near = np.asarray(losses_dict['losses_div_near'])
        y_loss_project = np.asarray(losses_dict['losses_project'])
        y_loss_project_mnfld = np.asarray(losses_dict['losses_project_mnfld'])
        y_loss_project_nonmnfld = np.asarray(losses_dict['losses_project_nonmnfld'])
        y_loss_project_near = np.asarray(losses_dict['losses_project_near'])
        x_epoch = np.arange(len(loader))
        fig, ax = plt.subplots(nrows=3, ncols=3, figsize=(20, 20))
        fig.tight_layout(h_pad=2)
        ax = ax.flatten()
        ax[0].cla()
        ax[0].set_xlabel('epoch')
        ax[0].set_ylabel('loss')
        ax[0].scatter(x_epoch, y_loss, c='r', marker='*')
        ax[0].plot(x_epoch, y_loss, c='b')
        ax[0].set_title('Loss_All')
        ax[1].cla()
        ax[1].set_xlabel('epoch')
        ax[1].set_ylabel('loss_data')
        ax[1].scatter(x_epoch, y_loss_data, c='r', marker='*')
        ax[1].plot(x_epoch, y_loss_data, c='b', label='loss_data')
        ax[1].plot(x_epoch, y_loss_mnfld, 'go-', label='loss_mnfld')
        ax[1].plot(x_epoch, y_loss_near, 'y^-', label='loss_near')
        ax[1].plot(x_epoch, y_loss_nonmnfld, 'cs-', label='loss_nonmnfld')
        ax[1].legend()
        ax[1].set_title('Loss_Data')
        ax[2].cla()
        ax[2].set_xlabel('epoch')
        ax[2].set_ylabel('loss_minsurf')
        ax[2].scatter(x_epoch, y_loss_minsurf, c='r', marker='*')
        ax[2].plot(x_epoch, y_loss_minsurf, c='b', label='loss_ms')
        ax[2].plot(x_epoch, y_loss_minsurf_mnfld, 'go-', label='loss_ms_mnfld')
        ax[2].plot(x_epoch, y_loss_minsurf_near, 'y^-', label='loss_ms_near')
        ax[2].plot(x_epoch, y_loss_minsurf_nonmnfld, 'cs-', label='loss_ms_nonmnfld')
        ax[2].legend()
        ax[2].set_title('Loss_Minsurf')
        ax[3].cla()
        ax[3].set_xlabel('epoch')
        ax[3].set_ylabel('loss_normal')
        ax[3].scatter(x_epoch, y_loss_normal, c='r', marker='*')
        ax[3].plot(x_epoch, y_loss_normal, c='b')
        ax[3].set_title('Loss_Normal')
        ax[4].cla()
        ax[4].set_xlabel('epoch')
        ax[4].set_ylabel('loss_eikonal')
        ax[4].scatter(x_epoch, y_loss_eikonal, c='r', marker='*')
        ax[4].plot(x_epoch, y_loss_eikonal, c='b', label='loss_E')
        ax[4].plot(x_epoch, y_loss_eikonal_mnfld, 'go-', label='loss_E_mnfld')
        ax[4].plot(x_epoch, y_loss_eikonal_near, 'y^-', label='loss_E_near')
        ax[4].plot(x_epoch, y_loss_eikonal_nonmnfld, 'cs-', label='loss_E_nonmnfld')
        ax[4].legend()
        ax[4].set_title('Loss_Eikonal')
        ax[5].cla()
        ax[5].set_xlabel('epoch')
        ax[5].set_ylabel('loss_vis')
        ax[5].scatter(x_epoch, y_loss_vis, c='r', marker='*')
        ax[5].plot(x_epoch, y_loss_vis, c='b', label='loss_vis')
        ax[5].plot(x_epoch, y_loss_vis_mnfld, 'go-', label='loss_vis_mnfld')
        ax[5].plot(x_epoch, y_loss_vis_near, 'y^-', label='loss_vis_near')
        ax[5].plot(x_epoch, y_loss_vis_nonmnfld, 'cs-', label='loss_vis_nonmnfld')
        ax[5].legend()
        ax[5].set_title('Loss_Viscosity')
        ax[6].cla()
        ax[6].set_xlabel('epoch')
        ax[6].set_ylabel('loss_div')
        ax[6].scatter(x_epoch, y_loss_div, c='r', marker='*')
        ax[6].plot(x_epoch, y_loss_div, c='b', label='loss_div')
        ax[6].plot(x_epoch, y_loss_div_mnfld, 'go-', label='loss_div_mnfld')
        ax[6].plot(x_epoch, y_loss_div_near, 'y^-', label='loss_div_near')
        ax[6].plot(x_epoch, y_loss_div_nonmnfld, 'cs-', label='loss_div_nonmnfld')
        ax[6].legend()
        ax[6].set_title('Loss_Div')
        ax[7].cla()
        ax[7].set_xlabel('epoch')
        ax[7].set_ylabel('loss_project')
        ax[7].scatter(x_epoch, y_loss_project, c='r', marker='*')
        ax[7].plot(x_epoch, y_loss_project, c='b', label='loss_project')
        ax[7].plot(x_epoch, y_loss_project_mnfld, 'go-', label='loss_project_mnfld')
        ax[7].plot(x_epoch, y_loss_project_near, 'y^-', label='loss_project_near')
        ax[7].plot(x_epoch, y_loss_project_nonmnfld, 'cs-', label='loss_project_nonmnfld')
        ax[7].legend()
        ax[7].set_title('Loss_project_po')

        plt.subplots_adjust(top=0.9, bottom=0.1, left=0.1, right=0.9)
        plt.tight_layout()
        plt.savefig(loss_dict_save_path)
        plt.close(fig)

        if isinstance(self.model.encoder, nn.Module):
            y_encoder_grad = np.asarray(para_grads)
            encoder_grads_save_path = os.path.join(self.workspace, 'validation',
                                                   f'{self.name}_{self.epoch}_e_grads.png')
            os.makedirs(os.path.dirname(encoder_grads_save_path), exist_ok=True)
            num_levels = self.net_para["TPBEncoder"]["num_levels"]
            fig2, ax2 = plt.subplots(nrows=num_levels // 3 + 1, ncols=3, figsize=(20, 20))
            fig2.tight_layout(h_pad=2)
            ax2 = ax2.flatten()
            ax2[num_levels].cla()
            ax2[num_levels].set_xlabel('epoch')
            ax2[num_levels].set_ylabel('encoder_grads_all')
            for j in range(num_levels):
                ax2[j].cla()
                ax2[j].set_xlabel('epoch')
                ax2[j].set_ylabel('encoder_grads_' + str(j))
                ax2[j].plot(x_epoch, y_encoder_grad[:, j], label='e_grad_' + str(j))
                ax2[num_levels].plot(x_epoch, y_encoder_grad[:, j], label='e_grad_' + str(j))
                ax2[j].legend()
                ax2[j].set_title('Encoder_grads_' + str(j))
            ax2[num_levels].legend()
            ax2[num_levels].set_title('Encoder_grads')
            plt.savefig(encoder_grads_save_path)
            plt.close(fig2)

        self.log(f"==> Finished Epoch {self.epoch}.")

    def evaluate_one_epoch(self, loader):
        self.log(f"++> Evaluate at epoch {self.epoch} ...")

        total_loss = 0
        coefficient = self.update_coefficient(self.net_para["BasicInfo"]["batch_num"] - 1,
                                              self.net_para["BasicInfo"]["batch_num"])

        self.model.eval()

        pbar = tqdm.tqdm(total=len(loader) * loader.batch_size,
                         bar_format='{desc}: {percentage:3.0f}% {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')

        with torch.no_grad():
            self.local_step = 0
            for data in loader:
                self.local_step += 1

                data = self.prepare_data(data)

                if self.ema is not None:
                    self.ema.store()
                    self.ema.copy_to()

                with torch.cuda.amp.autocast(enabled=self.fp16):
                    loss = self.eval_step(data)

                if self.ema is not None:
                    self.ema.restore()

                loss_val = loss.item()
                total_loss += loss_val

                pbar.set_description(f"loss={loss_val:.4f} ({total_loss / self.local_step:.4f})")
                pbar.update(loader.batch_size)

        average_loss = total_loss / self.local_step
        self.stats["valid_loss"].append(average_loss)

        pbar.close()
        self.stats["results"].append(average_loss)  # if no metric, choose best by min loss

        coef_mnfld = coefficient['coef_mnfld']
        coef_ms = coefficient['coef_ms']
        alpha = coefficient['coef_alpha']
        coef_eikonal = coefficient['coef_eikonal']
        coef_vis = coefficient['coef_vis']
        coef_div = coefficient['coef_div']
        lr = self.optimizer.param_groups[0]['lr']
        self.log(f'cuurrent_lr:{lr:.6f}\n coefficient_mnfld:{coef_mnfld}, '
                 f'coefficient_minsurf:{coef_ms}, alpha:{alpha}, coefficient_eikonal:{coef_eikonal}, '
                 f'coefficient_vis:{coef_vis}, coefficient_div:{coef_div}')

        self.log(f"++> Evaluate epoch {self.epoch} Finished.")

    def train(self, train_loader, valid_loader, max_epochs):
        # if self.use_tensorboardX:
        #     self.writer = tensorboardX.SummaryWriter(os.path.join(self.workspace, "run", self.name))

        if self.epoch == 0:
            self.log(f'"BasicInfo": {json.dumps(self.net_para["BasicInfo"], indent=4)}')
            self.log(f'"Loss": {json.dumps(self.net_para["Loss"], indent=4)}')
            if not self.use_siren:
                self.log(f'"TPBEncoder": {json.dumps(self.net_para["TPBEncoder"], indent=4)}')
                self.log(f'"MLP": {json.dumps(self.net_para["MLP"], indent=4)}')
            else:
                self.log(f'"input_dim": {self.net_para["BasicInfo"]["input_dim"]}')
                self.log(f'"SIREN":{json.dumps(self.net_para["SIREN"], indent=4)}')

        for epoch in range(self.epoch + 1, max_epochs + 1):
            self.epoch = epoch

            torch.cuda.empty_cache()
            self.train_one_epoch(train_loader)
            if self.workspace is not None:
                self.save_checkpoint(full=True, best=False)

            if self.epoch % self.eval_interval == 0:
                self.evaluate_one_epoch(valid_loader)
                if self.dimension == '1d':
                    self.draw_implicit_function_1d(bbox=train_loader.dataset.bbox,
                                                   a=train_loader.dataset.points_mnfld[0, 0],
                                                   b=train_loader.dataset.points_mnfld[1, 0])
                elif self.dimension == '2d':
                    self.draw_implicit_function_2d(bbox=train_loader.dataset.bbox_model,
                                                   points_gt=train_loader.dataset.points_mnfld)
                elif self.dimension == '3d':
                    if train_loader.dataset.mesh_gt is not None:
                        mesh_gt = train_loader.dataset.mesh_gt.copy()
                    else:
                        mesh_gt = None
                    self.draw_implicit_function_3d(bbox=train_loader.dataset.bbox,
                                                   center=train_loader.dataset.center,
                                                   scale=train_loader.dataset.scale,
                                                   bbox_model=train_loader.dataset.bbox_model,
                                                   resolution=self.net_para['BasicInfo']['grid_res'],
                                                   mesh_gt=mesh_gt,
                                                   normals_gt_all=train_loader.dataset.normals_mnfld)
                self.save_checkpoint(full=False, best=True)

        # if self.use_tensorboardX and self.local_rank == 0:
        if self.use_tensorboardX:
            self.writer.close()

    def evaluate(self, loader):
        self.use_tensorboardX, use_tensorboardX = False, self.use_tensorboardX
        self.evaluate_one_epoch(loader)
        self.use_tensorboardX = use_tensorboardX

    def save_checkpoint(self, full=False, best=False):

        state = {
            'epoch': self.epoch,
            'stats': self.stats,
        }

        if full:
            state['optimizer'] = self.optimizer.state_dict()
            state['scaler'] = self.scaler.state_dict()
            if self.ema is not None:
                state['ema'] = self.ema.state_dict()

        if not best:

            state['model'] = self.model.state_dict()

            file_path = f"{self.ckpt_path}/{self.name}_ep{self.epoch:04d}.pth.tar"

            self.stats["checkpoints"].append(file_path)

            if len(self.stats["checkpoints"]) > self.max_keep_ckpt:
                old_ckpt = self.stats["checkpoints"].pop(0)
                if os.path.exists(old_ckpt):
                    os.remove(old_ckpt)

            torch.save(state, file_path)

        else:
            if len(self.stats["results"]) > 0:
                if self.stats["best_result"] is None or self.stats["results"][-1] < self.stats["best_result"]:
                    self.log(f"[INFO] New best result: {self.stats['best_result']} --> {self.stats['results'][-1]}")
                    self.stats["best_result"] = self.stats["results"][-1]

                    # save ema results
                    if self.ema is not None:
                        self.ema.store()
                        self.ema.copy_to()

                    state['model'] = self.model.state_dict()

                    if self.ema is not None:
                        self.ema.restore()

                    torch.save(state, self.best_path)
            else:
                self.log(f"[WARN] no evaluated results found, skip saving best checkpoint.")

    def load_checkpoint(self, checkpoint=None):
        if checkpoint is None:
            checkpoint_list = sorted(glob.glob(f'{self.ckpt_path}/{self.name}_ep*.pth.tar'))
            if checkpoint_list:
                checkpoint = checkpoint_list[-1]
                self.log(f"[INFO] Latest checkpoint is {checkpoint}")
            else:
                self.log("[WARN] No checkpoint found, model randomly initialized.")
                return

        checkpoint_dict = torch.load(checkpoint, map_location=self.device)

        if 'model' not in checkpoint_dict:
            self.model.load_state_dict(checkpoint_dict)
            self.log("[INFO] loaded model.")
            return

        missing_keys, unexpected_keys = self.model.load_state_dict(checkpoint_dict['model'], strict=False)
        self.log("[INFO] loaded model.")
        if len(missing_keys) > 0:
            self.log(f"[WARN] missing keys: {missing_keys}")
        if len(unexpected_keys) > 0:
            self.log(f"[WARN] unexpected keys: {unexpected_keys}")

        if self.ema is not None and 'ema' in checkpoint_dict:
            self.ema.load_state_dict(checkpoint_dict['ema'])

        self.stats = checkpoint_dict['stats']
        self.epoch = checkpoint_dict['epoch']

        if self.optimizer and 'optimizer' in checkpoint_dict:
            try:
                self.optimizer.load_state_dict(checkpoint_dict['optimizer'])
                self.log("[INFO] loaded optimizer.")
            except:
                self.log("[WARN] Failed to load optimizer, use default.")

        if 'scaler' in checkpoint_dict:
            self.scaler.load_state_dict(checkpoint_dict['scaler'])