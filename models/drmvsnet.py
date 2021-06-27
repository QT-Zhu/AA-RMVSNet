import torch
import torch.nn as nn
import torch.nn.functional as F
from .module import *

class FeatNet(nn.Module):
    def __init__(self, gn=True):
        super(FeatNet, self).__init__()
        base_filter = 8

        self.conv0_0 = convgnrelu(3, base_filter , kernel_size=3, stride=1, dilation=1)
        self.conv0_1 = convgnrelu(base_filter, base_filter * 2, kernel_size=3, stride=1, dilation=1)
        self.conv0_2 = convgnrelu(base_filter * 2, base_filter * 2, kernel_size=3, stride=1, dilation=1)
        self.conv0_3 = deformconvgnrelu(base_filter * 2, base_filter * 2, kernel_size=3, stride=1, dilation=1)

        # conv1_2 with conv0_2
        self.conv1_1 = convgnrelu(base_filter * 2, base_filter * 2, kernel_size=3, stride=2, dilation=1)
        self.conv1_2 = deformconvgnrelu(base_filter * 2, base_filter * 1, kernel_size=3, stride=1, dilation=1)

        # conv2_2 with conv0_2
        self.conv2_1 = convgnrelu(base_filter * 2, base_filter * 2, kernel_size=3, stride=2, dilation=1)
        self.conv2_2 = deformconvgnrelu(base_filter * 2, base_filter * 1, kernel_size=3, stride=1, dilation=1)
            

    def forward(self, x):
        
        conv0_0 = self.conv0_0(x)
        conv0_1 = self.conv0_1(conv0_0)
        conv0_2 = self.conv0_2(conv0_1)
        conv0_3 = self.conv0_3(conv0_2)
        
        conv1_1 = self.conv1_1(conv0_2)

        conv1_2 = self.conv1_2(conv1_1)

        conv2_2 = self.conv2_2(self.conv2_1(conv1_1))
        
        #conv3_2 = self.conv3_2(self.conv3_1(conv0_2))
        
        m = nn.Upsample(scale_factor=2, mode='bilinear',align_corners=True)
        conv = torch.cat([conv0_3, m(conv1_2), m(m(conv2_2))], 1)
        
        return conv

# input 3D Feature Volume
class UNetConvLSTM(nn.Module): # input 3D feature volume
    def __init__(self, input_size, input_dim, hidden_dim, kernel_size, num_layers,
                 batch_first=False, bias=True, return_all_layers=False, gn=True):
        super(UNetConvLSTM, self).__init__()

        self._check_kernel_size_consistency(kernel_size)

        # Make sure that both `kernel_size` and `hidden_dim` are lists having len == num_layers
        kernel_size = self._extend_for_multilayer(kernel_size, num_layers)
        hidden_dim  = self._extend_for_multilayer(hidden_dim, num_layers)
        if not len(kernel_size) == len(hidden_dim) == num_layers:
            raise ValueError('Inconsistent list length.')

        self.height, self.width = input_size #feature: height, width)
        self.gn = gn
        print('Training Phase in UNetConvLSTM: {}, {}, gn: {}'.format(self.height, self.width, self.gn))
        self.input_dim  = input_dim # input channel
        self.hidden_dim = hidden_dim # output channel [16, 16, 16, 16, 16, 8]
        self.kernel_size = kernel_size # kernel size  [[3, 3]*5]
        self.num_layers = num_layers # Unet layer size: must be odd
        self.batch_first = batch_first # TRUE
        self.bias = bias #
        self.return_all_layers = return_all_layers

        cell_list = []
        self.down_num = (self.num_layers+1) / 2 
        
        for i in range(0, self.num_layers):
            scale = 2**i if i < self.down_num else 2**(self.num_layers-i-1)
            cell_list.append(ConvLSTMCell(input_size=(int(self.height/scale), int(self.width/scale)),
                                        input_dim=self.input_dim[i],
                                        hidden_dim=self.hidden_dim[i],
                                        kernel_size=self.kernel_size[i],
                                        bias=self.bias))

        self.cell_list = nn.ModuleList(cell_list)
        self.deconv_0 = deConvGnReLU(
            16,
            16, 
            kernel_size=3,
            stride=2,
            padding=1,
            bias=self.bias,
            output_padding=1
            )
        self.deconv_1 = deConvGnReLU(
            16,
            16, 
            kernel_size=3,
            stride=2,
            padding=1,
            bias=self.bias,
            output_padding=1
            )
        self.conv_0 = nn.Conv2d(8, 1, 3, 1, padding=1)

    def forward(self, input_tensor, hidden_state=None, idx = 0, process_sq=True):
        """
        
        Parameters
        ----------
        input_tensor: todo 
            5-D Tensor either of shape (t, b, c, h, w) or (b, t, c, h, w)
        hidden_state: todo
            None. todo implement stateful
            
        Returns
        -------
        last_state_list, layer_output
        """
        if idx ==0 : # input the first layer of input image
           hidden_state = self._init_hidden(batch_size=input_tensor.size(0))

        layer_output_list = []
        last_state_list   = []

        seq_len = input_tensor.size(1)
        
        cur_layer_input = input_tensor
        
        if process_sq:
            
            h0, c0 = hidden_state[0]= self.cell_list[0](input_tensor=cur_layer_input,
                                                cur_state=hidden_state[0])

            h0_1 = nn.MaxPool2d((2, 2), stride=2)(h0)
            h1, c1 = hidden_state[1] = self.cell_list[1](input_tensor=h0_1, 
                                                cur_state=hidden_state[1])

            h1_0 = nn.MaxPool2d((2, 2), stride=2)(h1)  
            h2, c2 = hidden_state[2] = self.cell_list[2](input_tensor=h1_0,
                                                cur_state=hidden_state[2])
            h2_0 = self.deconv_0(h2) # auto reuse

            h2_1 = torch.cat([h2_0, h1], 1)
            h3, c3 = hidden_state[3] = self.cell_list[3](input_tensor=h2_1,
                                                cur_state=hidden_state[3])
            h3_0 = self.deconv_1(h3) # auto reuse
            h3_1 = torch.cat([h3_0, h0], 1)
            h4, c4 = hidden_state[4] = self.cell_list[4](input_tensor=h3_1,
                                                cur_state=hidden_state[4])
            
            cost = self.conv_0(h4) # auto reuse

            return cost, hidden_state
        else:   
            for t in range(seq_len):
                h0, c0 = self.cell_list[0](input_tensor=cur_layer_input[:, t, :, :, :],
                                                    cur_state=hidden_state[0])
                hidden_state[0] = [h0, c0]
                h0_1 = nn.MaxPool2d((2, 2), stride=2)(h0)
                h1, c1 = self.cell_list[1](input_tensor=h0_1, 
                                                    cur_state=hidden_state[1])
                hidden_state[1] = [h1, c1]
                h1_0 = nn.MaxPool2d((2, 2), stride=2)(h1)  
                h2, c2 = self.cell_list[2](input_tensor=h1_0,
                                                    cur_state=hidden_state[2])
                hidden_state[2] = [h2, c2]
                h2_0 = self.deconv_0(h2) # auto reuse

                h2_1 = torch.concat([h2_0, h1], 1)
                h3, c3 = self.cell_list[3](input_tensor=h2_1,
                                                    cur_state=hidden_state[3])
                hidden_state[3] = [h3, c3]
                h3_0 = self.deconv_1(h3) # auto reuse
                h3_1 = torch.concat([h3_0, h0], 1)
                h4, c4 = self.cell_list[4](input_tensor=h3_1,
                                                    cur_state=hidden_state[4])
                hidden_state[4] = [h4, c4]
                
                cost = self.conv_0(h4) # auto reuse
                cost = nn.Tanh(cost)
                # output cost
                layer_output_list.append(cost)

            prob_volume = torch.stack(layer_output_list, dim=1)

            return prob_volume

    def _init_hidden(self, batch_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size))
        return init_states

    @staticmethod
    def _check_kernel_size_consistency(kernel_size):
        if not (isinstance(kernel_size, tuple) or
                    (isinstance(kernel_size, list) and all([isinstance(elem, tuple) for elem in kernel_size]))):
            raise ValueError('`kernel_size` must be tuple or list of tuples')

    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        if not isinstance(param, list):
            param = [param] * num_layers
        return param

class DrMVSNet(nn.Module):
    def __init__(self, image_scale=0.25, max_h=960, max_w=480, return_depth=False):

        super(DrMVSNet,self).__init__()
        self.gn = True
        self.feature = FeatNet(gn=self.gn)
        input_size = (int(max_h * image_scale), int(max_w * image_scale))  # height, width

        input_dim = [32, 16, 16, 32, 32]
        hidden_dim = [16, 16, 16, 16, 8]
        num_layers = 5
        kernel_size = [(3, 3) for _ in range(num_layers)]

        self.cost_regularization = UNetConvLSTM(input_size, input_dim, hidden_dim, kernel_size, num_layers,
                                                batch_first=False, bias=True, return_all_layers=False, gn=self.gn)
        self.gatenet = gatenet(self.gn, 32)

        self.return_depth = return_depth

    def forward(self, imgs, proj_matrices, depth_values):
        imgs = torch.unbind(imgs, 1)
        proj_matrices = torch.unbind(proj_matrices, 1)
        assert len(imgs) == len(proj_matrices), "Different number of images and projection matrices"

        num_depth = depth_values.shape[1]

        # in: images; out: 32-channel feature maps
        features = [self.feature(img) for img in imgs]
        ref_feature, src_features = features[0], features[1:]
        ref_proj, src_projs = proj_matrices[0], proj_matrices[1:]

        # Recurrent process i-th depth layer
        cost_reg_list = []
        hidden_state = None
        
        
        if not self.return_depth:  # Training Phase;
            for d in range(num_depth):           
                ref_volume = ref_feature
                warped_volumes = None
                for src_fea, src_proj in zip(src_features, src_projs):
                    warped_volume = homo_warping_depthwise(src_fea, src_proj, ref_proj, depth_values[:, d])
                    warped_volume = (warped_volume - ref_volume).pow_(2)
                    reweight = self.gatenet(warped_volume)  
                    if warped_volumes is None:
                        warped_volumes = (reweight + 1) * warped_volume
                    else:
                        warped_volumes = warped_volumes + (reweight + 1) * warped_volume

                volume_variance = warped_volumes / len(src_features)
                cost_reg, hidden_state = self.cost_regularization(-1 * volume_variance, hidden_state, d)
                cost_reg_list.append(cost_reg)
                
            prob_volume = torch.stack(cost_reg_list, dim=1).squeeze(2)
            prob_volume = F.softmax(prob_volume,dim=1)  # get prob volume use for recurrent to decrease memory consumption

            return {'prob_volume': prob_volume}
            
        else: #Test phase
            shape = ref_feature.shape
            depth_image = torch.zeros(shape[0], shape[2], shape[3]).cuda()  # B * H * W
            max_prob_image = torch.zeros(shape[0], shape[2], shape[3]).cuda()
            exp_sum = torch.zeros(shape[0], shape[2], shape[3]).cuda()

            for d in range(num_depth):
                ref_volume = ref_feature
                warped_volumes = None
                for src_fea, src_proj in zip(src_features, src_projs):
                    warped_volume = homo_warping_depthwise(src_fea, src_proj, ref_proj, depth_values[:, d])
                    warped_volume = (warped_volume - ref_volume).pow_(2)
                    reweight = self.gatenet(warped_volume)  # saliency

                    if warped_volumes is None:
                        warped_volumes = (reweight + 1) * warped_volume
                    else:
                        warped_volumes = warped_volumes + (reweight + 1) * warped_volume
                            
                    volume_variance = warped_volumes / len(src_features)

                cost_reg, hidden_state = self.cost_regularization(-1 * volume_variance, hidden_state, d)
                cost_reg_list.append(cost_reg)
                
            
            for d in range(num_depth):
                prob = torch.exp(cost_reg_list[d].squeeze(1))
                exp_sum=exp_sum + prob


            prob_sum = torch.zeros(shape[0], shape[2], shape[3]).cuda()

            
            for d in range( num_depth ):
            
                prob = torch.exp(cost_reg_list[d].squeeze(1))
            
                prob = prob/exp_sum

                depth = depth_values[:, d]  # B
                temp_depth_image = depth.view(shape[0], 1, 1).repeat(1, shape[2], shape[3])
                update_flag_image = (max_prob_image < prob).type(torch.float)

                new_max_prob_image = torch.mul(update_flag_image, prob) + torch.mul(1 - update_flag_image,
                                                                                    max_prob_image)
                new_depth_image = torch.mul(update_flag_image, temp_depth_image) + torch.mul(1 - update_flag_image,
                                                                                             depth_image)
                max_prob_image = new_max_prob_image
                depth_image = new_depth_image
                
                prob_sum = prob_sum + prob 

            forward_exp_sum = prob_sum  # + 1e-7

            conf = max_prob_image / forward_exp_sum

            return {"depth": depth_image, "photometric_confidence": conf}

def mvsnet_cls_loss(prob_volume, depth_gt, mask, depth_value, return_prob_map=False): #depth_num, depth_start, depth_interval):
    # depth_value: B * NUM
    # get depth mask
    mask_true = mask 
    valid_pixel_num = torch.sum(mask_true, dim=[1,2]) + 1e-6

    shape = depth_gt.shape

    depth_num = depth_value.shape[-1]
    depth_value_mat = depth_value.repeat(shape[1], shape[2], 1, 1).permute(2,3,0,1)
   
    gt_index_image = torch.argmin(torch.abs(depth_value_mat-depth_gt.unsqueeze(1)), dim=1)

    gt_index_image = torch.mul(mask_true, gt_index_image.type(torch.float))
    gt_index_image = torch.round(gt_index_image).type(torch.long).unsqueeze(1) # B, 1, H, W
 
    # gt index map -> gt one hot volume (B x 1 x H x W )
    gt_index_volume = torch.zeros(shape[0], depth_num, shape[1], shape[2]).type(mask_true.type()).scatter_(1, gt_index_image, 1)
    # print('shape:', gt_index_volume.shape, )
    # cross entropy image (B x D X H x W)
    cross_entropy_image = -torch.sum(gt_index_volume * torch.log(prob_volume), dim=1).squeeze(1) # B, 1, H, W
    #print('cross_entropy_image', cross_entropy_image)
    # masked cross entropy loss
    masked_cross_entropy_image = torch.mul(mask_true, cross_entropy_image) # valid pixel
    masked_cross_entropy = torch.sum(masked_cross_entropy_image, dim=[1, 2])

    masked_cross_entropy = torch.mean(masked_cross_entropy / valid_pixel_num) # Origin use sum : aggregate with batch
    # winner-take-all depth map
    wta_index_map = torch.argmax(prob_volume, dim=1, keepdim=True).type(torch.long)
    wta_depth_map = torch.gather(depth_value_mat, 1, wta_index_map).squeeze(1)

    if return_prob_map:
        photometric_confidence = torch.max(prob_volume, dim=1)[0] # output shape dimension B * H * W
        return masked_cross_entropy, wta_depth_map, photometric_confidence
    return masked_cross_entropy, wta_depth_map








