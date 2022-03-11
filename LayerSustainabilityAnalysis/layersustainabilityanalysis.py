
import os
import random
import time
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F

class LayerSustainabilityAnalysis:
    def __init__(self,pretrained_model):
        self.pretrained_model = pretrained_model
        self.pretrained_model.eval()
        self.selected_probs = ["ReLU", "ELU", "LeakyReLU"] #, "MaxPool2d","Sigmoid""Conv2d",

    def imshow(img):
        print(img.shape)
        img = img / 2 + 0.5     # unnormalize to bring in range [0,1]
        npimg = img.detach().numpy() #(3,M,N)
        print(npimg.shape)
        plt.imshow(np.transpose(npimg, (1, 2, 0)),cmap='gray') #Reshaping as imshow takes image in (M,N,3) format.
        plt.show()

    def np_normalize_output(img):
        if np.size(img)>1:
            img = img - np.min(img)
        img = img / np.max(img)
        return img

    def representation_tensors(self,img_clean, img_perturbed, measure='MSE', verbose=True):
        model = self.pretrained_model

        # working with model variable
        lst_tmp_representations = []
        def hook_fn(m, i, o):
            lst_tmp_representations.append(o)

#     #   l = [module for module in model.modules() if not isinstance(module, nn.Sequential)]
#     #   for layer in l[1:len(l)-2]:#model._modules.items():
#     #        layer.register_forward_hook(hook_fn)

        l = [module for module in model.modules() if not isinstance(module, nn.Sequential)]

        # verbose
        if verbose:
            for layer in l[1:len(l)-2]:#model._modules.items():
                print(layer.__str__())
            print(l)

        # prob layers
        lst_blacklist = self.selected_probs
        for item in model._modules:
            try:
                cnt = 0
                for idx,(name, layer) in enumerate(enumerate(nn.ModuleList(list(model._modules[item])))):
                    if verbose:
                        print(idx,name)
                    if layer.__str__().split('(')[0] in lst_blacklist:
                        if verbose:
                            print(cnt,'=>',layer.__str__())
                        cnt = cnt + 1
                        layer.register_forward_hook(hook_fn)
                        if verbose:
                            print('submit :)', layer.__str__())
            except:
                cnt = 0
                l = [module for module in model.modules() if not isinstance(module, nn.Sequential)]
                for layer in l[0:len(l) ]:  # model._modules.items():
                    if layer.__str__().split('(')[0] in lst_blacklist:
                        if verbose:
                            print(cnt, '=>', layer.__str__())

                        cnt = cnt +1
                        layer.register_forward_hook(hook_fn)
                        if verbose:
                            print('submit :)', layer.__str__())
                break
        # ---------------------------------------------------------
        #       for name,layer in model._modules.items():
        #            layer.register_forward_hook(hook_fn)
        #-----------------------------------------------------------

        # check length of two identical representations
        lst_tmp_representations=[]
        _ = model(img_clean)
        lst_representation_clean = lst_tmp_representations
        if verbose:
            print('visualization len is : {}'.format(len(lst_representation_clean)))

        return lst_tmp_representations

    def representation_comparisons(self,img_clean, img_perturbed, measure='MSE', verbose=True):
        model = self.pretrained_model

        # working with model variable
        lst_tmp_representations = []
        def hook_fn(m, i, o):
            lst_tmp_representations.append(o)

#     #   l = [module for module in model.modules() if not isinstance(module, nn.Sequential)]
#     #   for layer in l[1:len(l)-2]:#model._modules.items():
#     #        layer.register_forward_hook(hook_fn)

        l = [module for module in model.modules() if not isinstance(module, nn.Sequential)]

        # verbose
        if verbose:
            for layer in l[1:len(l)-2]:#model._modules.items():
                print(layer.__str__())
            print(l)

        # prob layers
        lst_blacklist = self.selected_probs
        for item in model._modules:
            try:
                cnt = 0
                for idx,(name, layer) in enumerate(enumerate(nn.ModuleList(list(model._modules[item])))):
                    if verbose:
                        print(idx,name)
                    if layer.__str__().split('(')[0] in lst_blacklist:
                        if verbose:
                            print(cnt,'=>',layer.__str__())
                        cnt = cnt + 1
                        layer.register_forward_hook(hook_fn)
                        if verbose:
                            print('submit :)', layer.__str__())
            except:
                cnt = 0
                l = [module for module in model.modules() if not isinstance(module, nn.Sequential)]
                for layer in l[0:len(l) ]:  # model._modules.items():
                    if layer.__str__().split('(')[0] in lst_blacklist:
                        if verbose:
                            print(cnt, '=>', layer.__str__())

                        cnt = cnt +1
                        layer.register_forward_hook(hook_fn)
                        if verbose:
                            print('submit :)', layer.__str__())
                break
        # ---------------------------------------------------------
        #       for name,layer in model._modules.items():
        #            layer.register_forward_hook(hook_fn)
        #-----------------------------------------------------------

        # check length of two identical representations
        lst_tmp_representations=[]
        _ = model(img_clean)
        lst_representation_clean = lst_tmp_representations
        if verbose:
            print('visualization len is : {}'.format(len(lst_representation_clean)))

        lst_tmp_representations=[]
        _ = model(img_perturbed)
        lst_representation_adv = lst_tmp_representations
        if verbose:
            print('visualization len is : {}'.format(len(lst_representation_adv)))

        lst_comparison_measures = []
        for i in range(len(lst_representation_clean)):
            # extract each layer features
            tensor_featuremap_clean = lst_representation_clean[i].detach().cpu().numpy()
            tensor_featuremap_adv = lst_representation_adv[i].detach().cpu().numpy()
            # calculate comparison measure (for simplicity, as mentioned in paper, we use relative error)
            n_relative_error = np.linalg.norm(tensor_featuremap_clean - tensor_featuremap_adv) / np.linalg.norm(tensor_featuremap_clean)
            measure_value = n_relative_error
            lst_comparison_measures.append(measure_value)

        return lst_comparison_measures


    def plot_representation_cm(plot_title = 'LSA result : Comparison measure of model',
                                 plot_label = 'measure',
                                 plot_style = '-o',
                                 lst_comparison_measures = [],
                                 xlabel = 'layer',
                                 ylabel = 'relative-error',
                                 save_img=None,
                                 fig=None):

            x = np.arange(start=0,stop=len(lst_comparison_measures),step=1)
            y = np.array(lst_comparison_measures)

            plt.plot(x, y, plot_style,label=plot_label)
            plt.xticks(x)
            plt.legend()
            plt.xlabel(xlabel)
            plt.ylabel(ylabel)
            plt.title(plot_title)
            # plt.xlim([0,1])
            plt.ylim([0,2])

            xs, ys = x, y

            # zip joins x and y coordinates in pairs
            for x,y in zip(xs,ys):
                label = "{:.4f}".format(y)
                if label != "0.0000":
                    plt.annotate(label, # this is the text
                             (x,y), # this is the point to label
                             textcoords="offset points", # how to position the text
                             xytext=(0,10), # distance from text to points (x,y)
                             ha='center') # horizontal alignment can be left, right or center
                if save_img:
                    plt.savefig(save_img, bbox_inches='tight')
