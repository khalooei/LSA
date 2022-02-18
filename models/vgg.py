'''VGG11/13/16/19 in Pytorch.'''
import torch
import torch.nn as nn


cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class VGG(nn.Module):
    def __init__(self, vgg_name):
        super(VGG, self).__init__()
        self.features = self._make_layers(cfg[vgg_name])
        self.classifier = nn.Linear(512, 10)
        self._representations = []


    def get_layer_outputs(self, input):
        tmp_out = self.forward(input)
        #         print(len(self._representations))
        return self._representations  # , tmp_out.cpu()

    def kh_vis_init(self):
        """
        visualize layer outputs
        """

        # Visualize feature maps
        # visualisation = []
        def hook_fn(m, i, o):
            # self._representations[m] = o
            # if self._b_visout:
            self._representations.append(o)

        #         for name, layer in self._modules.items():
        #             layer.register_forward_hook(hook_fn)
        #             print('submit :)',name)

        lst_blacklist = ["ReLU", "ELU"]
        # for item in model._modules:
        #     print(item)
        #     for name, layer in enumerate(nn.ModuleList(list(model._modules[item]))):
        #         if layer.__str__().split('(')[0] in lst_blacklist:
        #             layer.register_forward_hook(hook_fn)
        #             print('submit :)', layer.__str__())

        cnt = 0
        l = [module for module in self.modules() if not isinstance(module, nn.Sequential)]
        for layer in l[0:len(l)]:  # model._modules.items():
            if layer.__str__().split('(')[0] in lst_blacklist:
                print(cnt, '=>', layer.__str__())

                cnt = cnt + 1
                layer.register_forward_hook(hook_fn)
                print('submit :)', layer.__str__())
        # print(len(self._modules.items()),len( self._representations))

    def forward(self, x):
        self._representations = []
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]

                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)


def test():
    net = VGG('VGG11')
    x = torch.randn(2,3,32,32)
    y = net(x)
    print(y.size())

# test()
