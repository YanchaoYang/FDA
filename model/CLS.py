import torch.nn as nn
import torch.nn.functional as F
import torch

class Classifier(nn.Module):
    def __init__(self, input_nc=3, ndf=64, norm_layer=nn.BatchNorm2d):
        super(Classifier, self).__init__()
        kw = 3
        sequence = [
            nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2),
            nn.LeakyReLU(0.2, True)
        ]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(3):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                          kernel_size=kw, stride=2),
                norm_layer(ndf * nf_mult, affine=True),
                nn.LeakyReLU(0.2, True)
            ]
        self.before_linear = nn.Sequential(*sequence)
        
        sequence = [
            nn.Linear(ndf * nf_mult, 1024),
            nn.Linear(1024, 10)
        ]
        self.after_linear = nn.Sequential(*sequence)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, 0.01)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        self.criterionCLS = torch.nn.modules.CrossEntropyLoss()

    def forward(self, x, lbl=None, ita=1.5):
        bs = x.size(0)
        out = self.after_linear(self.before_linear(x).view(bs, -1))
        x = out

        P = F.softmax(x, dim=1)        # [B, 19, H, W]
        logP = F.log_softmax(x, dim=1) # [B, 19, H, W]
        PlogP = P * logP               # [B, 19, H, W]
        ent = -1.0 * PlogP.sum(dim=1)  # [B, 1, H, W]
        ent = ent / 2.9444         # chanage when classes is not 19
        # compute robust entropy
        ent = ent ** 2.0 + 1e-8
        ent = ent ** ita
        self.loss_ent = ent.mean()

        if lbl is not None:
            self.loss_cls = self.criterionCLS(x, lbl)

        return x

    def get_lr_params(self):
        b = []
        b.append(self.before_linear.parameters())
        b.append(self.after_linear.parameters())
        for j in range(len(b)):
            for i in b[j]:
                yield i

    def optim_parameters(self, args):
        return [{'params': self.get_lr_params(), 'lr': args.learning_rate}]

    def adjust_learning_rate(self, args, optimizer, i):
        lr = args.learning_rate * (  (1-float(i)/args.num_steps) ** (args.power)  )
        optimizer.param_groups[0]['lr'] = lr
        if len(optimizer.param_groups) > 1:
            optimizer.param_groups[1]['lr'] = lr * 10  

def CLSNet(restore_from=None):
    model = Classifier()
    if restore_from is not None: 
        model.load_state_dict(torch.load(restore_from + '.pth', map_location=lambda storage, loc: storage))        
    return model

