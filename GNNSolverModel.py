import torch
from torch.nn import Parameter
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.inits import reset, uniform
from torch_geometric.nn import NNConv
import torch.nn as nn
import torch.nn.functional as F

class CplxKernel(torch.nn.Module):
    def __init__(self, inchannel, transchannel, outchannel, ker_width, edgefeasize):
        super(CplxKernel, self).__init__()
        self.onestep1 = onestepKernel(transchannel,  ker_width, edgefeasize)
        self.onestep2 = onestepKernel(transchannel,  ker_width, edgefeasize)
        self.onestep3 = onestepKernel(transchannel,  ker_width, edgefeasize)
        self.onestep4 = onestepKernel(transchannel,  ker_width, edgefeasize)
        self.onestep5 = onestepKernel(transchannel,  ker_width, edgefeasize)
        self.onestep6 = onestepKernel(transchannel,  ker_width, edgefeasize)
        self.onestep7 = onestepKernel(transchannel,  ker_width, edgefeasize)
        self.onestep8 = onestepKernel(transchannel,  ker_width, edgefeasize)
        self.onestep9 = onestepKernel(transchannel,  ker_width, edgefeasize)
        self.onestep10 = onestepKernel(transchannel,  ker_width, edgefeasize)
        self.onestep11 = onestepKernel(transchannel, ker_width, edgefeasize)
        self.onestep12 = onestepKernel(transchannel,  ker_width, edgefeasize)
        self.onestep13 = onestepKernel(transchannel, ker_width, edgefeasize)

        self.ifcup1 = nn.Linear(inchannel, transchannel//4)
        self.iNLup1 = nn.PReLU()
        self.ifcup2 = nn.Linear(transchannel//4, transchannel//2)
        self.iNLup2 = nn.PReLU()
        self.ifcup3 = nn.Linear(transchannel//2, transchannel)

        self.xrfcdown1 = nn.Linear(transchannel, transchannel//2)
        self.xrNLdown1 = nn.PReLU()
        self.xrfcdown2 = nn.Linear(transchannel//2, transchannel//4)
        self.xrNLdown2 = nn.PReLU()
        self.xrfcdown3 = nn.Linear(transchannel//4, outchannel)

        self.xifcdown1 = nn.Linear(transchannel, transchannel//2)
        self.xiNLdown1 = nn.PReLU()
        self.xifcdown2 = nn.Linear(transchannel//2, transchannel//4)
        self.xiNLdown2 = nn.PReLU()
        self.xifcdown3 = nn.Linear(transchannel//4, outchannel)

        self.yrfcdown1 = nn.Linear(transchannel, transchannel//2)
        self.yrNLdown1 = nn.PReLU()
        self.yrfcdown2 = nn.Linear(transchannel//2, transchannel//4)
        self.yrNLdown2 = nn.PReLU()
        self.yrfcdown3 = nn.Linear(transchannel//4, outchannel)

        self.yifcdown1 = nn.Linear(transchannel, transchannel//2)
        self.yiNLdown1 = nn.PReLU()
        self.yifcdown2 = nn.Linear(transchannel//2, transchannel//4)
        self.yiNLdown2 = nn.PReLU()
        self.yifcdown3 = nn.Linear(transchannel//4, outchannel)

        self.zrfcdown1 = nn.Linear(transchannel, transchannel//2)
        self.zrNLdown1 = nn.PReLU()
        self.zrfcdown2 = nn.Linear(transchannel//2, transchannel//4)
        self.zrNLdown2 = nn.PReLU()
        self.zrfcdown3 = nn.Linear(transchannel//4, outchannel)

        self.zifcdown1 = nn.Linear(transchannel, transchannel//2)
        self.ziNLdown1 = nn.PReLU()
        self.zifcdown2 = nn.Linear(transchannel//2, transchannel//4)
        self.ziNLdown2 = nn.PReLU()
        self.zifcdown3 = nn.Linear(transchannel//4, outchannel)



    def forward(self,  xxinc, xxcord, edgeidx, edgeattr):

        inall = torch.cat((xxinc, xxcord), dim=1)

        xr = self.ifcup3(self.iNLup2(self.ifcup2(self.iNLup1(self.ifcup1(inall)))))

        xr = self.onestep1(xr, edgeidx, edgeattr)
        xr = self.onestep2(xr, edgeidx, edgeattr)
        xr = self.onestep3(xr, edgeidx, edgeattr)
        xr = self.onestep4(xr, edgeidx, edgeattr)
        xr = self.onestep5(xr, edgeidx, edgeattr)
        xr = self.onestep6(xr, edgeidx, edgeattr)
        xr = self.onestep7(xr, edgeidx, edgeattr)
        xr = self.onestep8(xr, edgeidx, edgeattr)
        xr = self.onestep9(xr, edgeidx, edgeattr)
        xr = self.onestep10(xr, edgeidx, edgeattr)
        xr = self.onestep11(xr, edgeidx, edgeattr)
        xr = self.onestep12(xr, edgeidx, edgeattr)
        xr = self.onestep13(xr, edgeidx, edgeattr)

        ixr = self.xrfcdown3(self.xrNLdown2(self.xrfcdown2(self.xrNLdown1(self.xrfcdown1(xr)))))
        ixi = self.xifcdown3(self.xiNLdown2(self.xifcdown2(self.xiNLdown1(self.xifcdown1(xr)))))
        iyr = self.yrfcdown3(self.yrNLdown2(self.yrfcdown2(self.yrNLdown1(self.yrfcdown1(xr)))))
        iyi = self.yifcdown3(self.yiNLdown2(self.yifcdown2(self.yiNLdown1(self.yifcdown1(xr)))))
        izr = self.zrfcdown3(self.zrNLdown2(self.zrfcdown2(self.zrNLdown1(self.zrfcdown1(xr)))))
        izi = self.zifcdown3(self.ziNLdown2(self.zifcdown2(self.ziNLdown1(self.zifcdown1(xr)))))

        return torch.cat((ixr, ixi, iyr, iyi, izr, izi), dim=1)

class onestepKernel(torch.nn.Module):
    def __init__(self, transchannel, ker_width, edgefeasize):
        super(onestepKernel, self).__init__()

        self.realaggr = nn.Sequential(nn.Linear(edgefeasize, ker_width), nn.PReLU(),
                                     nn.Linear(ker_width, ker_width), nn.PReLU(),
                                     nn.Linear(ker_width, ker_width), nn.PReLU(),
                                     nn.Linear(ker_width, transchannel**2))
        self.realconv = NNConv(transchannel, transchannel, self.realaggr, aggr='mean')

        self.nonlinearr = nn.PReLU()

    def forward(self, xr, edge_index, edge_attr):
        rr = self.nonlinearr(self.realconv(xr, edge_index, edge_attr))

        return rr
