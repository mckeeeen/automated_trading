import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
import time
import plotly.plotly as py
import plotly.offline as ol
import plotly.graph_objs as go
import plotly


class RRL(nn.Module):
    '''
    * Optimization: online learning algorithm RRL
    * Rt: additive profits
    * Ut: differential Sharpe Ratio
    * Ft: mono layer neural network:: tanh(wx + b + uFt-1)
    '''

    def __init__(self, m=10, mu=1, delta=0.01, rho=0.01, alpha=0, eta=0.01):
        '''
        - hyper parameter -
        m: numbers to go back in input feature vector
        mu: max securities or shares for trading
        delta: transaction cost
        rho: learning rate
        alpha: L2 normalization parameter
        eta: decresing rate in exponential moving avarage
        '''
        super(RRL, self).__init__()

        self.epoch_count = 0

        self.m = Tensor([m])
        self.mu = Tensor([mu])
        self.delta = Tensor([delta])
        self.rho = Tensor([rho])
        self.alpha = Tensor([alpha])
        self.eta = Tensor([eta])

        self.fc = nn.Linear(int((self.m + 1) + 1 + 1), 1, bias=False)  # bias項は手動で追加

        self.params = list(self.parameters())[0]
        self.grad = list(self.parameters())[0].grad
        self.init_params = self.params.clone()

        self.price_data = Tensor(1)
        self.other_data = Tensor(1)
        self.It_seq = []
        self.It = Tensor(1)
        self.rt_seq = []
        self.rt = Tensor(1)

        self.features = Tensor((self.m + 1) + 1)
        self.X = Tensor(1)
        self.Ft = Tensor(1)
        self.Ft1 = Tensor(1)
        self.dFt_dFt1 = Tensor(1)
        self.Rt = Tensor(1)
        self.At = Tensor(1)
        self.At1 = Tensor(1)
        self.Bt = Tensor([0.01])
        self.Bt1 = Tensor([0.01])
        self.dUt_dRt = Tensor(1)
        self.dRt_dFt = Tensor(1)
        self.dRt_dFt1 = Tensor(1)
        self.thetat = torch.zeros_like(self.params.data)
        self.dFt_dthetat = torch.zeros_like(self.params.data)
        self.DFt_Dthetat = torch.zeros_like(self.params.data)
        self.DFt1_Dthetat1 = torch.zeros_like(self.params.data)
        self.DUt_Dthetat = torch.zeros_like(self.params.data)
        self.delta_thetat = torch.zeros_like(self.params.data)

        self.Ut = Tensor(1)
        self.Ut_seq = []
        self.signal_seq = []
        self.Rt_seq = []
        self.params_seq = [[] for i in range(int(self.m) + 1 + 1 + 1)]
        self.delta_thetat_seq = [[] for i in range(int(self.m) + 1 + 1 + 1)]
        self.At_seq = []
        self.Bt_seq = []
        self.Ft_seq = []

        self.fig = None

    def fit(self, price_data, other_data=None, n_iter=1):

        start = time.time()

        for i in range(n_iter):

            self.epoch_count += 1

            self.price_data = Tensor(1)
            self.other_data = Tensor(1)
            self.It_seq = []
            self.It = Tensor(1)
            self.rt_seq = []
            self.rt = Tensor(1)

            self._input_data(price_data, other_data)

            for t in range(len(self.It_seq)):
                self.rt = self.rt_seq[t]
                self.It = self.It_seq[t]

                self._neural_net()
                self._calculate()
                self._update()
                self._record()
                self._delay()

            if (100 * ((i + 1) / n_iter)) % 10 == 0:
                print('progress... %i %% , elapsed time... %i sec' % (
                int(100 * ((i + 1) / n_iter)), int(time.time() - start)))

    def profit_seq(self):

        return np.cumsum(self.Rt_seq)

    def attempt(self, input):
        '''
        input : torch.Tensor, size = (m+1)
        '''

    def plot_results(self, filename=None):

        trace0 = go.Scatter(y=self.profit_seq())
        trace1 = go.Scatter(y=self.Ut_seq)
        trace2 = go.Scatter(y=self.Ft_seq)
        trace3 = go.Scatter(y=self.signal_seq)

        self.fig = plotly.tools.make_subplots(rows=2, cols=2, subplot_titles=('Profit', 'Utility', 'Trader', 'Signal'))

        self.fig.append_trace(trace0, 1, 1)
        self.fig.append_trace(trace1, 1, 2)
        self.fig.append_trace(trace2, 2, 1)
        self.fig.append_trace(trace3, 2, 2)

        self.fig['layout'].update(title='Results',
                                  xaxis1={'title': 'iterations'},
                                  yaxis1={'title': 'JPY'},
                                  xaxis2={'title': 'iterations'},
                                  yaxis2={'title': '-'},
                                  xaxis3={'title': 'iterations'},
                                  yaxis3={'title': '-'},
                                  xaxis4={'title': 'iterations'},
                                  yaxis4={'title': '-'},
                                  plot_bgcolor='rgba(0,0,0,0)',
                                  paper_bgcolor='rgba(0,0,0,0)',
                                  showlegend=False,
                                  width=800,
                                  height=600)

        if filename:
            ol.plot(self.fig, filename=filename, auto_open=False)
            filename_ = filename.split('/')[-1].split('.')[0]
            print(py.plot(self.fig, filename=filename_, auto_open=True))

        else:
            ol.iplot(self.fig)

    def plot_parameters(self, filename=None):

        cols = int(np.ceil(np.sqrt(len(self.params_seq))))
        rows = int(np.ceil(len(self.params_seq) / cols))
        subplot_titles = [str(i) for i in range(len(self.params_seq))]

        trace = []
        for i in range(len(self.params_seq)):
            trace.append(go.Scatter(y=self.params_seq[i]))

        self.fig = plotly.tools.make_subplots(rows=rows, cols=cols, subplot_titles=subplot_titles)

        index = 0
        for i in range(rows):
            for j in range(cols):
                self.fig.append_trace(trace[index], i + 1, j + 1)
                index += 1
                if index >= len(self.params_seq):
                    break

        self.fig['layout'].update(title='Parameters',
                             plot_bgcolor='rgba(0,0,0,0)',
                             paper_bgcolor='rgba(0,0,0,0)',
                             showlegend=False,
                             width=300 * rows,
                             height=225 * cols)

        if filename:
            ol.plot(self.fig, filename=filename, auto_open=False)
            filename_ = filename.split('/')[-1].split('.')[0]
            print(py.plot(self.fig, filename=filename_, auto_open=True))

        else:
            ol.iplot(self.fig)

    def _input_data(self, price_data, other_data=None):
        '''
        price_data: np.ndarray, shape = (T,)    # T: time span
        other_data: np.ndarray, shape = (N,T)   # N: number of data kinds
        :type other_data: object
        '''

        self.price_data = torch.from_numpy(price_data)
        if other_data:
            self.other_data = torch.from_numpy(other_data)

        _price_diff = torch.from_numpy(np.diff(price_data))

        _price_diff_piece = []
        for i in range(int(self.m + 1), _price_diff.size()[0] + 1):
            self.rt_seq.append(_price_diff[i - 1])
            _price_diff_piece.append(_price_diff[i - int(self.m + 1): i])

        if other_data:
            # TODO priceの系列以外の入力を使うことになったら、あとで書く
            pass

        else:
            self.It_seq = _price_diff_piece

    def _neural_net(self):

        # forward

        self.features = Variable(torch.unsqueeze(torch.cat((self.It.float(), Tensor([1]), self.Ft1.squeeze())),
                                                 0))  # add bias and Ft-1 to input vector and transform to Variable
        self.X = self.fc(self.features)
        self.Ft = F.tanh(self.X)

        # backward

        self.zero_grad()
        self.Ft.backward()

        self.thetat = self.params.data.clone()
        self.dFt_dthetat = self.params.grad.data.clone()

    def _calculate(self):

        self.dFt_dFt1 = list(self.parameters())[0][0][int(self.m) + 1 + 1].data * torch.cosh(self.X.data) ** (-2)
        self.delta_F = (self.Ft.data - self.Ft1).abs()
        self.Rt = self.mu * (self.rt * self.Ft1 - self.delta * self.delta_F)
        self.At = self.At1 + self.eta * (self.Rt - self.At1)
        self.Bt = self.Bt1 + self.eta * (self.Rt ** 2 - self.Bt1)
        self.dUt_dRt = self.eta * (self.Bt1 - self.At1 * self.Rt) / ((self.Bt1 - self.At1 ** 2) ** 1.5)
        self.dRt_dFt = - self.mu * self.delta * torch.sign(self.Ft.data - self.Ft1)
        self.dRt_dFt1 = self.mu * self.rt + self.mu * self.delta * torch.sign(self.Ft.data - self.Ft1)
        self.DFt_Dthetat = self.dFt_dthetat + self.dFt_dFt1 * self.DFt1_Dthetat1
        self.DUt_Dthetat = self.dUt_dRt * (self.dRt_dFt * self.DFt_Dthetat + self.dRt_dFt1 * self.DFt1_Dthetat1)
        self.delta_thetat = self.rho * self.eta * self.DUt_Dthetat - self.alpha * self.thetat

    def _update(self):

        list(self.parameters())[0].data.add_(self.delta_thetat)

    def _record(self):

        self.Ut = self.At / ((((1 - self.eta / 2) / (1 - self.eta)) ** 0.5) * ((self.Bt - self.At ** 2) ** 0.5))
        self.Ut_seq.append(self.Ut.clone()[0][0])
        self.Ft_seq.append(self.Ft.data.clone()[0][0])
        self.signal_seq.append(torch.sign(self.Ft.data.clone())[0][0])
        self.Rt_seq.append(self.Rt.clone()[0][0])

        for i in range(len(self.params_seq)):
            self.params_seq[i].append(self.params.clone().data[0][i])
            self.delta_thetat_seq[i].append(self.delta_thetat.clone()[0][i])

        self.At_seq.append(self.At[0][0])
        self.Bt_seq.append(self.Bt[0][0])

    def _delay(self):

        self.Ft1 = self.Ft.data.clone()
        self.At1 = self.At.clone()
        self.Bt1 = self.Bt.clone()
        self.DFt1_Dthetat1 = self.DFt_Dthetat.clone()
