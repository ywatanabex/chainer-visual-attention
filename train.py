"""
chainer implementation of "Recurrent Models of Visual Attention" by Mnih et al.
(http://arxiv.org/abs/1406.6247)

This model sequentially crop/attend small part of input image and classify the image
according to the sequence of small images.
The crop/attend action is not differentiable.
Therefore the whole model should be trained with REINFORCE algorithm.
"""
import argparse
import os

import numpy as np

import chainer
import chainer.cuda as cuda
from chainer import training
from chainer.training import extensions
from chainer import reporter
import chainer.functions as F
import chainer.links as L


def crop_batch_images(image, coords, scale):
    """
    Crop a batch of images around the given coordinate.
    If the coordinate is close to boundary, the protruded area is padded with zero.

    image : shape == (batchsize, height, width)
    x, y : shape == (batchsize,),  float (0 to 1)

    cropped_image : shape == (batchsize, height * scale, width * scale)
    """
    xp = cuda.get_array_module(image)
    batchsize, height0, width0 = image.shape
    height1 = int(np.round(scale * height0))  # height of cropped region
    width1 = int(np.round(scale * width0))
    cropped_image = xp.zeros((batchsize, height1, width1)).astype(np.float32)
    for k, (x, y) in enumerate(coords):
        i_min = int(np.round(float(x) * height0 - height1 * 0.5))
        j_min = int(np.round(float(y) * width0 - width1 * 0.5))
        i_max = i_min + height1
        j_max = j_min + width1
        cropped_image[k, max(-i_min, 0): min(height0-i_min, height1), max(-j_min, 0): min(width0-j_min, width1)] =\
            image[k, max(i_min, 0):min(i_max, height0), max(j_min, 0):min(j_max, width0)]
    return cropped_image


def chop_01(a):
    shape = a.shape
    return np.max([np.zeros(shape).astype(np.float32), np.min([np.ones(shape).astype(np.float32), a], axis=0)], axis=0)


class RAM0(chainer.Chain):
    """
    Neural Network part of Recurrent Attention Model
    implemented with MLP and LSTM.
    """
    def __init__(self, n_units, n_hidden, n_in, n_class, sigma, train=True):
        super(RAM0, self).__init__(
            ll1=L.Linear(2, n_units),          # 2 is x,y coordinate dimension
            lrho1=L.Linear(n_in, n_units),
            lh1=L.Linear(n_units * 2, n_hidden),
            lh2=L.Linear(n_hidden, n_hidden),
            lstm=L.LSTM(n_hidden, n_hidden),
            lc=L.Linear(n_hidden, n_class),    # class output
            ll=L.Linear(n_hidden, 2),          # location output
            lb=L.Linear(n_hidden, 1),          # baseline output
            )
        self.sigma = sigma
        self.train = train

    def reset_state(self):
        self.lstm.reset_state()

    def __call__(self, l, rho):
        """
        :param l: center of cropping (from 0 to 1)
        :param rho: cropped image
        """
        g0 = F.relu(self.ll1(l))
        g1 = F.relu(self.lrho1(rho))
        h0 = F.concat([g0, g1], axis=1)
        h1 = F.relu(self.lh1(h0))
        h2 = F.relu(self.lh2(h1))
        # lstm
        h_out = self.lstm(h2)
        self.h_out = h_out
        c = self.lc(h_out)
        l_out = F.sigmoid(self.ll(h_out))
        return c, l_out

    def sample_location(self, l_data):
        """
        sample new location from center l_data
        """
        if self.train:
            batchsize = l_data.shape[0]
            xp = cuda.get_array_module(l_data)
            l_sampled = chop_01(np.random.normal(scale=self.sigma, size=(batchsize, 2)).astype(np.float32)
                                + cuda.to_cpu(l_data))
            return xp.array(l_sampled)
        else:
            return l_data  # do not sample location in test phase

    def get_location_loss(self, l, mean_data):
        mean = chainer.Variable(mean_data, volatile=chainer.flag.AUTO)
        term1 = 0.5 * (l - mean)**2 * self.sigma**-2
        term2 = 0.5 * np.log(2 * np.pi * self.sigma**2)
        return F.sum(term1 + term2, axis=1)


class ReinforceClassifier(chainer.Chain):
    def __init__(self, ram, len_seq, scale, baseline, coeff=1.0, coeffb=1.0, train=True):
        super(ReinforceClassifier, self).__init__(ram=ram)
        self.scale = scale  # crop scale
        self.len_seq = len_seq
        self.baseline = baseline
        self.coeff = coeff
        self.coeffb = coeffb
        self._train = train

    @property
    def train(self):
        return self._train

    @train.setter
    def train(self, value):
        self._train = value
        self.ram.train = value

    def __call__(self, x, t):
        batchsize = x.shape[0]
        xp = cuda.get_array_module(x.data)
        self.ram.reset_state()
        l = chainer.Variable(xp.zeros((batchsize, 2)).astype(np.float32)+0.5, volatile=chainer.flag.AUTO)  # start from center
        loss_R0 = 0
        self.l_list = [l]

        for s in range(self.len_seq):
            # sample location
            l_sampled = self.ram.sample_location(l.data)
            rho = chainer.Variable(crop_batch_images(x.data.reshape(-1, 28, 28), l_sampled, self.scale),
                                   volatile=chainer.flag.AUTO)
            # REINFORCE loss
            if s < self.len_seq - 1:
                loss_R0 = self.ram.get_location_loss(l, l_sampled)
            # compute next step
            ls = chainer.Variable(xp.array(l_sampled), volatile=chainer.flag.AUTO)
            c, l = self.ram(ls, rho)
            if not self.train:
                self.l_list.append(l)

        self.y = F.argmax(c, axis=1)
        loss_c = F.softmax_cross_entropy(c, t)  # scalar
        accuracy = chainer.functions.accuracy(c, t)
        reporter.report({'accuracy': accuracy}, self)

        loss_c0 = - F.log(F.select_item(F.softmax(c), t))  # shape == (batchsize,)
        # assertion: F.sum(loss_c0.data).data / batchsize == loss_c.data
        # TODO: use aggragation option of softmax_cross_entropy if chainer is updated
        if self.baseline == 'adaptive':
            b = F.reshape(self.ram.lb(self.ram.h_out), (-1,))
        elif self.baseline == 'mean':
            b = F.sum(loss_c0) / batchsize
        elif self.baseline == 'zero':
            b = 0
        else:
            raise ValueError('Unknown baseline mode')
        penalty = (loss_c0.data - b.data)
        loss_r = F.sum(loss_R0 * chainer.Variable(penalty, volatile=chainer.flag.AUTO)) / batchsize
        loss_total = loss_c + self.coeff * loss_r
        if self.baseline == 'adaptive':
            loss_b = F.sum((chainer.Variable(loss_c0.data, volatile=chainer.flag.AUTO) - b)**2) / batchsize  # MSE between
            loss_total += loss_b
        reporter.report({'c': loss_c}, self)
        reporter.report({'r': loss_r}, self)
        reporter.report({'loss': loss_total}, self)

        return loss_total


class ReinforceClassifier1(ReinforceClassifier):
    """
    Use accuracy instead of log-likelihood for reward.
    """
    def __init__(self, ram, len_seq, scale, baseline, coeff=1.0, coeffb=1.0, train=True):
        super(ReinforceClassifier1, self).__init__(ram, len_seq, scale, baseline, coeff, coeffb, train)

    def __call__(self, x, t):
        batchsize = x.shape[0]
        xp = cuda.get_array_module(x.data)
        self.ram.reset_state()
        l = chainer.Variable(xp.zeros((batchsize, 2)).astype(np.float32)+0.5, volatile=chainer.flag.AUTO)  # start from center
        loss_R0 = 0
        self.l_list = [l]

        for s in range(self.len_seq):
            # sample location
            l_sampled = self.ram.sample_location(l.data)
            rho = chainer.Variable(crop_batch_images(x.data.reshape(-1, 28, 28), l_sampled, self.scale),
                                   volatile=chainer.flag.AUTO)
            # REINFORCE loss
            if s < self.len_seq - 1:
                loss_R0 += self.ram.get_location_loss(l, l_sampled)
            # compute next step
            ls = chainer.Variable(xp.array(l_sampled), volatile=chainer.flag.AUTO)
            c, l = self.ram(ls, rho)
            if not self.train:
                self.l_list.append(l)

        loss_c = F.softmax_cross_entropy(c, t)  # scalar
        accuracy = chainer.functions.accuracy(c, t)
        reporter.report({'accuracy': accuracy}, self)

        # REINFORCE penalty
        self.y = F.argmax(c, axis=1)
        miss = 1 - (self.y.data == t.data).astype(c.data.dtype)
        if self.baseline == 'adaptive':
            b = self.ram.lb(self.ram.h_out).data.reshape((-1,))
        elif self.baseline == 'mean':
            b = miss.sum() / batchsize
        elif self.baseline == 'zero':
            b = 0
        else:
            raise ValueError('Unknown baseline mode')
        penalty = (miss - b)
        loss_r = F.sum(loss_R0 * chainer.Variable(penalty, volatile=chainer.flag.AUTO)) / batchsize
        loss_total = loss_c + self.coeff * loss_r
        if self.baseline == 'adaptive':
            loss_b = F.sum((chainer.Variable(miss, volatile=chainer.flag.AUTO) - b)**2) / batchsize  # MSE between
            loss_total += loss_b
        reporter.report({'c': loss_c}, self)
        reporter.report({'r': loss_r}, self)
        reporter.report({'loss': loss_total}, self)

        return loss_total


class ReinforceClassifier2(ReinforceClassifier):
    """
    Reinforce Classifier with accumulative classification loss.

    The model is trained to return correct class label at all step.
    """
    def __init__(self, ram, len_seq, scale, baseline, coeff=1.0, coeffb=1.0, train=True):
        super(ReinforceClassifier2, self).__init__(ram, len_seq, scale, baseline, coeff, coeffb, train)

    def __call__(self, x, t):
        batchsize = x.shape[0]
        xp = cuda.get_array_module(x.data)
        self.ram.reset_state()
        l = chainer.Variable(xp.zeros((batchsize, 2)).astype(np.float32) + 0.5,
                             volatile=chainer.flag.AUTO)  # start from center
        loss_R0_list  = []
        loss_c0_list = []
        for s in range(self.len_seq):
            # sample location
            l_sampled = self.ram.sample_location(l.data)
            rho = chainer.Variable(crop_batch_images(x.data.reshape(-1, 28, 28), l_sampled, self.scale),
                                   volatile=chainer.flag.AUTO)
            # REINFORCE loss
            loss_R0_list.append(self.ram.get_location_loss(l, l_sampled))

            # compute next step
            ls = chainer.Variable(xp.array(l_sampled), volatile=chainer.flag.AUTO)
            c, l = self.ram(ls, rho)
            # classification loss
            loss_c0_list.append(- F.log(F.select_item(F.softmax(c), t)))  # shape == (batchsize,)

        loss_c = F.sum(sum(loss_c0_list)) / batchsize
        accuracy = chainer.functions.accuracy(c, t)
        reporter.report({'accuracy': accuracy}, self)

        loss_c1_list = [sum(loss_c0_list[i+1:]) for i in range(self.len_seq-1)]
        b_list = [F.sum(lc) / batchsize for lc in loss_c1_list]

        loss_R = sum([lp * (pen.data - b.data) for (lp, pen, b) in zip(loss_R0_list, loss_c1_list, b_list)])
        loss_r = F.sum(loss_R) / batchsize

        loss_total = loss_c + self.coeff * loss_r
        reporter.report({'c': loss_c}, self)
        reporter.report({'r': loss_r}, self)
        reporter.report({'loss': loss_total}, self)

        return loss_total


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Chainer example: MNIST')
    parser.add_argument('--batchsize', '-b', type=int, default=500,
                        help='Number of images in each mini-batch')
    parser.add_argument('--epoch', '-e', type=int, default=1000,
                        help='Number of sweeps over the dataset to train')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--out', '-o', default='result',
                        help='Directory to output the result')
    parser.add_argument('--alg', default='rc', choices=['rc', 'rc1', 'rc2'],
                        help='Training algorithm')
    parser.add_argument('--resume', '-r', default='',
                        help='Resume the training from snapshot')
    parser.add_argument('--unit', '-u', type=int, default=128,
                        help='Dimension of locator, glimplse hidden state')
    parser.add_argument('--hidden', type=int, default=256,
                        help='Dimension of lstm hidden state')
    parser.add_argument('--len_seq', '-l', type=int, default=6,
                        help='Length of action sequence')
    parser.add_argument('--scale', '-s', type=float, default=0.3,
                        help='Scale of cropped image (0 to 1)')
    parser.add_argument('--sigma', type=float, default=0.1,
                        help='sigma of location sampling model')
    parser.add_argument('--coeff', type=float, default=1.0,
                        help='coefficient of reinforce objective')
    parser.add_argument('--opt', type=str, default='adam', choices=['sgd', 'adam'],
                        help='coefficient of reinforce objective')
    parser.add_argument('--baseline', type=str, default='adaptive', choices=['adaptive', 'mean', 'zero'],
                        help='coefficient of reinforce objective')
    parser.add_argument('--coeffb', type=float, default=1.0,
                        help='coefficient of adaptive baseline objective')
    parser.add_argument('--eval', type=str,
                        help='Evaluation mode: path to saved model file')
    args = parser.parse_args()

    print('GPU: {}'.format(args.gpu))
    print('# n_units: {}'.format(args.unit))
    print('# n_hidden: {}'.format(args.hidden))
    print('# Length of action sequence: {}'.format(args.len_seq))
    print('# Reinforce Algorithm: {}'.format(args.alg))
    print('# scale: {}'.format(args.scale))
    print('# sigma: {}'.format(args.sigma))
    print('# baseline mode: {}'.format(args.baseline))
    print('# loss coefficient: {}'.format(args.coeff))
    print('# baseline loss coefficient: {}'.format(args.coeffb))
    print('# Optimization algorithm: {}'.format(args.opt))
    print('# Minibatch-size: {}'.format(args.batchsize))
    print('# epoch: {}'.format(args.epoch))
    print('')
    save_path = 'alg={},unit={},hidden={},len={},scale={},sigma={},coeff={},opt={},baseline={},coeffb={}'.format(
        args.alg, args.unit, args.hidden, args.len_seq, args.scale, args.sigma, args.coeff, args.opt, args.baseline,
        args.coeffb)

    n_in = int(np.round(28 * args.scale)**2)
    ram = RAM0(args.unit, args.hidden, n_in, n_class=10, sigma=args.sigma)
    if args.alg == 'rc':
        model = ReinforceClassifier(ram, len_seq=args.len_seq, scale=args.scale, coeff=args.coeff,
                                    baseline=args.baseline, coeffb=args.coeffb)
    elif args.alg == 'rc1':
        model = ReinforceClassifier1(ram, len_seq=args.len_seq, scale=args.scale, coeff=args.coeff,
                                     baseline=args.baseline, coeffb=args.coeffb)
    elif args.alg == 'rc2':
        model = ReinforceClassifier2(ram, len_seq=args.len_seq, scale=args.scale, coeff=args.coeff,
                                     baseline=args.baseline, coeffb=args.coeffb)
    if args.gpu >= 0:
        chainer.cuda.get_device(args.gpu).use()  # Make a specified GPU current
        model.to_gpu()  # Copy the model to the GPU

    # Setup an optimizer
    if args.opt == 'sgd':
        optimizer = chainer.optimizers.SGD(lr=0.01)
    elif args.opt == 'adam':
        optimizer = chainer.optimizers.Adam()
    else:
        raise ValueError('Unknown optimizatin algorithm')
    optimizer.setup(model)
    optimizer.add_hook(chainer.optimizer.GradientClipping(5))

    # Load the MNIST dataset
    train, test = chainer.datasets.get_mnist()
    train_iter = chainer.iterators.SerialIterator(train, args.batchsize)
    test_iter = chainer.iterators.SerialIterator(test, args.batchsize,
                                                 repeat=False, shuffle=False)

    # Set up a trainer
    updater = training.StandardUpdater(train_iter, optimizer, device=args.gpu)
    trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=os.path.join(args.out, save_path))

    # Evaluate the model with the test dataset for each epoch
    test_model = model.copy()
    test_model.train = False
    trainer.extend(extensions.Evaluator(test_iter, model, device=args.gpu), name='val')
    trainer.extend(extensions.Evaluator(test_iter, test_model, device=args.gpu), name='test')

    # Dump a computational graph from 'loss' variable at the first iteration
    # The "main" refers to the target link of the "main" optimizer.
    trainer.extend(extensions.dump_graph('main/loss'))

    # Take a snapshot at each epoch
    trainer.extend(extensions.snapshot(), trigger=(5, 'epoch'))

    # Write a log of evaluation statistics for each epoch
    trainer.extend(extensions.LogReport())

    # Print selected entries of the log to stdout
    # Here "main" refers to the target link of the "main" optimizer again, and
    # "validation" refers to the default name of the Evaluator extension.
    # Entries other than 'epoch' are reported by the Classifier link, called by
    # either the updater or the evaluator.
    trainer.extend(extensions.PrintReport(
        ['epoch', 'main/c', 'main/r', 'main/loss', 'main/accuracy',
         'val/main/c', 'val/main/r', 'val/main/loss', 'val/main/accuracy',
         'test/main/c', 'test/main/r', 'test/main/loss', 'test/main/accuracy']))

    # Print a progress bar to stdout
    trainer.extend(extensions.ProgressBar())

    if args.resume:
        # Resume from a snapshot
        chainer.serializers.load_npz(args.resume, trainer)

    if not args.eval:
        # Run the training
        trainer.run()
    else:
        chainer.serializers.load_npz(args.eval, trainer)
        import matplotlib.pyplot as plt
        from chainer.dataset.convert import concat_examples
        save_path = os.path.split(args.eval)[0]
        model.train = False
        data = test_iter.next()
        #data = train_iter.next()
        x, t = concat_examples(data)
        loss = model(chainer.Variable(x), chainer.Variable(t))

        #from IPython import embed; embed()
        # visualize the last batch
        for k in range(min(200, args.batchsize)):
            xy = [28 * loc.data[k] for loc in model.l_list]
            xc, yc = zip(*xy)
            plt.figure()
            plt.imshow(x[k].reshape(28, 28), cmap = plt.get_cmap('gray'))
            plt.plot(yc, xc)
            plt.scatter(yc, xc, color='r')
            filename = '{}:true={},est={}.png'.format(k, t[k], model.y.data[k])
            plt.savefig(os.path.join(save_path, filename))
            plt.close()

