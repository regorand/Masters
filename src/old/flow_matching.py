from src.old.neural_net import MLP
from src.old.execution_context import ExecutionContext
from src.old.training_context import TrainingContext

import numpy as np
import torch
import tqdm
import src.old.mcmc as mcmc
import src.old.utils as utils
from PIL import Image
from itertools import product


class FlowM_Model:
    def __init__(self, net: MLP, ex_context: ExecutionContext, train_context: TrainingContext):
        self.net = net
        self.ex_context = ex_context
        self.train_context = train_context

    def eval(self, input, inference_steps = 10, requires_grad = False):
        self.net.eval().requires_grad_(requires_grad)

        for i, t in enumerate(torch.linspace(0, 1, inference_steps), start=1):
            pred = self.net(input, t.expand(input.size(0)))
            input = input + (1 / inference_steps) * pred
            input = self.ex_context.domain.wrap_torch(input)

        return input

    def sample(self, N: int, inference_steps = 10, requires_grad = False):
        points = self.ex_context.input_dist.sample(N)

        return self.eval(points, inference_steps = inference_steps, requires_grad = requires_grad)

    def get_input_shape(self):
        return self.net.input_shape()

    def train(self):
        optim = torch.optim.AdamW(self.net.parameters(), lr=self.train_context.lr)

        data = self.ex_context.target_dist.rejection_sample(self.train_context.batch_size)

        scores = []
        imgs = []

        pbar = tqdm.tqdm(range(self.train_context.training_steps))
        for i in pbar:
            for sample in data:
                sample = mcmc.MCMC_step(self.ex_context.target_dist, sample, 0.5, 0.1)

            x1 = torch.Tensor(data)
            x0 = torch.from_numpy(self.ex_context.input_dist.sample(len(x1))).type(torch.float32)

            target = self.ex_context.domain.dist_func_torch(x0, x1)

            t = torch.rand(x1.size(0))
            xt = self.ex_context.domain.wrap_func_torch(x0 + t[:, None] * target)

            pred = self.net(xt, t)
            loss = ((target - pred) ** 2).mean()
            loss.backward()
            optim.step()
            optim.zero_grad()

            if self.train_context.track_score and i % self.train_context.track_every == 0:
                if self.train_context.animate_training:
                    score, hist = self.score_model(requires_grad=True, return_hist = True)
                    # imgs.append(Image.fromarray(255 * hist))

                    filt_40 = utils.filter_and_norm(hist, 20)
                    imgs.append(Image.fromarray(filt_40 * 255))
                else:
                    score = self.score_model(requires_grad = True, return_hist = False)

                scores.append((i, score))
                pbar.set_postfix(score=score)

        if self.train_context.animate_training:
            ms = 1000 * self.train_context.gif_length / len(imgs)

            imgs[0].save(self.train_context.training_gif_name, save_all=True, append_images=imgs[1:], duration=int(ms), loop=0)

        if self.train_context.track_score:
            return scores

    def grid_eval(self, inference_steps = 10, pixel_gap = 20, requires_grad = False, return_hist = False):
        target_dist = self.ex_context.target_dist
        predicted_distribution = np.zeros(target_dist.get_dims(True), dtype=np.uint64)

        self.net.eval().requires_grad_(requires_grad)

        pixel_gap_x = pixel_gap / target_dist.width
        pixel_gap_y = pixel_gap / target_dist.height

        unif_points = list(product(np.arange(0.001, 1, pixel_gap_x), np.arange(0.0001, 1, pixel_gap_y)))
        unif_points.append([1, 1])

        xt = torch.Tensor(np.array(self.ex_context.input_dist.transform(unif_points)))
        xt = self.eval(xt, inference_steps, requires_grad = requires_grad)

        predicted_distribution = utils.points_to_image(xt, target_dist.get_dims(True))

        predicted_distribution.astype(np.float64)
        return predicted_distribution

    def score_model(self, inference_steps = 10, pixel_gap = 20, requires_grad = False, return_hist = False):
        hist = self.grid_eval(inference_steps = inference_steps, pixel_gap = pixel_gap, requires_grad = requires_grad)
        prepared = utils.filter_and_norm(hist, 20)
        if return_hist:
            return (self.ex_context.target_dist.abs_dist(prepared), hist)
        else:
            return self.ex_context.target_dist.abs_dist(prepared)

    def animate_inference(self, images, pixel_gap, length, gif_name, filter = False):
        imgs = []
        target_dist = self.ex_context.target_dist


        inference_steps = images
        ms = length / images

        self.net.eval().requires_grad_(False)

        pixel_gap_x = pixel_gap / target_dist.width
        pixel_gap_y = pixel_gap / target_dist.height

        unif_points = list(product(np.arange(0.001, 1, pixel_gap_x), np.arange(0.0001, 1, pixel_gap_y)))
        unif_points.append([1, 1])

        xt = torch.Tensor(np.array(self.ex_context.input_dist.transform(unif_points)))

        predicted_distribution = utils.points_to_image(xt, target_dist.get_dims(True))

        if filter:
            predicted_distribution = utils.filter_and_norm(predicted_distribution, 20)

        predicted_distribution.astype(np.float64)
        imgs.append(Image.fromarray(predicted_distribution * 255))

        for i, t in enumerate(torch.linspace(0, 1, inference_steps)):
            pred = self.net(xt, t.expand(xt.size(0)))
            xt = xt + (1 / inference_steps) * pred
            xt = self.ex_context.domain.wrap_torch(xt)

            predicted_distribution = utils.points_to_image(xt, target_dist.get_dims(True))

            if filter:
                predicted_distribution = utils.filter_and_norm(predicted_distribution, 20)

            predicted_distribution.astype(np.float64)
            imgs.append(Image.fromarray(predicted_distribution * 255))

        imgs[0].save(gif_name, save_all=True, append_images=imgs[1:], duration=int(ms), loop=0)