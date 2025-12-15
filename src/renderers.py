from textwrap import indent
from xml.etree.ElementInclude import include
from matplotlib.font_manager import weight_dict
import mitsuba as mi
import drjit as dr
import torch

import numpy as np

import matplotlib.pyplot as plt
from torch import normal

import kernels

import time
from contexts import Context
from coordinate_system import CoordinateSystem

def current_milli_time():
    return round(time.time() * 1000)


class Renderer():
    def __init__(self, scene: mi.Scene, render_context: Context, coordinate_system: CoordinateSystem, seed = 0):
        self.scene = scene
        self.ctx = render_context
        self.rng = dr.rng(seed)
        self.coordinate_system = coordinate_system


    def sample_sensor(self, width, height, jitter = True):
        if len(self.scene.sensors()) == 0:
            raise RuntimeError("Scene needs a sensor")
        
        sens: mi.Sensor = self.scene.sensors()[0]

        x, y = dr.meshgrid(
            dr.linspace(dr.auto.ad.Float, 0, 1, width, endpoint=False),
            dr.linspace(dr.auto.ad.Float, 0, 1, height, endpoint=False)
        )

        norm = dr.auto.Array2f(width, height)
        jitter = (self.rng.random(dr.auto.Array2f, (2, width * height)) - 0.5) / norm

        p = mi.Point2f(x, y) + jitter

        return sens.sample_ray(time = 0, sample1=0, sample2=p, sample3=mi.Point2f(0))[0]

    def get_active_render_ctx(self, render_context: Context):
        if render_context is None:
            return self.ctx
        return render_context

    def render(self, 
               width: int, 
               height: int, 
               render_context: Context = None):
        ctx = self.get_active_render_ctx(render_context)

        target: mi.Color3f = mi.Color3f(0)

        
        for i in range(ctx.spp):
            rays = self.sample_sensor(width, height)
            target += kernels.compute_path(rays, self.scene, ctx, rng=self.rng) 
            dr.eval(target)


        return target / ctx.spp
    
    def render_with_model(self, 
                          width: int, 
                          height: int, 
                          model, 
                          render_context: Context = None):
        ctx = self.get_active_render_ctx(render_context)

        target: mi.Color3f = mi.Color3f(0)

        with torch.no_grad():
            for i in range(ctx.spp):
                rays = self.sample_sensor(width, height)
                target += kernels.compute_path_model(rays, self.scene, model, ctx, rng=self.rng)

                dr.eval(target)

            return target / ctx.spp
    

    def compute_path_model_single_bounce(self, 
                          width: int, 
                          height: int, 
                          model, 
                          render_context: Context = None):
        ctx = self.get_active_render_ctx(render_context)

        target: mi.Color3f = mi.Color3f(0)

        for i in range(ctx.spp):
            rays = self.sample_sensor(width, height)
            target += kernels.compute_path_model_single_bounce(rays, self.scene, model, ctx, rng=self.rng)

        return target / ctx.spp
    
    def generate_samples_from_sensor(self, 
                                     context: Context,
                                     model,
                                     apply_mask = True):
                
        sens = self.scene.sensors()[0]
        value, new_dirs, normals, in_dirs, pos, pdf, bsdf_pdf, si, active = kernels.generate_samples_sensor(context.width, 
                                                                                              context.height, 
                                                                                              self.scene, 
                                                                                              sens,
                                                                                              self.rng,
                                                                                              context, 
                                                                                              model=model,
                                                                                              include_brdf=context.include_brdf, 
                                                                                              sample_model=context.sample_model)

        out_angles = kernels.cartesian_to_system(new_dirs, model.coordinate_system)

        dr.eval(value, out_angles, normals, in_dirs, pos, pdf, active)

        values    = mi.TensorXf(value)     .torch().detach().transpose(0, 1)
        angles    = mi.TensorXf(out_angles).torch().detach().transpose(0, 1)
        normals   = mi.TensorXf(normals)   .torch().detach().transpose(0, 1)
        in_dirs   = mi.TensorXf(in_dirs)   .torch().detach().transpose(0, 1)
        pos       = mi.TensorXf(pos)       .torch().detach().transpose(0, 1)
        pdfs      = mi.TensorXf(pdf)       .torch().detach()
        bsdf_pdfs = mi.TensorXf(bsdf_pdf)  .torch().detach()

        mask_torch = active.torch()
        if apply_mask:
            return values[mask_torch], angles[mask_torch], normals[mask_torch], in_dirs[mask_torch], pos[mask_torch], pdfs[mask_torch], bsdf_pdfs[mask_torch], dr.gather(mi.SurfaceInteraction3f, si, active)
        
        return values, angles, normals, in_dirs, pos, pdfs, bsdf_pdfs, mask_torch


    def generate_sample_light_tracing(self, context: Context, coordinate_system: CoordinateSystem, apply_mask = True):
        if not hasattr(self, "sample_gen_state") or self.sample_gen_state is None:
            self.sample_gen_state = kernels.empty_sample_gen_state(self.scene, context.num_samples)


        self.sample_gen_state, mask = kernels.generate_samples_light_tracing(self.scene, self.sample_gen_state, self.rng, include_brdf=context.include_brdf)

        # out_angles = kernels.directions_to_angles(self.sample_gen_state.si.to_world(self.sample_gen_state.si.wi))
        dirs = self.sample_gen_state.si.to_world(self.sample_gen_state.si.wi)
        out_angles = kernels.cartesian_to_system(dirs, coordinate_system)

        dr.eval(self.sample_gen_state.spec, 
                out_angles, 
                self.sample_gen_state.si.n, 
                self.sample_gen_state.new_dir, 
                self.sample_gen_state.new_dir, 
                self.sample_gen_state.si.p, 
                self.sample_gen_state.pdf, 
                self.sample_gen_state.bsdf_pdf,
                self.sample_gen_state.forward_si,
                mask)

        values    = mi.TensorXf(self.sample_gen_state.spec)    .torch().detach().transpose(0, 1)
        angles    = mi.TensorXf(out_angles)                    .torch().detach().transpose(0, 1)
        normals   = mi.TensorXf(self.sample_gen_state.si.n)    .torch().detach().transpose(0, 1)
        in_dirs   = mi.TensorXf(self.sample_gen_state.new_dir) .torch().detach().transpose(0, 1)
        pos       = mi.TensorXf(self.sample_gen_state.si.p)    .torch().detach().transpose(0, 1)
        pdfs      = mi.TensorXf(self.sample_gen_state.pdf)     .torch().detach()
        bsdf_pdfs = mi.TensorXf(self.sample_gen_state.bsdf_pdf).torch().detach()

        mask_torch = mask.torch()
        if apply_mask:
            masked_si = dr.gather(mi.SurfaceInteraction3f, self.sample_gen_state.forward_si, dr.compress(mask))
            return values[mask_torch], angles[mask_torch], normals[mask_torch], in_dirs[mask_torch], pos[mask_torch], pdfs[mask_torch], bsdf_pdfs[mask_torch], masked_si
        
        return values, angles, normals, in_dirs, pos, pdfs, bsdf_pdfs, self.sample_gen_state.forward_si, mask_torch
        

    def generate_samples(self, context: Context, coordinate_system: CoordinateSystem, model, apply_mask = True):
        if context.sample_generation_strategy == "sensor_sampling":
            return self.generate_samples_from_sensor(context, model, apply_mask=apply_mask)
        elif context.sample_generation_strategy == "light_tracing":
            return self.generate_sample_light_tracing(context, coordinate_system, apply_mask=apply_mask)
        

        # Fall back to light tracing
        return self.generate_sample_light_tracing(context, coordinate_system, apply_mask=apply_mask)


    def eval_samples(self, samples: mi.Point2f, positions: mi.Point3f, coordinate_system: CoordinateSystem, render_context: Context = None):
        # TODO
        ### Possibly render using more spp
        ### Possibly give target resolution from outside to determine how much to jiggle ?
        ##### Maybe give jiggle range instead
        ##### Or don't jiggle at all <- This might be it
        
        ctx = self.get_active_render_ctx(render_context)

        # dirs = kernels.angles_to_dirs(samples[:, 0], samples[:, 1])
        dirs = kernels.system_to_cartesian(samples, coordinate_system)
        rays = mi.Ray3f(o = positions, d = dirs)

        return kernels.compute_path(rays, self.scene, ctx, rng=self.rng)

    def sample_sensor_first_si(self, repeat_until_hit = True):
        while(repeat_until_hit):
            sample2 = self.rng.random(mi.Point2f, (2, 1))

            ray = self.scene.sensors()[0].sample_ray(time = 0, sample1=0, sample2=sample2, sample3=mi.Point2f(0))[0]
            si = self.scene.ray_intersect(ray)

            if si.is_valid():
                return si

        return si
    
    def sample_sensor_at_point(self, point):
        sample2 = mi.Point2d(point)

        ray = self.scene.sensors()[0].sample_ray(time = 0, sample1=0, sample2=sample2, sample3=mi.Point2f(0))[0]
        si = self.scene.ray_intersect(ray)

        return si


    def build_distribution_on_surface(self, sample_si: mi.SurfaceInteraction3f, resolution, coordinate_system: CoordinateSystem, include_brdf = True, render_context: Context = None, print_debug = False):
        ctx = self.get_active_render_ctx(render_context)
        
        N =    sample_si.n
        wi =   sample_si.to_world(sample_si.wi)
        p =    sample_si.p
        bsdf = sample_si.bsdf()

        bsdf_ctx = mi.BSDFContext(mi.TransportMode.Importance)

        
        # domain depends on coordinate system
        domain = kernels.get_domain_from_coords(coordinate_system)

        # TODO this is not fully correct for bsfs that have delta and non-delta components!
        if (bsdf.flags()[0] & mi.BSDFFlags.Delta) != 0:
            bsdf_sample, spec = bsdf.sample(bsdf_ctx, 
                                            sample_si, 
                                            self.rng.random(mi.Float, (1, )),
                                            self.rng.random(mi.Point2f, (2, 1)))
            
            dist = dr.zeros(mi.Color3f, (3, resolution[0] * resolution[1]))

            angles = kernels.cartesian_to_system(sample_si.to_world(bsdf_sample.wo), coordinate_system)

            indices = dr.floor(([angles.x[0], angles.y[0]] - domain[:, 0]) / (domain[:, 1] - domain[:, 0]) * resolution)
            flat_index = indices[1] * resolution[0] + indices[0]
                        
            # dist[flat_index] = spec[0]
            dr.scatter(dist, spec, flat_index)
            norm_target = dr.mean(dist, axis = 0)
            dr.eval(norm_target)
            return norm_target



        phi, theta = dr.meshgrid(
            dr.linspace(dr.auto.ad.Float, domain[0][0], domain[1][0], resolution[0], endpoint=False),
            dr.linspace(dr.auto.ad.Float, domain[0][1], domain[1][1], resolution[1], endpoint=False)
        )

        norm = dr.auto.Array2f(resolution[0], resolution[1])
        eps = 1e-4

        target: mi.Color3f = mi.Color3f(0)
        for i in range(ctx.spp):
            start = current_milli_time()

            jitter = (self.rng.random(dr.auto.Array2f, (2, resolution[0] * resolution[1])) - 0.5) / norm

            jittered_phi = phi + jitter[:, 0]
            jittered_theta = theta + jitter[:, 1]

            dirs = kernels.angles_to_dirs(jittered_phi, jittered_theta)
            dirs = kernels.system_to_cartesian(mi.Point2f(jittered_phi, jittered_theta), coordinate_system)           

            rays = mi.Ray3f(o = p + eps * dirs, d = dirs)

            path_val, _ = kernels.compute_path(rays, 
                                               self.scene, 
                                               ctx, 
                                               rng=self.rng, 
                                               return_last_si=True)

            if include_brdf:
                bsdf_eval = bsdf.eval(bsdf_ctx, sample_si, sample_si.to_local(dirs))
                target += path_val * bsdf_eval
            else:
                inital_lambertian = dr.maximum(dr.dot(dirs, N), 0.0)
                target += path_val * inital_lambertian

            dr.eval(target)

            if print_debug:
                print(f"Iteration {i} rendered in {current_milli_time() - start} ms")


        norm_target = dr.mean(target / ctx.spp, axis = 0)
        dr.eval(norm_target)

        return norm_target
    

    def build_distribution_floating(self, pos: mi.Point3f, resolution, coordinate_system: CoordinateSystem, render_context: Context = None):
        ctx = self.get_active_render_ctx(render_context)
        
        domain = kernels.get_domain_from_coords(coordinate_system)

        phi, theta = dr.meshgrid(
            dr.linspace(dr.auto.ad.Float, domain[0][0], domain[1][0], resolution[0], endpoint=False),
            dr.linspace(dr.auto.ad.Float, domain[0][1], domain[1][1], resolution[1], endpoint=False)
        )

        norm = dr.auto.Array2f(resolution[0], resolution[1])
        
        target: mi.Color3f = mi.Color3f(0)
        for _ in range(ctx.spp):
            jitter = (self.rng.random(dr.auto.Array2f, (2, resolution[0] * resolution[1])) - 0.5) / norm

            jittered_phi = phi + jitter[:, 0]
            jittered_theta = theta + jitter[:, 1]

            # dirs = kernels.angles_to_dirs(jittered_phi, jittered_theta)
            dirs = kernels.system_to_cartesian(mi.Point2f(jittered_phi, jittered_theta), coordinate_system)
            rays = mi.Ray3f(o = pos, d = dirs)

            target += kernels.compute_path(rays, self.scene, ctx, rng=self.rng)
            dr.eval(target)
        
        norm_target = dr.mean(target / ctx.spp, axis = 0)
        dr.eval(norm_target)

        return norm_target
        # return target / ctx.spp
