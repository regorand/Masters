from tokenize import Special
from xml.etree.ElementInclude import include
import drjit as dr
import mitsuba as mi
from coordinate_system import CoordinateSystem
import pprint

if mi.variant() == None:
    raise RuntimeError("Can't import the kernels module without setting a mi variant")


from contexts import Context


##################################################################
##################################################################
# Coordinate transforms
##################################################################
##################################################################

@dr.syntax
def cartesian_to_spherical(directions):
    theta = dr.safe_acos(directions[1])
    phi = dr.atan2(directions[2], (directions[0]))

    return mi.Point2f(phi, theta)

@dr.syntax
def spherical_to_cartesian(angles):
    phi = angles[:, 0]
    theta = angles[:, 1]

    return mi.Vector3f(
        dr.sin(theta) * dr.cos(phi),
        dr.cos(theta),
        dr.sin(theta) * dr.sin(phi)
    )

@dr.syntax
def cartesian_to_cylindrical(vectors):
    y = -vectors[:, 1]
    phi = dr.atan2(vectors[:, 2], vectors[:, 0])
    return mi.Point2f(phi, y)

@dr.syntax
def cylindrical_to_cartesian(cylindrical):
    phi = cylindrical[:, 0]
    y = -cylindrical[:, 1]
    rho = dr.sqrt(1 - dr.square(y))

    return mi.Vector3f(
        rho * dr.cos(phi),
        y,
        rho * dr.sin(phi)
    )

@dr.syntax
def spherical_to_cylindrical(angles):
    phi = angles[:, 0]
    theta = angles[:, 1]

    y = theta.cos()

    return mi.Point2f(phi, theta)

@dr.syntax
def cylindrical_to_spherical(cylindrical):
    phi = cylindrical[:, 0]
    y = cylindrical[:, 1]

    theta = y.acos()

    return mi.Point2f(phi, theta)

@dr.syntax
def cartesian_to_system(vectors, coordsSystem: CoordinateSystem):
    isCylindrical = coordsSystem == CoordinateSystem.Cylindrical
    return dr.select(isCylindrical, cartesian_to_cylindrical(vectors), cartesian_to_spherical(vectors))


@dr.syntax
def system_to_cartesian(coordinates, coordsSystem: CoordinateSystem):
    isCylindrical = coordsSystem == CoordinateSystem.Cylindrical
    return dr.select(isCylindrical, cylindrical_to_cartesian(coordinates), spherical_to_cartesian(coordinates))


def get_domain_from_coords(coordinate_system: CoordinateSystem):
    # Mitsuba's data types are array based, so first array is x values, second is y values
    if coordinate_system == CoordinateSystem.Cylindrical:
        return mi.Point2f([[-dr.pi, -1], [dr.pi, 1]])
    elif coordinate_system == CoordinateSystem.Spherical:
        return mi.Point2f([[-dr.pi, 0], [dr.pi, dr.pi]])



@dr.syntax
def directions_to_angles(directions: mi.Vector3f):
    theta = dr.safe_acos(directions[1])
    phi = dr.atan2(directions[2], (directions[0]))

    return phi, theta


@dr.syntax
def angles_to_dirs(phi, theta):
    return mi.Vector3f(
        dr.sin(theta) * dr.cos(phi),
        dr.cos(theta),
        dr.sin(theta) * dr.sin(phi)
    )


##################################################################
##################################################################
# Distribution utils
##################################################################
##################################################################

@dr.syntax
def modulate_with_cosine(angles, values, normals):
    directions = angles_to_dirs(angles[:, 0], angles[:, 1])
    weights = dr.maximum(dr.dot(directions, normals), 0.0)
    return values * weights


@dr.syntax
def random_dirs(num_samples, rng: dr.rng):
    return dr.normalize(rng.normal(shape=(3, num_samples), dtype=mi.Vector3f))



##################################################################
##################################################################
# Ray generation
##################################################################
##################################################################

@dr.syntax
def init_rays_linear(pos, look_at):
    ray_dir = dr.normalize(look_at - pos)
    return mi.Ray3f(o = pos, d = ray_dir)

@dr.syntax
def init_rays_look_at(pos, look_at, width, height, fov = dr.pi / 3, up = mi.Vector3f([0, 1, 0])):
    return init_rays(pos, dr.normalize(look_at - pos), width, height, fov, up)

@dr.syntax
def init_rays(origin, view_dir, width, height, fov = dr.pi / 3, up = mi.Vector3f([0, 1, 0])):
    aspect=float(width) / float(height)
    updist = dr.tan(fov / 2)
    side_dist = updist * aspect

    tangentDir = dr.cross(dr.auto.ad.Array3f(view_dir), dr.auto.ad.Array3f(up))

    eps = 0 # 1e-5

    x, y = dr.meshgrid(
        dr.linspace(dr.auto.Float32, -side_dist + eps, side_dist + eps, width),
        dr.linspace(dr.auto.Float32, -updist, updist, height)
    )

    ray_dir = dr.normalize(view_dir + x * tangentDir - y * up)

    rays = mi.Ray3f(o = origin, d = ray_dir)

    return rays


##################################################################
##################################################################
# Ray Intersection
##################################################################
##################################################################

@dr.syntax
def intersect_rays(scene, ray, active, update_active = True):
    si: mi.SurfaceInteraction3f = scene.ray_intersect(ray=ray, active=active)
    dr.eval(si)

    if update_active:
        active &= si.is_valid()
    
    return si, active


##################################################################
##################################################################
# Path tracing utilities
##################################################################
##################################################################

@dr.syntax
def direct_light(si, scene, active, stop_on_direct = True):
    e = si.emitter(scene)
    emitter_hit = e != None

    direct_eval = dr.select(emitter_hit, e.eval(si, active), 0.0)

    if stop_on_direct:
        active &= ~emitter_hit    

    return direct_eval
    


@dr.syntax
def get_nee_sample(si, scene, rng, active):
    return scene.sample_emitter_direction(si, rng.random(mi.Point2f, shape=(2, dr.width(active))), active=active)



@dr.syntax
def emitter_sampling_brdf(si, scene, rng, lanes, bsdf_ctx, active):
    res = scene.sample_emitter_direction(si, rng.random(mi.Point2f, shape=(2, lanes)))
    
    out_dir_sample: mi.DirectionSample3f = res[0]
    spec: mi.Spectrum = res[1]

    bsdf: mi.BSDFPtr = si.bsdf()

    bsdf_reflectance =  bsdf.eval(bsdf_ctx, si, si.to_local(out_dir_sample.d), active=active)
    
    nee_pdf = scene.pdf_emitter_direction(si, out_dir_sample, active=active)
    return bsdf_reflectance * spec, nee_pdf


@dr.syntax
def sample_new_dir_bsdf(si, rng, bsdf: mi.BSDF, bsdf_ctx, lanes, active):
    sample1 = rng.random(mi.Float, shape = (lanes))
    sample2 = rng.random(mi.Point2f, shape = (2, lanes))
    bs, spec = bsdf.sample(ctx=bsdf_ctx, 
                           si=si, 
                           sample1=sample1, 
                           sample2=sample2, 
                           active=active)
    return bs, spec



@dr.syntax
def mis(direct_contrib, nee_contrib, bsdf_pdf, nee_pdf, active, ctx: Context, lanes):
    nee_weight =        dr.select(active, 
                                dr.power(nee_pdf , ctx.mis_beta) if ctx.emitter_sampling else 
                                    dr.zeros(dr.auto.ad.Float, (lanes)), 
                                0)


    prev_brdf_weight =  dr.select(active, 
                                dr.power(bsdf_pdf, ctx.mis_beta) if ctx.bsdf_sampling    
                                    else dr.zeros(dr.auto.ad.Float, (lanes)),
                                0)

    # nee_weight = dr.power(nee_pdf , ctx.mis_beta) if ctx.emitter_sampling else dr.zeros(dr.auto.ad.Float, (lanes))
    # prev_brdf_weight = dr.power(bsdf_pdf, ctx.mis_beta) if ctx.bsdf_sampling else dr.zeros(dr.auto.ad.Float, (lanes))

    weight_sum = nee_weight + prev_brdf_weight
    norm_mask = weight_sum > 0

    nee_weight = dr.select(norm_mask, nee_weight / weight_sum, nee_weight)
    prev_brdf_weight = dr.select(norm_mask, prev_brdf_weight / weight_sum, prev_brdf_weight)

    return direct_contrib + nee_weight * nee_contrib, prev_brdf_weight


##################################################################
##################################################################
# General Utilities
##################################################################
##################################################################

@dr.syntax
def conditioning_to_torch(normal, in_dir, position, mask_idx):
    masked_normal = dr.gather(mi.Vector3f, mi.Vector3f(normal), mask_idx)
    masked_in_dir = dr.gather(mi.Vector3f, mi.Vector3f(in_dir), mask_idx)
    masked_pos = dr.gather(mi.Point3f, mi.Point3f(position), mask_idx)

    num_samples = dr.width(masked_normal)

    normal_torch = masked_normal.torch().transpose(0, 1)
    in_dir_torch = masked_in_dir.torch().transpose(0, 1)
    pos_torch = masked_pos.torch().transpose(0, 1)

    return normal_torch, in_dir_torch, pos_torch, num_samples

##################################################################
##################################################################
# Sampling
##################################################################
##################################################################

@dr.syntax
def sample_emitter_rays(scene, rng, num_samples):   
    ems = scene.emitters_dr()
    emitter_count = len(ems)

    sample = rng.random(mi.Float, (num_samples))
    e_idx = mi.UInt(sample * emitter_count)
    e_pdf = dr.repeat(mi.Float(1) / emitter_count, num_samples)

    emitters = dr.gather(mi.EmitterPtr, ems, e_idx)

    sample1 = rng.random(mi.Point2f, (2, num_samples))
    sample2 = rng.random(mi.Point2f, (2, num_samples))

    rays, weighted_radiance = emitters.sample_ray(0.5, 0, sample1, sample2)

    return rays, weighted_radiance, e_pdf



class SampleGenState():
    def __init__(self, si, forward_si, spec, pdf, new_dir):
        self.si = si
        self.forward_si = forward_si # surface interaction forwards
        self.spec = spec
        self.pdf = pdf
        self.new_dir = new_dir
        self.prev_bsdf_eval = dr.ones_like(self.spec)
        self.bsdf_pdf = dr.ones_like(self.spec)
        self.is_direct = dr.repeat(mi.Bool(True), dr.width(self.spec))

def empty_sample_gen_state(scene, num_samples):
    pos = dr.repeat(scene.bbox().min - mi.Point3f(1), num_samples)
    dir = dr.repeat(dr.normalize(mi.Vector3f(-1)), num_samples)

    rays = mi.Ray3f(pos, dir)
    si = scene.ray_intersect(rays)

    spec    = dr.repeat(mi.Spectrum(0),              num_samples)
    pdf     = dr.repeat(mi.Float(1),                 num_samples)
    new_dir = dr.repeat(mi.Vector3f(0),              num_samples)

    return SampleGenState(si, si, spec, pdf, new_dir)

@dr.syntax
def grid_sample_sensor(sens: mi.Sensor, width, height, rng, do_jitter = True):
    x, y = dr.meshgrid(
        dr.linspace(dr.auto.ad.Float, 0, 1, width, endpoint=False),
        dr.linspace(dr.auto.ad.Float, 0, 1, height, endpoint=False)
    )

    p = mi.Point2f(x, y)

    if do_jitter:
        norm = dr.auto.Array2f(width, height)
        jitter = (rng.random(dr.auto.Array2f, (2, width * height)) - 0.5) / norm

        p += jitter

    return sens.sample_ray(time = 0, sample1=0, sample2=p, sample3=mi.Point2f(0))[0]


@dr.syntax
def generate_samples_sensor(width, height, scene, sensor, rng, ctx: Context, model, sample_model, include_brdf = True):
    num_samples = width * height
    sample1 = rng.random(mi.Float, (num_samples))
    sample2 = rng.random(mi.Point2f, (2, num_samples))

    bsdf_ctx = mi.BSDFContext(mi.TransportMode.Importance)

    ray = grid_sample_sensor(sensor, width, height, rng)

    si = scene.ray_intersect(ray)

    bsdf = si.bsdf()
    active = (si.is_valid()) & (bsdf is not None)

    in_dirs_world = si.to_world(si.wi)

    new_dir = mi.Vector3f(0)
    new_dir_pdf = mi.Float(0)
    bsdf_spec = mi.Spectrum(0)
    new_dirs_world = mi.Vector3f(0)
    bsdf_pdf = mi.Float(0)
    if sample_model:
        new_dir_sample, new_dir_pdf = sample_from_model(model, active, si.n, in_dirs_world, si.p)
        new_dirs_world = system_to_cartesian(new_dir_sample, model.coordinate_system)
        new_dir = si.to_local(new_dirs_world)
        bsdf_spec = bsdf.eval(bsdf_ctx, si, new_dir, active)
        bsdf_pdf  = bsdf.pdf(bsdf_ctx, si, new_dir, active)
    else:
        sample2 = rng.random(mi.Point2f, (2, num_samples))

        bsdf_sample, bsdf_spec = bsdf.sample(bsdf_ctx, si, sample1, sample2, active)
        new_dir = bsdf_sample.wo
        new_dirs_world = si.to_world(new_dir)
        bsdf_pdf = bsdf_sample.pdf

        # Bsdf pdf is already divided out by bsdf.sample
        new_dir_pdf = dr.ones(mi.Float, (dr.width(active)))



    rays = si.spawn_ray(d = new_dirs_world)
    value = compute_path(rays, 
                         scene, 
                         ctx, 
                         rng, 
                         incoming_active=active)

    if include_brdf:
        value *= bsdf_spec
    else:
        value *= dr.maximum(dr.dot(new_dir, si.n), 0.0)

    return value, new_dirs_world, si.n, in_dirs_world, si.p, new_dir_pdf, bsdf_pdf, si, active



# @dr.syntax
def generate_samples_light_tracing(scene: mi.Scene, state: SampleGenState, rng, include_brdf = True):
    if not include_brdf:
        state.spec *= state.prev_bsdf_eval

    active = (state.si.is_valid()) & (dr.mean(state.spec) > 0.01)
    # active = (state.si.is_valid()) & (dr.mean(state.spec) > rng.random(mi.Float, (dr.width(state.si, ))))
    active_indices = dr.compress(active)
    inactive_indices = dr.compress(~active)

    total_samples = dr.width(state.si)
    rays = dr.zeros(mi.Ray3f, (total_samples))

    old_rays = dr.gather(mi.Ray3f, state.si.spawn_ray(d = state.new_dir), active_indices)
    dr.scatter(rays, old_rays, active_indices)

    num_samples = dr.width(inactive_indices)
    if num_samples != 0:    
        new_rays, spec, pdf = sample_emitter_rays(scene, rng, num_samples)

        dr.scatter(rays,            new_rays,       inactive_indices)
        dr.scatter(state.spec,      spec,           inactive_indices)
        dr.scatter(state.pdf,       pdf,            inactive_indices)

    si = scene.ray_intersect(rays)

    bsdf: mi.BSDFPtr = si.bsdf()
    active = si.is_valid() & (bsdf is not None)

    ctx = mi.BSDFContext(mi.TransportMode.Radiance)
    bsdf_sample, spec = sample_new_dir_bsdf(si, rng, bsdf, ctx, lanes=dr.width(si), active=active)

    # Compute the bsdf pdf that would be used for MIS when rendering
    si_c = dr.copy(si)
    si_c.wi = dr.copy(bsdf_sample.wo)
    
    importance_ctx = mi.BSDFContext(mi.TransportMode.Importance)
    mis_bsdf_pdf = bsdf.pdf(importance_ctx, si_c, si.wi, active)

    test = dr.any(dr.all(si_c.bsdf().sample(importance_ctx, si_c, rng.random(mi.Float, shape = dr.width(si)), rng.random(mi.Point2f, shape = (2, dr.width(si)))) == 0))


    state.si = si
    state.forward_si = si_c
    state.new_dir = si.to_world(bsdf_sample.wo)

    state.prev_bsdf_eval = spec
    state.bsdf_pdf = mis_bsdf_pdf
    if include_brdf:
        state.spec *=   spec
        state.pdf =     bsdf_sample.pdf
    else:
        state.spec *= dr.maximum(dr.dot(state.new_dir, si.n), 0.0)


    return state, active


@dr.syntax
def get_samples(ray: mi.Ray3f, scene: mi.Scene, ctx: Context, rng: dr.random.Generator, include_brdf = True):
    out = mi.Spectrum([0, 0, 0])
    lanes = dr.width(ray)

    bsdf_ctx: mi.BSDFContext = mi.BSDFContext(mode = mi.TransportMode.Radiance)

    active = dr.full(dr.auto.ad.Bool, True, lanes)

    si, active = intersect_rays(scene, ray, active)

    mask = si.is_valid()
    normals = dr.select(mask, si.n, mi.Vector3f([0, 0, 0]))

    bsdf: mi.BSDFPtr = si.bsdf()
    active &= bsdf is not None

    bsdf_sample, spec = sample_new_dir_bsdf(si, rng, bsdf, bsdf_ctx, lanes, active)
    in_dirs = si.to_world(si.wi)
    pos = si.p
    out_dirs = si.to_world(bsdf_sample.wo)

    ray = si.spawn_ray(d = out_dirs)
    si, _ = intersect_rays(scene, ray, active, update_active=False)
    
    out = direct_light(si, scene, active)
    if include_brdf:
        out *= spec
    else:
        out *= dr.maximum(dr.dot(out_dirs, si.n), 0.0)  

    return out, out_dirs, normals, in_dirs, pos, bsdf_sample.pdf, mask

@dr.syntax
def sample_from_model(model, active, normal: mi.Vector3f, in_dir: mi.Vector3f, position: mi.Point3f, bsdf_prior = False, rng = None, si = None):   
    mask_idx = dr.compress(active)


    out_samples = dr.full(mi.Point2f, 0.0, dr.width(active))
    out_pdf = dr.full(mi.Float, 1e-6, dr.width(active))

    if dr.width(mask_idx) > 0:
        normal_torch, in_dir_torch, pos_torch, num_samples = conditioning_to_torch(normal, in_dir, position, mask_idx)

        samples = None
        pdf = None
        if bsdf_prior:
            active_si = dr.gather(mi.SurfaceInteraction3f, si, mask_idx)
            samples, pdf = model.sample_with_pdf_bsdf_prior(num_samples, active_si, rng, normal_torch, in_dir_torch, pos_torch)
        else:
            samples, pdf = model.sample_with_pdf(num_samples, normal_torch, in_dir_torch, pos_torch)
        
        mi_samples = mi.Point2f(samples.transpose(0, 1))

        dr.scatter(out_samples, mi_samples,     mask_idx)
        dr.scatter(out_pdf,     mi.Float(pdf),  mask_idx)

    return out_samples, out_pdf



@dr.syntax
def sample_model_respect_deltas(si, model, rng, active):
    """
    Need to consider delta materials seperately, as their eval will always be 0, so I need to sample the bsdf for them
    """

    bsdf: mi.BSDFPtr = si.bsdf()

    ctx_smooth = mi.BSDFContext(mode = mi.TransportMode.Importance, type_mask=mi.BSDFFlags.Smooth)
    ctx_all =    mi.BSDFContext(mode = mi.TransportMode.Importance)

    sample1 = rng.random(mi.Float, (dr.width(si), ))
    sample2 = rng.random(mi.Point2f, (2, dr.width(si)))


    bsdf_sample_all, all_spec = bsdf.sample(ctx_all, si, sample1, sample2, active = active)
    bsdf_sample_smooth, _ = bsdf.sample(ctx_smooth, si, sample1, sample2, active = active)

    sampled_smooth = (bsdf_sample_all.sampled_type & mi.UInt(mi.BSDFFlags.Smooth)) != 0 

    p_l = dr.select(sampled_smooth, 
                    dr.clamp(bsdf_sample_all.pdf / bsdf_sample_smooth.pdf, 0, 1.0), 
                    1.0)

    model_active = (active) & (sampled_smooth)
    model_samples, pdf = sample_from_model(model, model_active, si.n, si.to_world(si.wi), si.p)
    new_model_dirs = system_to_cartesian(model_samples, model.coordinate_system)

    bsdf_eval = bsdf.eval(ctx_smooth, si, si.to_local(new_model_dirs), model_active)

    return dr.select(
                sampled_smooth, 
                (new_model_dirs,                       bsdf_eval,            p_l * pdf), 
                (si.to_world(bsdf_sample_all.wo),      all_spec,             p_l))

@dr.syntax
def bsdf_model_MIS(si, model, rng, active, beta, bsdf_prior = False):
    """
    Need to consider delta materials seperately, as their eval will always be 0, so I need to sample the bsdf for them
    """

    bsdf: mi.BSDFPtr = si.bsdf()

    ctx_delta =  mi.BSDFContext(mode = mi.TransportMode.Importance, type_mask=mi.BSDFFlags.Delta)
    ctx_smooth = mi.BSDFContext(mode = mi.TransportMode.Importance, type_mask=mi.BSDFFlags.Smooth)
    ctx_all =    mi.BSDFContext(mode = mi.TransportMode.Importance)

    sample1 = rng.random(mi.Float, (dr.width(si), ))
    sample2 = rng.random(mi.Point2f, (2, dr.width(si)))


    bsdf_sample_all, all_spec = bsdf.sample(ctx_all, si, sample1, sample2, active = active)
    # bsdf_sample_delta, delta_spec =  bsdf.sample(ctx_delta, si, sample1, sample2, active = active)
    bsdf_sample_smooth, _ = bsdf.sample(ctx_smooth, si, sample1, sample2, active = active)

    sampled_smooth = (bsdf_sample_all.sampled_type & mi.UInt(mi.BSDFFlags.Smooth)) != 0 

    p_l = dr.select(sampled_smooth, 
                    dr.clamp(bsdf_sample_all.pdf / bsdf_sample_smooth.pdf, 0, 1.0), 
                    1.0)

    model_active = (active) & (sampled_smooth)
    model_samples, pdf = sample_from_model(model, model_active, si.n, si.to_world(si.wi), si.p, bsdf_prior=bsdf_prior, rng=rng, si = si)
    new_model_dirs = system_to_cartesian(model_samples, model.coordinate_system)

    model_pdf_weight  = dr.power(pdf, beta)
    smooth_pdf_weight = dr.power(bsdf_sample_smooth.pdf, beta)

    model_pdf_weight /= (model_pdf_weight + smooth_pdf_weight)

    smooth_decider = rng.random(mi.Float, (1, )) < model_pdf_weight

    smooth_spec = bsdf.eval(ctx_smooth, si, si.to_local(new_model_dirs), model_active)
    pdf *= p_l

    return dr.select(
                     sampled_smooth, 
                    #  (new_model_dirs,                       smooth_spec,            pdf), 
                    #  (si.to_world(bsdf_sample_all.wo),      all_spec,               dr.ones(mi.Float, sampled_smooth.shape)),
                     dr.select(
                        smooth_decider,
                        (new_model_dirs,                       smooth_spec,            pdf), 
                        (si.to_world(bsdf_sample_all.wo),      all_spec,               dr.ones(mi.Float, sampled_smooth.shape))
                        # (si.to_world(bsdf_sample_all.wo),      all_spec,               1.0) #  - model_pdf_weight)
                     ),
                     (si.to_world(bsdf_sample_all.wo),      all_spec,               dr.ones(mi.Float, sampled_smooth.shape)))


##################################################################
##################################################################
# Path tracing
##################################################################
##################################################################

def project_env(ray: mi.Ray3f, scene: mi.Scene):
    lanes = dr.width(ray)

    active = dr.full(dr.auto.ad.Bool, True, lanes)

    si, active = intersect_rays(scene, ray, active, update_active=False)

    bsdf: mi.BSDFPtr = si.bsdf()
    active &= bsdf is not None

    ray = si.spawn_ray(d = si.n)
    si, active = intersect_rays(scene, ray, active, update_active=False)

    direct = direct_light(si, scene, active)
    
    return direct

def project_model_prediction(ray: mi.Ray3f, scene: mi.Scene, model):
    lanes = dr.width(ray)

    active = dr.full(dr.auto.ad.Bool, True, lanes)

    si, active = intersect_rays(scene, ray, active)
    
    mask_idx = dr.compress(active)

    masked_normal = dr.gather(mi.Vector3f, mi.Vector3f(si.n), mask_idx)
    masked_in_dir = dr.gather(mi.Vector3f, mi.Vector3f(si.to_world(si.wi)), mask_idx)
    masked_pos = dr.gather(mi.Point3f, mi.Point3f(si.p), mask_idx)

    normal_torch = masked_normal.torch().transpose(0, 1)
    in_dir_torch = masked_in_dir.torch().transpose(0, 1)
    pos_torch = masked_pos.torch().transpose(0, 1)

    p = cartesian_to_system(masked_normal, model.coordinate_system)

    batch = p.torch().transpose(0, 1)

    probs = model.log_prob(batch, normal_torch, in_dir_torch, pos_torch).exp()

    out = dr.zeros(mi.Float, lanes)
    dr.scatter(out, mi.Float(probs), mask_idx)

    return out


def compute_model_debug(ray: mi.Ray3f, scene: mi.Scene, model, ctx: Context, rng: dr.random.Generator):
    out = mi.Spectrum([0, 0, 0])
    current = mi.Spectrum([1, 1, 1])

    lanes = dr.width(ray)
    bsdf_ctx: mi.BSDFContext = mi.BSDFContext(mode = mi.TransportMode.Radiance)

    active = dr.full(dr.auto.ad.Bool, True, lanes)

    
    si, active = intersect_rays(scene, ray, active, update_active=False)

    direct = direct_light(si, scene, active)
    out += current * direct

    active &= si.is_valid()

    bsdf: mi.BSDFPtr = si.bsdf()
    active &= bsdf is not None

    
    samples, pdf = sample_from_model(model, normal=si.n, in_dir=si.to_world(si.wi), position=si.p, active=active)

    mask_idx = dr.compress(active)

    normal_torch, in_dir_torch, pos_torch, _ = conditioning_to_torch(si.n, si.to_world(si.wi), si.p, mask_idx)

    masked_samples = dr.gather(mi.Point2f, samples, mask_idx)
    samples_torch = masked_samples.torch().transpose(0, 1)


    prob = model.log_prob(samples_torch, normal_torch, in_dir_torch, pos_torch).exp()

    probs = dr.repeat(mi.Float(0), lanes)
    dr.scatter(probs, prob, mask_idx)

    # new_dir = angles_to_dirs(samples[:, 0], samples[:, 1])
    new_dir = system_to_cartesian(samples, model.coordinate_system)
    

    new_dir_local = si.to_local(new_dir)
    spec = bsdf.eval(bsdf_ctx, si, new_dir_local, active=active)

    current = dr.select(active, current * spec / pdf, current)

    ray = si.spawn_ray(d = new_dir)
    si, active = intersect_rays(scene, ray, active, update_active=False)

    direct = direct_light(si, scene, active, stop_on_direct=False)
    
    return dr.select(active, direct, 0.0)


def compute_path_model_single_bounce_debug(ray: mi.Ray3f, scene: mi.Scene, model, ctx: Context, rng: dr.random.Generator):
    out = mi.Spectrum([0, 0, 0])
    current = mi.Spectrum([1, 1, 1])

    lanes = dr.width(ray)
    bsdf_ctx: mi.BSDFContext = mi.BSDFContext(mode = mi.TransportMode.Radiance)

    active = dr.full(dr.auto.ad.Bool, True, lanes)

    si, active = intersect_rays(scene, ray, active, update_active=False)

    out = direct_light(si, scene, active)

    active &= si.is_valid()
    bsdf: mi.BSDFPtr = si.bsdf()
    active &= bsdf is not None

    samples, pdf = sample_from_model(model, normal=si.n, in_dir=si.to_world(si.wi), position=si.p, active=active)
    
    # new_dir = angles_to_dirs(samples[:, 0], samples[:, 1])
    new_dir = system_to_cartesian(samples, model.coordinate_system)

    lambert_cos = dr.dot(new_dir, si.n)

    new_dir_local = si.to_local(new_dir)
    

    bsdf_eval = bsdf.eval(bsdf_ctx, si, new_dir_local, active=active)
    bsdf_eval /= dr.maximum(dr.abs(new_dir_local.z), 1e-6) 
    
    ray = si.spawn_ray(d = new_dir)
    si, active = intersect_rays(scene, ray, active, update_active=False)

    direct = direct_light(si, scene, active, stop_on_direct=False)

    out = dr.select(active, bsdf_eval * direct / pdf, out)
    
    return out, bsdf_eval, direct, pdf, new_dir, samples, lambert_cos, active


def compute_path_debug(ray: mi.Ray3f, scene: mi.Scene, ctx: Context, rng: dr.random.Generator, return_last_si = False):
    # TODO at some point go through this and rename the variables so they are more descriptive and accurate

    out = mi.Spectrum([0, 0, 0])
    current = mi.Spectrum([1, 1, 1])
    lanes = dr.width(ray)

    bsdf_ctx: mi.BSDFContext = mi.BSDFContext(mode = mi.TransportMode.Radiance)

    active = dr.full(dr.auto.ad.Bool, True, lanes)

    # MIS needs to keep track of the weight of brdf sampling the previous bounce, as direct lighting for that hit 
    # is only computed the next bounce (if an emissive surface is hit)
    prev_brdf_weight = dr.ones(dr.auto.ad.Float, (lanes))
    nee_weight = dr.zeros_like(prev_brdf_weight)

    nee_pdf = dr.zeros_like(prev_brdf_weight)

    brdf_contrib = mi.Spectrum([0, 0, 0])
    nee_contrib  = mi.Spectrum([0, 0, 0])

    for i in range(ctx.bounces):
        si, active = intersect_rays(scene, ray, active, update_active=False)

        if i == 0 or ctx.bsdf_sampling:
            direct = direct_light(si, scene, active)

            brdf_contrib = prev_brdf_weight * direct

        active &= si.is_valid()
        bsdf: mi.BSDFPtr = si.bsdf()

        active &= bsdf is not None

        if ctx.emitter_sampling:
            nee_contrib, nee_pdf = emitter_sampling_brdf(si, scene, rng, lanes, bsdf_ctx, active)

        return nee_pdf

        bsdf_sample, spec = sample_new_dir_bsdf(si, rng, bsdf, bsdf_ctx, lanes, active)

        contrib, prev_brdf_weight = mis(brdf_contrib, nee_contrib, bsdf_sample.pdf, nee_pdf, ctx, lanes)

        out += current * contrib

        current *= spec
        ray = si.spawn_ray(d = si.to_world(bsdf_sample.wo))
    
    return_val = out
    if return_last_si:
        return_val = (out, si)

    return return_val
    


def compute_path_model_single_bounce(ray: mi.Ray3f, scene: mi.Scene, model, ctx: Context, rng: dr.random.Generator):
    out = mi.Spectrum([0, 0, 0])
    current = mi.Spectrum([1, 1, 1])

    lanes = dr.width(ray)
    bsdf_ctx: mi.BSDFContext = mi.BSDFContext(mode = mi.TransportMode.Radiance)

    active = dr.full(dr.auto.ad.Bool, True, lanes)

    si, active = intersect_rays(scene, ray, active, update_active=False)

    out = direct_light(si, scene, active, stop_on_direct=False)

    active &= si.is_valid()
    bsdf: mi.BSDFPtr = si.bsdf()
    active &= bsdf is not None

    samples, pdf = sample_from_model(model, normal=si.n, in_dir=si.to_world(si.wi), position=si.p, active=active)
    
    # new_dir = angles_to_dirs(samples[:, 0], samples[:, 1])
    new_dir = system_to_cartesian(samples, model.coordinate_system)

    new_dir_local = si.to_local(new_dir)
    

    bsdf_eval = bsdf.eval(bsdf_ctx, si, new_dir_local, active=active)    
    
    ray = si.spawn_ray(d = new_dir)
    si, active = intersect_rays(scene, ray, active, update_active=False)

    direct = direct_light(si, scene, active, stop_on_direct=False)

    mask_idx = dr.compress(active)

    out = dr.select(active, bsdf_eval * direct / pdf, out)

    return out

@dr.syntax
def compute_path_model(ray: mi.Ray3f, scene: mi.Scene, model, ctx: Context, rng: dr.random.Generator):
    out = mi.Spectrum([0, 0, 0])
    current = mi.Spectrum([1, 1, 1])

    lanes = dr.width(ray)
    bsdf_ctx: mi.BSDFContext = mi.BSDFContext(mode = mi.TransportMode.Radiance)

    active = dr.full(dr.auto.ad.Bool, True, lanes)

    
    si, active = intersect_rays(scene, ray, active, update_active=False)
    direct = direct_light(si, scene, active)
    out += current * direct

    for i in range(ctx.bounces):
        active &= si.is_valid()

        bsdf: mi.BSDFPtr = si.bsdf()
        active &= bsdf is not None
        
        
        new_dir, spec, pdf = (bsdf_model_MIS(si, model, rng, active, ctx.model_MIS_beta, bsdf_prior=ctx.bsdf_prior)
                              if ctx.model_sample_MIS
                              else sample_model_respect_deltas(si, model, rng, active))

        current = dr.select(active, current * spec / (pdf + 1e-6), current)

        ray = si.spawn_ray(d = new_dir)
        si, active = intersect_rays(scene, ray, active, update_active=False)
            
        direct = direct_light(si, scene, active)
        out += current * direct
    
    return out



@dr.syntax
def compute_path(ray: mi.Ray3f, 
                 scene: mi.Scene, 
                 ctx: Context, 
                 rng: dr.random.Generator, 
                 return_last_si = False, 
                 incoming_active = None):
    out = mi.Spectrum([0, 0, 0])
    current = mi.Spectrum([1, 1, 1])
    lanes = dr.width(ray)

    bsdf_ctx: mi.BSDFContext = mi.BSDFContext(mode = mi.TransportMode.Radiance)

    si = mi.SurfaceInteraction3f()
    active = dr.full(dr.auto.ad.Bool, True, lanes)
    if incoming_active is not None:
        active = dr.select(incoming_active, mi.Bool(True), mi.Bool(False))
        # dr.scatter(active, mi.Bool(False), dr.compress(~incoming_active))

    skip_loop = False
    if ctx.only_direct:
        si, active = intersect_rays(scene, ray, active, update_active=False)
        out = direct_light(si, scene, active)
        skip_loop = True

    # MIS needs to keep track of the weight of brdf sampling the previous bounce, as direct lighting for that hit 
    # is only computed the next bounce (if an emissive surface is hit)
    prev_brdf_weight = dr.ones(dr.auto.ad.Float, (lanes))

    nee_pdf = dr.zeros_like(prev_brdf_weight)

    brdf_contrib = mi.Spectrum([0, 0, 0])
    nee_contrib  = mi.Spectrum([0, 0, 0])

    if not skip_loop:
        i = 0
        continue_loop = True
        # while dr.width(dr.compress(active)) > 0 and i < ctx.bounces:
        while continue_loop and i < ctx.bounces:
            i += 1

            si, active = intersect_rays(scene, ray, active, update_active=False)

            if (i == 0 or ctx.bsdf_sampling) and (i != 0 or not ctx.only_indirect):
                direct = direct_light(si, scene, active)
                
                brdf_contrib = prev_brdf_weight * direct

            active &= si.is_valid()
            bsdf: mi.BSDFPtr = si.bsdf()

            active &= bsdf is not None

            if ctx.emitter_sampling:
                nee_contrib, nee_pdf = emitter_sampling_brdf(si, scene, rng, lanes, bsdf_ctx, active)
            
            bsdf_sample, spec = sample_new_dir_bsdf(si, rng, bsdf, bsdf_ctx, lanes, active)

            active &= dr.any(spec != 0, axis = 0)
            active &= ~dr.all(dr.isnan(spec))
            contrib, prev_brdf_weight = mis(brdf_contrib, nee_contrib, bsdf_sample.pdf, nee_pdf, active, ctx, lanes)

            out += current * contrib
            current *= spec

            # alpha_prob = dr.max(current, axis=0)
            # alphas = rng.random(mi.Float, dr.width(active,))

            # active &= alphas < alpha_prob

            # continue_loop &= rng.random(mi.Float, (1, )) < ctx.rr_alpha
            # current /= ctx.rr_alpha

            ray = si.spawn_ray(d = si.to_world(bsdf_sample.wo))
    
    return_val = out
    if return_last_si:
        return_val = (out, si)

    return return_val
    