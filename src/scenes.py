import mitsuba as mi
from mitsuba import ScalarTransform4f as T

def env_scene_sphere():
    return mi.load_dict({
        'type': 'scene',
        'integrator': {
            'type': 'path',
            'max_depth': 8
        },
        # -------------------- Sensor --------------------
        'sensor': {
            'type': 'perspective',
            'fov_axis': 'smaller',
            'near_clip': 0.001,
            'far_clip': 100.0,
            'focus_distance': 1000,
            'fov': 39.3077,
            'to_world': mi.ScalarTransform4f().look_at(
                origin=[0, 0, -3],
                target=[0, 0, 0],
                up=[0, 1, 0]
            ),
            'sampler': {
                'type': 'independent',
                'sample_count': 64
            },
            'film': {
                'type': 'hdrfilm',
                'width' : 256,
                'height': 256,
                'rfilter': {
                    'type': 'gaussian',
                },
                'pixel_format': 'rgb',
                'component_format': 'float32',
            }
        },
        # -------------------- BSDFs --------------------
        'white': {
            'type': 'diffuse',
            'reflectance': {
                'type': 'rgb',
                'value': [0.885809, 0.698859, 0.666422],
            }
        },
        # -------------------- Env Map --------------------
        'emitter_id': {
            'type': 'envmap',
            # 'to_world': mi.ScalarTransform4f().look_at(
            #     origin=[-1, 0, 0],
            #     target=[0, 0, 0],
            #     up=[0, 1, 0]
            # ),
            'filename': '../assets/test_distributions/museum.jpg'
            # 'filename': '../assets/test_distributions/Uniform_Env.png'
            # 'filename': '../assets/test_distributions/Gradient_Env.png'
            # 'filename': '../assets/test_distributions/High_freq_checker.png'
            # 'filename': '../assets/test_distributions/Bright_Sun_Env.png'
            # 'filename': '../assets/test_distributions/TestDist2.png',
            # 'scale': 20
        },
        # -------------------- Shapes --------------------
        'object': {
            'type': 'sphere',
            'to_world': mi.ScalarTransform4f(),
            'bsdf': {
                'type': 'ref',
                'id':  'white'
            }
        },
    })


def env_scene_sphere_metallic():
    return mi.load_dict({
        'type': 'scene',
        'integrator': {
            'type': 'path',
            'max_depth': 8
        },
        # -------------------- Sensor --------------------
        'sensor': {
            'type': 'perspective',
            'fov_axis': 'smaller',
            'near_clip': 0.001,
            'far_clip': 100.0,
            'focus_distance': 1000,
            'fov': 39.3077,
            'to_world': mi.ScalarTransform4f().look_at(
                origin=[3, 0, 0],
                target=[0, 0, 0],
                up=[0, 1, 0]
            ),
            'sampler': {
                'type': 'independent',
                'sample_count': 64
            },
            'film': {
                'type': 'hdrfilm',
                'width' : 256,
                'height': 256,
                'rfilter': {
                    'type': 'gaussian',
                },
                'pixel_format': 'rgb',
                'component_format': 'float32',
            }
        },
        # -------------------- BSDFs --------------------
        'metallic': {
            'type': 'roughconductor',
            'alpha': 0.3
        },
        # -------------------- Env Map --------------------
        'emitter_id': {
            'type': 'envmap',
            # 'to_world': mi.ScalarTransform4f().look_at(
            #     origin=[-1, 0, 0],
            #     target=[0, 0, 0],
            #     up=[0, 1, 0]
            # ),
            'filename': '../assets/test_distributions/museum.jpg'
            # 'filename': '../assets/test_distributions/Uniform_Env.png'
            # 'filename': '../assets/test_distributions/Gradient_Env.png'
            # 'filename': '../assets/test_distributions/High_freq_checker.png'
            # 'filename': '../assets/test_distributions/Bright_Sun_Env.png'
            # 'filename': '../assets/test_distributions/TestDist2.png',
            # 'scale': 20
        },
        # -------------------- Shapes --------------------
        'object': {
            'type': 'sphere',
            'to_world': mi.ScalarTransform4f(),
            'bsdf': {
                'type': 'ref',
                'id':  'metallic'
            }
        },
    })

def env_scene_sphere_dielectric():
    return mi.load_dict({
        'type': 'scene',
        'integrator': {
            'type': 'path',
            'max_depth': 8
        },
        # -------------------- Sensor --------------------
        'sensor': {
            'type': 'perspective',
            'fov_axis': 'smaller',
            'near_clip': 0.001,
            'far_clip': 100.0,
            'focus_distance': 1000,
            'fov': 39.3077,
            'to_world': mi.ScalarTransform4f().look_at(
                origin=[3, 0, 0],
                target=[0, 0, 0],
                up=[0, 1, 0]
            ),
            'sampler': {
                'type': 'independent',
                'sample_count': 64
            },
            'film': {
                'type': 'hdrfilm',
                'width' : 256,
                'height': 256,
                'rfilter': {
                    'type': 'gaussian',
                },
                'pixel_format': 'rgb',
                'component_format': 'float32',
            }
        },
        # -------------------- BSDFs --------------------
        'glass': {
            'type': 'dielectric',
        },
        # -------------------- Env Map --------------------
        'emitter_id': {
            'type': 'envmap',
            # 'to_world': mi.ScalarTransform4f().look_at(
            #     origin=[-1, 0, 0],
            #     target=[0, 0, 0],
            #     up=[0, 1, 0]
            # ),
            'filename': '../assets/test_distributions/museum.jpg'
            # 'filename': '../assets/test_distributions/Uniform_Env.png'
            # 'filename': '../assets/test_distributions/Gradient_Env.png'
            # 'filename': '../assets/test_distributions/High_freq_checker.png'
            # 'filename': '../assets/test_distributions/Bright_Sun_Env.png'
            # 'filename': '../assets/test_distributions/TestDist2.png',
            # 'scale': 20
        },
        # -------------------- Shapes --------------------
        'object': {
            'type': 'sphere',
            'to_world': mi.ScalarTransform4f(),
            'bsdf': {
                'type': 'ref',
                'id':  'glass'
            }
        },
    })


def env_scene_sphere_rough_dielectric():
    return mi.load_dict({
        'type': 'scene',
        'integrator': {
            'type': 'path',
            'max_depth': 8
        },
        # -------------------- Sensor --------------------
        'sensor': {
            'type': 'perspective',
            'fov_axis': 'smaller',
            'near_clip': 0.001,
            'far_clip': 100.0,
            'focus_distance': 1000,
            'fov': 39.3077,
            'to_world': mi.ScalarTransform4f().look_at(
                origin=[3, 0, 0],
                target=[0, 0, 0],
                up=[0, 1, 0]
            ),
            'sampler': {
                'type': 'independent',
                'sample_count': 64
            },
            'film': {
                'type': 'hdrfilm',
                'width' : 256,
                'height': 256,
                'rfilter': {
                    'type': 'gaussian',
                },
                'pixel_format': 'rgb',
                'component_format': 'float32',
            }
        },
        # -------------------- BSDFs --------------------
        'glass': {
            'type': 'roughdielectric',
            'alpha': 0.1
        },
        # -------------------- Env Map --------------------
        'emitter_id': {
            'type': 'envmap',
            # 'to_world': mi.ScalarTransform4f().look_at(
            #     origin=[-1, 0, 0],
            #     target=[0, 0, 0],
            #     up=[0, 1, 0]
            # ),
            'filename': '../assets/test_distributions/museum.jpg'
            # 'filename': '../assets/test_distributions/Uniform_Env.png'
            # 'filename': '../assets/test_distributions/Gradient_Env.png'
            # 'filename': '../assets/test_distributions/High_freq_checker.png'
            # 'filename': '../assets/test_distributions/Bright_Sun_Env.png'
            # 'filename': '../assets/test_distributions/TestDist2.png',
            # 'scale': 20
        },
        # -------------------- Shapes --------------------
        'object': {
            'type': 'sphere',
            'to_world': mi.ScalarTransform4f(),
            'bsdf': {
                'type': 'ref',
                'id':  'glass'
            }
        },
    })


def env_scene_empty():
    return mi.load_dict({
        'type': 'scene',
        'integrator': {
            'type': 'path',
            'max_depth': 8
        },
        # -------------------- Sensor --------------------
        'sensor': {
            'type': 'perspective',
            'fov_axis': 'smaller',
            'near_clip': 0.001,
            'far_clip': 100.0,
            'focus_distance': 1000,
            'fov': 39.3077,
            'to_world': mi.ScalarTransform4f().look_at(
                origin=[0, 0, 6],
                target=[0, 0, 0],
                up=[0, 1, 0]
            ),
            'sampler': {
                'type': 'independent',
                'sample_count': 64
            },
            'film': {
                'type': 'hdrfilm',
                'width' : 256,
                'height': 256,
                'rfilter': {
                    'type': 'gaussian',
                },
                'pixel_format': 'rgb',
                'component_format': 'float32',
            }
        },
        # -------------------- BSDFs --------------------
        'white': {
            'type': 'diffuse',
            'reflectance': {
                'type': 'rgb',
                'value': [0.885809, 0.698859, 0.666422],
            }
        },
        # -------------------- Env Map --------------------
        'emitter_id': {
            'type': 'envmap',
            # 'to_world': mi.ScalarTransform4f().look_at(
            #     origin=[-1, 0, 0],
            #     target=[0, 0, 0],
            #     up=[0, 1, 0]
            # ),
            'filename': '../assets/test_distributions/museum.jpg'
        },
    })

def glossy_cbox():
    return mi.load_dict({
        'type': 'scene',
        'integrator': {
            'type': 'path',
            'max_depth': 8
        },
        # -------------------- Sensor --------------------
        'sensor': {
            'type': 'perspective',
            'fov_axis': 'smaller',
            'near_clip': 0.001,
            'far_clip': 100.0,
            'focus_distance': 1000,
            'fov': 39.3077,
            'to_world': T().look_at(
                origin=[0, 0, 3.90],
                target=[0, 0, 0],
                up=[0, 1, 0]
            ),
            'sampler': {
                'type': 'independent',
                'sample_count': 64
            },
            'film': {
                'type': 'hdrfilm',
                'width' : 256,
                'height': 256,
                'rfilter': {
                    'type': 'gaussian',
                },
                'pixel_format': 'rgb',
                'component_format': 'float32',
            }
        },
        # -------------------- BSDFs --------------------
        'white': {
            'type': 'diffuse',
            'reflectance': {
                'type': 'rgb',
                'value': [0.885809, 0.698859, 0.666422],
            }
        },
        'Silver': {
            'type': 'roughconductor',
            'material': 'Au',
            'alpha': 0.3
        },
        'Copper': {
            'type': 'roughconductor',
            'material': 'Cu2O',
            'alpha': 0.3
        },
        'Gold': {
            'type': 'roughconductor',
            'material': 'Au',
            'alpha': 0.1
        },
        # -------------------- Light --------------------
        'light': {
            'type': 'rectangle',
            'to_world': T().translate([0.0, 0.99, 0.01]).rotate([1, 0, 0], 90).scale([0.23, 0.19, 0.19]),
            'bsdf': {
                'type': 'ref',
                'id': 'white'
            },
            'emitter': {
                'type': 'area',
                'radiance': {
                    'type': 'rgb',
                    #'value': [18.387, 13.9873, 6.75357],
                    'value': [50, 35, 20],
                }
            }
        },
        # -------------------- Shapes --------------------
        'floor': {
            'type': 'rectangle',
            'to_world': T().translate([0.0, -1.0, 0.0]).rotate([1, 0, 0], -90),
            'bsdf': {
                'type': 'ref',
                'id':  'white'
            }
        },
        'ceiling': {
            'type': 'rectangle',
            'to_world': T().translate([0.0, 1.0, 0.0]).rotate([1, 0, 0], 90),
            'bsdf': {
                'type': 'ref',
                'id':  'white'
            }
        },
        'back': {
            'type': 'rectangle',
            'to_world': T().translate([0.0, 0.0, -1.0]),
            'bsdf': {
                'type': 'ref',
                'id':  'white'
            }
        },
        'green-wall': {
            'type': 'rectangle',
            'to_world': T().translate([1.0, 0.0, 0.0]).rotate([0, 1, 0], -90),
            'bsdf': {
                'type': 'ref',
                'id':  'Copper'
            }
        },
        'red-wall': {
            'type': 'rectangle',
            'to_world': T().translate([-1.0, 0.0, 0.0]).rotate([0, 1, 0], 90),
            'bsdf': {
                'type': 'ref',
                'id':  'Silver'
            }
        },
        'small-box': {
            'type': 'cube',
            'to_world': T().translate([0.335, -0.7, 0.38]).rotate([0, 1, 0], -17).scale(0.3),
            'bsdf': {
                'type': 'ref',
                'id':  'Gold'
            }
        },
        'large-box': {
            'type': 'cube',
            'to_world': T().translate([-0.33, -0.4, -0.28]).rotate([0, 1, 0], 18.25).scale([0.3, 0.61, 0.3]),
            'bsdf': {
                'type': 'ref',
                'id':  'Gold'
            }
        },
    })

def cbox():
    return mi.load_dict({
        'type': 'scene',
        'integrator': {
            'type': 'path',
            'max_depth': 8
        },
        # -------------------- Sensor --------------------
        'sensor': {
            'type': 'perspective',
            'fov_axis': 'smaller',
            'near_clip': 0.001,
            'far_clip': 100.0,
            'focus_distance': 1000,
            'fov': 39.3077,
            'to_world': T().look_at(
                origin=[0, 0, 3.90],
                target=[0, 0, 0],
                up=[0, 1, 0]
            ),
            'sampler': {
                'type': 'independent',
                'sample_count': 64
            },
            'film': {
                'type': 'hdrfilm',
                'width' : 512,
                'height': 512,
                'rfilter': {
                    'type': 'gaussian',
                },
                'pixel_format': 'rgb',
                'component_format': 'float32',
            }
        },
        # -------------------- BSDFs --------------------
        'white': {
            'type': 'diffuse',
            'reflectance': {
                'type': 'rgb',
                'value': [0.885809, 0.698859, 0.666422],
            }
        },
        'green': {
            'type': 'diffuse',
            'reflectance': {
                'type': 'rgb',
                'value': [0.105421, 0.37798, 0.076425],
            }
        },
        'red': {
            'type': 'diffuse',
            'reflectance': {
                'type': 'rgb',
                'value': [0.570068, 0.0430135, 0.0443706],
            }
        },
        'blue': {
            'type': 'diffuse',
            'reflectance': {
                'type': 'rgb',
                'value': [0.188, 0.235, 0.820],
            }
        },
        # -------------------- Light --------------------
        'light': {
            'type': 'rectangle',
            'to_world': T().translate([0.0, 0.99, 0.01]).rotate([1, 0, 0], 90).scale([0.23, 0.19, 0.19]),
            'bsdf': {
                'type': 'ref',
                'id': 'white'
            },
            'emitter': {
                'type': 'area',
                'radiance': {
                    'type': 'rgb',
                    #'value': [18.387, 13.9873, 6.75357],
                    'value': [50, 35, 20],
                }
            }
        },
        # -------------------- Shapes --------------------
        'floor': {
            'type': 'rectangle',
            'to_world': T().translate([0.0, -1.0, 0.0]).rotate([1, 0, 0], -90),
            'bsdf': {
                'type': 'ref',
                'id':  'white'
            }
        },
        'ceiling': {
            'type': 'rectangle',
            'to_world': T().translate([0.0, 1.0, 0.0]).rotate([1, 0, 0], 90),
            'bsdf': {
                'type': 'ref',
                'id':  'white'
            }
        },
        'back': {
            'type': 'rectangle',
            'to_world': T().translate([0.0, 0.0, -1.0]),
            'bsdf': {
                'type': 'ref',
                'id':  'white'
            }
        },
        'green-wall': {
            'type': 'rectangle',
            'to_world': T().translate([1.0, 0.0, 0.0]).rotate([0, 1, 0], -90),
            'bsdf': {
                'type': 'ref',
                'id':  'green'
            }
        },
        'red-wall': {
            'type': 'rectangle',
            'to_world': T().translate([-1.0, 0.0, 0.0]).rotate([0, 1, 0], 90),
            'bsdf': {
                'type': 'ref',
                'id':  'red'
            }
        },
        'small-box': {
            'type': 'cube',
            'to_world': T().translate([0.335, -0.7, 0.38]).rotate([0, 1, 0], -17).scale(0.3),
            'bsdf': {
                'type': 'ref',
                'id':  'blue'
            }
        },
        'large-box': {
            'type': 'cube',
            'to_world': T().translate([-0.33, -0.4, -0.28]).rotate([0, 1, 0], 18.25).scale([0.3, 0.61, 0.3]),
            'bsdf': {
                'type': 'ref',
                'id':  'red'
            }
        },
    })


def cbox_inverted_light():
    return mi.load_dict({
        'type': 'scene',
        'integrator': {
            'type': 'path',
            'max_depth': 8
        },
        # -------------------- Sensor --------------------
        'sensor': {
            'type': 'perspective',
            'fov_axis': 'smaller',
            'near_clip': 0.001,
            'far_clip': 100.0,
            'focus_distance': 1000,
            'fov': 39.3077,
            'to_world': T().look_at(
                origin=[0, 0, 3.90],
                target=[0, 0, 0],
                up=[0, 1, 0]
            ),
            'sampler': {
                'type': 'independent',
                'sample_count': 64
            },
            'film': {
                'type': 'hdrfilm',
                'width' : 512,
                'height': 512,
                'rfilter': {
                    'type': 'gaussian',
                },
                'pixel_format': 'rgb',
                'component_format': 'float32',
            }
        },
        # -------------------- BSDFs --------------------
        'white': {
            'type': 'diffuse',
            'reflectance': {
                'type': 'rgb',
                'value': [0.885809, 0.698859, 0.666422],
            }
        },
        'green': {
            'type': 'diffuse',
            'reflectance': {
                'type': 'rgb',
                'value': [0.105421, 0.37798, 0.076425],
            }
        },
        'red': {
            'type': 'diffuse',
            'reflectance': {
                'type': 'rgb',
                'value': [0.570068, 0.0430135, 0.0443706],
            }
        },
        'blue': {
            'type': 'diffuse',
            'reflectance': {
                'type': 'rgb',
                'value': [0.188, 0.235, 0.820],
            }
        },
        # -------------------- Light --------------------
        'light': {
            'type': 'rectangle',
            'to_world': T().translate([0.0, 0.80, 0.01]).rotate([1, 0, 0], -90).scale([0.23, 0.19, 0.19]),
            'bsdf': {
                'type': 'ref',
                'id': 'white'
            },
            'emitter': {
                'type': 'area',
                'radiance': {
                    'type': 'rgb',
                    #'value': [18.387, 13.9873, 6.75357],
                    'value': [50, 35, 20],
                }
            }
        },
        # -------------------- Shapes --------------------
        'floor': {
            'type': 'rectangle',
            'to_world': T().translate([0.0, -1.0, 0.0]).rotate([1, 0, 0], -90),
            'bsdf': {
                'type': 'ref',
                'id':  'white'
            }
        },
        'ceiling': {
            'type': 'rectangle',
            'to_world': T().translate([0.0, 1.0, 0.0]).rotate([1, 0, 0], 90),
            'bsdf': {
                'type': 'ref',
                'id':  'white'
            }
        },
        'back': {
            'type': 'rectangle',
            'to_world': T().translate([0.0, 0.0, -1.0]),
            'bsdf': {
                'type': 'ref',
                'id':  'white'
            }
        },
        'green-wall': {
            'type': 'rectangle',
            'to_world': T().translate([1.0, 0.0, 0.0]).rotate([0, 1, 0], -90),
            'bsdf': {
                'type': 'ref',
                'id':  'green'
            }
        },
        'red-wall': {
            'type': 'rectangle',
            'to_world': T().translate([-1.0, 0.0, 0.0]).rotate([0, 1, 0], 90),
            'bsdf': {
                'type': 'ref',
                'id':  'red'
            }
        },
        'small-box': {
            'type': 'cube',
            'to_world': T().translate([0.335, -0.7, 0.38]).rotate([0, 1, 0], -17).scale(0.3),
            'bsdf': {
                'type': 'ref',
                'id':  'blue'
            }
        },
        'large-box': {
            'type': 'cube',
            'to_world': T().translate([-0.33, -0.4, -0.28]).rotate([0, 1, 0], 18.25).scale([0.3, 0.61, 0.3]),
            'bsdf': {
                'type': 'ref',
                'id':  'red'
            }
        },
    })



def veach_mis_scene():
    return mi.load_file("../assets/scenes/veach_mis/scene_v3.xml")

def kitchen():
    return mi.load_file("../assets/scenes/kitchen/scene.xml")

def spaceship():
    return mi.load_file("../assets/scenes/spaceship/scene.xml")

def living_room_2():
    return mi.load_file("../assets/scenes/living-room-2/scene.xml")

def veach_ajar():
    return mi.load_file("../assets/scenes/veach-ajar/scene.xml")

def get_scene_by_name(name: str):
    if name == 'env_scene_sphere':
        return env_scene_sphere()
    elif name == 'env_scene_sphere_metallic':
        return env_scene_sphere_metallic()
    elif name == 'env_scene_sphere_dielectric':
        return env_scene_sphere_dielectric()
    elif name == 'env_scene_sphere_rough_dielectric':
        return env_scene_sphere_rough_dielectric()
    elif name == 'env_scene_empty':
        return env_scene_empty()
    elif name == 'glossy_cbox':
        return glossy_cbox()
    elif name == 'cbox':
        return cbox()
    elif name == 'cbox_inverted_light':
        return cbox_inverted_light()
    elif name == 'veach_mis_scene':
        return veach_mis_scene()
    elif name == 'kitchen':
        return kitchen()
    elif name == 'spaceship':
        return spaceship()
    elif name == 'living_room_2':
        return living_room_2()
    elif name == 'veach_ajar':
        return veach_ajar()
