import os
import util
import bpy
from contextlib import contextmanager
import sys
@contextmanager
def stdout_redirected(to=os.devnull):
    '''
    import os

    with stdout_redirected(to=filename):
        print("from Python")
        os.system("echo non-Python applications are also supported")
    '''
    fd = sys.stdout.fileno()

    ##### assert that Python and C stdio write using the same file descriptor
    ####assert libc.fileno(ctypes.c_void_p.in_dll(libc, "stdout")) == fd == 1

    def _redirect_stdout(to):
        sys.stdout.close() # + implicit flush()
        os.dup2(to.fileno(), fd) # fd writes to 'to' file
        sys.stdout = os.fdopen(fd, 'w') # Python writes to fd

    with os.fdopen(os.dup(fd), 'w') as old_stdout:
        with open(to, 'w') as file:
            _redirect_stdout(to=file)
        try:
            yield # allow code to be run with the redirected stdout
        finally:
            _redirect_stdout(to=old_stdout) # restore stdout.
                                            # buffering and flags such as
                                            # CLOEXEC may be different


class BlenderInterface():
    def __init__(self, resolution=128, background_color=(1,1,1)):
        self.resolution = resolution
        self.fov = 90
        # self.set_camera_params()

        # Delete the default cube (default selected)
        bpy.ops.object.delete()

        self.scene = bpy.context.scene
        # Deselect all. All new object added to the scene will automatically selected.
        self.blender_renderer = bpy.context.scene.render
        self.blender_renderer.use_high_quality_normals = False
        self.blender_renderer.resolution_x = resolution
        self.blender_renderer.resolution_y = resolution
        self.blender_renderer.resolution_percentage = 100
        self.blender_renderer.image_settings.file_format = 'PNG'  # set output format to .png
        self.blender_renderer.film_transparent = False


        world = bpy.context.scene.world
        world.color = background_color
        if world.use_nodes:
            bg_node = world.node_tree.nodes.get("Background")
            if not bg_node:
                bg_node = world.node_tree.nodes.new(type="ShaderNodeBackground")
            bg_node.inputs[0].default_value = (1, 1, 1, 1)
            bg_node.inputs[1].default_value = 1.0  # 控制环境光强度
        
        # 设置光照
        # self.setup_lights()

        # Set up the camera
        if bpy.context.scene.camera is None:
            bpy.ops.object.camera_add()  # 添加新相机
            self.camera = bpy.context.object  # 获取新创建的相机
            bpy.context.scene.camera = self.camera  # 设置为场景默认相机
        else:
            self.camera = bpy.context.scene.camera
        self.camera.data.sensor_height = self.camera.data.sensor_width # Square sensor
        util.set_camera_focal_length_in_world_units(self.camera.data, 525./512*resolution) # Set focal length to a common value (kinect)

        self.set_engine(cycles=True)

        bpy.ops.object.select_all(action='DESELECT')



    def set_engine(self, cycles=True):
        if cycles:
            bpy.context.scene.render.engine = 'CYCLES'
            self.scene.cycles.samples = 10000
            if True:
                bpy.context.preferences.addons[
                    "cycles"
                ].preferences.compute_device_type = "CUDA"
                bpy.context.scene.cycles.device = "GPU"
            else:
                bpy.context.scene.cycles.device = "CPU"


            bpy.context.preferences.addons["cycles"].preferences.get_devices()
            print(bpy.context.preferences.addons["cycles"].preferences.compute_device_type)
            self.current_engine = 'cycles'
        else:
            bpy.context.scene.render.engine = 'BLENDER_EEVEE'
            self.current_engine = 'eevee'

    def setup_lights(self):
        """设置光照"""
        # 删除所有光源
        for light in bpy.data.lights:
            bpy.data.lights.remove(light)

        # 创建太阳光
        bpy.ops.object.light_add(type='SUN')
        sun = bpy.context.object
        sun.data.energy = 1.0

        # 复制两个光源，以便均匀照明
        sun2 = sun.copy()
        sun2.location = (10, -10, 10)
        bpy.context.collection.objects.link(sun2)

        sun3 = sun.copy()
        sun3.location = (-10, 10, 10)
        bpy.context.collection.objects.link(sun3)
        sun3.data.energy = 0.3

    def import_mesh(self, fpath, scale=1., object_world_matrix=None):
        
        ext = os.path.splitext(fpath)[-1]
        if ext == '.obj':
            bpy.ops.import_scene.obj(filepath=str(fpath), split_mode='OFF')
        elif ext == '.ply':
            bpy.ops.import_mesh.ply(filepath=str(fpath))

        obj = bpy.context.selected_objects[0]
        util.dump(bpy.context.selected_objects)

        if object_world_matrix is not None:
            obj.matrix_world = object_world_matrix

        bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY', center='BOUNDS')
        obj.location = (0., 0., 0.) # center the bounding box!

        if scale != 1.:
            bpy.ops.transform.resize(value=(scale, scale, scale))

        # Disable transparency & specularities
        M = bpy.data.materials
        for i in range(len(M)):
            if M[i].use_nodes:  # 如果材质使用了节点
                bsdf_node = M[i].node_tree.nodes.get("Principled BSDF")
                if bsdf_node:
                    bsdf_node.inputs["Alpha"].default_value = 1.0  # 设置透明度为 1（不透明）
            M[i].specular_intensity = 0.0

        # Disable texture interpolation
        T = bpy.data.textures
        for i in range(len(T)):
            try:
                T[i].use_interpolation = False
                T[i].use_mipmap = False
                T[i].use_filter_size_min = True
                T[i].filter_type = "BOX"
            except:
                continue
    
    def render_pose(self, pose):
        
        self.set_camera_pose(pose)
        
        result = self.render() # 
        # img = result['mask']*result['image']
        img = rgb_to_rgba(result['image']*result['mask'], result['mask'])
        return img

    def render(self, output_dir, blender_cam2world_matrices, write_cam_params=False):

        if write_cam_params:
            img_dir = os.path.join(output_dir, 'rgb')
            pose_dir = os.path.join(output_dir, 'pose')

            util.cond_mkdir(img_dir)
            util.cond_mkdir(pose_dir)
        else:
            img_dir = output_dir
            util.cond_mkdir(img_dir)

        if write_cam_params:
            K = util.get_calibration_matrix_K_from_blender(self.camera.data)
            with open(os.path.join(output_dir, 'intrinsics.txt'),'w') as intrinsics_file:
                intrinsics_file.write('%f %f %f 0.\n'%(K[0][0], K[0][2], K[1][2]))
                intrinsics_file.write('0. 0. 0.\n')
                intrinsics_file.write('1.\n')
                intrinsics_file.write('%d %d\n'%(self.resolution, self.resolution))

        for i in range(len(blender_cam2world_matrices)):
            self.camera.matrix_world = blender_cam2world_matrices[i]

            # Render the object
            if os.path.exists(os.path.join(img_dir, '%06d.png' % i)):
                continue

            # Render the color image
            self.blender_renderer.filepath = os.path.join(img_dir, '%06d.png'%i)
            material_indices_backup = {}
            for obj in bpy.data.objects:
                if obj.type == 'MESH':
                    mesh = obj.data
                    material_indices_backup[obj.name] = [poly.material_index for poly in mesh.polygons]
            bpy.context.view_layer.update()
            
            bpy.ops.render.render(write_still=True)

            # 恢复材质
            for obj_name, indices in material_indices_backup.items():
                obj = bpy.data.objects.get(obj_name)
                if obj and obj.type == 'MESH':
                    mesh = obj.data
                    for poly, material_index in zip(mesh.polygons, indices):
                        poly.material_index = material_index
            

            if write_cam_params:
                # Write out camera pose
                RT = util.get_world2cam_from_blender_cam(self.camera)
                cam2world = RT.inverted()
                with open(os.path.join(pose_dir, '%06d.txt'%i),'w') as pose_file:
                    matrix_flat = []
                    for j in range(4):
                        for k in range(4):
                            matrix_flat.append(cam2world[j][k])
                    pose_file.write(' '.join(map(str, matrix_flat)) + '\n')

        # Remember which meshes were just imported
        meshes_to_remove = []
        for ob in bpy.context.selected_objects:
            meshes_to_remove.append(ob.data)

        bpy.ops.object.delete()

        # Remove the meshes from memory too
        for mesh in meshes_to_remove:
            bpy.data.meshes.remove(mesh)

if __name__=='__main__':
    mesh_fpath = "/mnt/hdd/zhengquan/Shapenet/ShapeNetCore.v2/02691156/1a04e3eab45ca15dd86060f189eb133/models/model_normalized.obj"
    output_dir = "/mnt/hdd/zhengquan/Shapenet/stanford_shapenet_render"

    instance_name = mesh_fpath.split('/')[-3]
    instance_dir = os.path.join(opt.output_dir, instance_name)

    renderer = blender_interface.BlenderInterface(resolution=128)