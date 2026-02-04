import os
import sys
sys.path.append(os.path.dirname(__file__))
import util
import bpy
from contextlib import contextmanager

import numpy as np
import mathutils
import torch
from gtsam import Point3, Pose3, Rot3
from PIL import Image
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

def convert_to_blender_pose(pose, return_matrix=False):
    if type(pose) is torch.Tensor:
        rot = Rot3.Quaternion(pose[3],*pose[:3])
        pose = Pose3(rot, pose[4:]).matrix()
    if type(pose) is Pose3:
        pose = pose.matrix()
    pose = mathutils.Matrix(pose)
    if return_matrix:
        return pose
    else:
        return pose.to_translation(), pose.to_euler()

class BlenderInterface():
    def __init__(self, resolution=128, background_color=(1,1,1)):
        self.resolution = resolution
        self.fov = 29.999999290482176
        self.objects = {}
        # self.set_camera_params()

        self.set_camera_params()
        self.scene = bpy.context.scene

        # Delete the default cube (default selected)
        bpy.ops.object.delete()
        [bpy.data.objects.remove(obj) for obj in bpy.data.objects if obj.type == "MESH"]

        self.add_camera()


        self.scene = bpy.context.scene
        # Deselect all. All new object added to the scene will automatically selected.
        self.blender_renderer = bpy.context.scene.render
        self.blender_renderer.use_high_quality_normals = False
        # self.scene.view_settings.view_transform = 'Standard'
        self.blender_renderer.resolution_x = resolution
        self.blender_renderer.resolution_y = resolution
        self.blender_renderer.resolution_percentage = 100
        self.blender_renderer.image_settings.color_mode = 'RGBA'  # set output mode to RGBA
        self.blender_renderer.image_settings.file_format = 'PNG'  # set output format to .png
        self.blender_renderer.film_transparent = True

        world = bpy.context.scene.world
        # new
        world.use_nodes = True
        node_tree = world.node_tree

        tex_coord_node = node_tree.nodes.new(type="ShaderNodeTexCoord")
        mapping_node = node_tree.nodes.new(type="ShaderNodeMapping")
        env_tex_node = node_tree.nodes.new(type="ShaderNodeTexEnvironment")
        bg_node = node_tree.nodes.get("Background")
        if not bg_node:
            bg_node = node_tree.nodes.new(type="ShaderNodeBackground")

        env_tex_node.image = bpy.data.images.load('data/assets/hdri/gray_hdri.exr')
        mapping_node.inputs["Rotation"].default_value = (0,0,0)
        bg_node.inputs[1].default_value = 1.0

        node_tree.links.new(tex_coord_node.outputs["Generated"], mapping_node.inputs["Vector"])
        node_tree.links.new(mapping_node.outputs["Vector"], env_tex_node.inputs["Vector"])
        node_tree.links.new(env_tex_node.outputs["Color"], bg_node.inputs["Color"])

        # 添加点光源
        # for obj in bpy.data.objects:
        #     if obj.type == 'LIGHT':
        #         bpy.data.objects.remove(obj, do_unlink=True)
        # light_data = bpy.data.lights.new(name="MyLight", type='POINT')
        # light_data.energy = 1000
        # light_object = bpy.data.objects.new(name="MyLight", object_data=light_data)
        # bpy.context.collection.objects.link(light_object)
        # light_object.location = (2.0, -2.0, 3.0)

        # origin
        world.color = background_color
        if world.use_nodes:
            bg_node = world.node_tree.nodes.get("Background")
            if not bg_node:
                bg_node = world.node_tree.nodes.new(type="ShaderNodeBackground")
            bg_node.inputs[0].default_value = (1, 1, 1, 1)
            bg_node.inputs[1].default_value = 0.15  # 控制环境光强度
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

    def add_camera(self, camera_matrix=np.eye(4)):
        for obj in bpy.context.scene.objects:
            if obj.type == 'CAMERA':
                bpy.data.objects.remove(obj, do_unlink=True)

        rot = mathutils.Matrix(camera_matrix).to_euler()
        translation = mathutils.Matrix(camera_matrix).to_translation()
        bpy.ops.object.camera_add(location=translation, rotation=rot)
        camera = bpy.context.object
        self.camera = camera
        return camera

    def add_object(self, key, obj_file_path, obj_matrix=np.eye(4), scale=1.):
        with stdout_redirected():
            bpy.ops.import_scene.obj(filepath=obj_file_path)
        obj = bpy.context.selected_objects[0]
        
        obj.scale = scale * np.ones(3)
        inst_id = len(self.objects) +1000
        obj["inst_id"] = inst_id
        for i, ob in enumerate(bpy.data.objects): 
            if ob.parent == obj: 
                ob["inst_id"]= inst_id+i+1
        with bpycv.activate_obj(obj):
            bpy.ops.rigidbody.object_add()
        self.objects[key] = obj

    def set_camera_params(self):
        self.height, self.width = self.resolution,self.resolution
        
        self.hfov = self.fov/180 *np.pi
        self.vfov = self.fov/180 *np.pi

        self.fx = 0.5*self.width/np.tan(self.hfov/2 )
        self.fy = 0.5*self.height/np.tan(self.vfov/2 )

    def set_engine(self, cycles=True):
        if cycles:
            bpy.context.scene.render.engine = 'CYCLES'
            self.scene.cycles.samples = 100
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
        sun1 = bpy.context.object
        sun1.location = (-10, 10, 10)
        sun1.data.energy = 0.3

        # 复制两个光源，以便均匀照明
        sun2 = sun1.copy()
        sun2.location = (10, 10, -10)
        bpy.context.collection.objects.link(sun2)
        sun2.data.energy = 0.3

        sun3 = sun1.copy()
        sun3.location = (10, -10, 10)
        bpy.context.collection.objects.link(sun3)
        sun3.data.energy = 0.3

        # sun4 = sun1.copy()
        # sun4.location = (10, -10, -10)
        # bpy.context.collection.objects.link(sun4)
        # sun4.data.energy = 0.5
        
        # sun5 = sun1.copy()
        # sun5.location = (-10, 10, 10)
        # bpy.context.collection.objects.link(sun5)
        # sun5.data.energy = 0.5
        # sun6 = sun1.copy()
        # sun6.location = (-10, 10, -10)
        # bpy.context.collection.objects.link(sun6)
        # sun6.data.energy = 0.5
        # sun7 = sun1.copy()
        # sun7.location = (-10, -10, 10)
        # bpy.context.collection.objects.link(sun7)
        # sun7.data.energy = 1.5
        # sun8 = sun1.copy()
        # sun8.location = (-10, -10, -10)
        # bpy.context.collection.objects.link(sun8)
        # sun8.data.energy = 1.5

    def import_mesh(self, fpath, scale=1., object_world_matrix=None):
        
        ext = os.path.splitext(fpath)[-1]
        if ext == '.obj':
            bpy.ops.import_scene.obj(filepath=str(fpath), split_mode='OFF')
        elif ext == '.ply':
            bpy.ops.import_mesh.ply(filepath=str(fpath))

        obj = bpy.context.selected_objects[0]
        # util.dump(bpy.context.selected_objects)

        if object_world_matrix is not None:
            obj.matrix_world = object_world_matrix

        # bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY', center='BOUNDS')
        # obj.location = (0., 0., 0.) # center the bounding box!

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

    def set_camera_pose(self, wTc, camera=None):
        self.camera_pose = wTc
        if hasattr(self, 'camera'):
            if camera is None:
                camera = self.camera
            loc, rot = convert_to_blender_pose(wTc)
            camera.location = loc
            camera.rotation_euler = rot

    def save_image(self, img_rgb, path):
        """保存 numpy 图像为 PNG 文件"""
        
        img_uint8 = (img_rgb * 255).astype(np.uint8)  # 转换为 0-255 范围
        img_pil = Image.fromarray(img_uint8)
        img_pil.save(path)

    def load_and_process_image(self, image_path):
        """
        读取 PNG 文件，并替换透明背景为白色
        """
        img = Image.open(image_path).convert("RGBA")  # 确保是 RGBA 模式
        img_array = np.array(img).astype(np.float32) / 255.0  # 归一化到 0-1

        # **拆分 RGBA 通道**
        rgb = img_array[:, :, :3]  # 取前三个通道
        alpha = img_array[:, :, 3]  # Alpha 通道

        # **将透明背景替换为白色**
        rgb[alpha == 0] = [1.0, 1.0, 1.0]  # 透明像素变成白色

        return rgb  # 返回 (H, W, 3) 格式的 numpy 数组

    def render(self, output_dir, blender_cam2world_matrices, write_cam_params=False):
        images = []
        if bpy.context.scene.camera is None:
            bpy.ops.object.camera_add()  # 添加新相机
            self.camera = bpy.context.object  # 获取新创建的相机
            bpy.context.scene.camera = self.camera  # 设置为场景默认相机
        else:
            self.camera = bpy.context.scene.camera
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
            # self.camera.matrix_world = blender_cam2world_matrices[i]
            self.set_camera_pose(blender_cam2world_matrices[i])

            # Render the object
            # if os.path.exists(os.path.join(img_dir, '%06d.png' % i)):
            #     img_rgb = self.load_and_process_image(os.path.join(img_dir, '%06d.png' % i))
            #     images.append(img_rgb)
            #     continue

            # Render the color image
            if os.path.exists(os.path.join(img_dir, '%06d.png'%i)):
                os.remove(os.path.join(img_dir, '%06d.png'%i))
            self.blender_renderer.filepath = os.path.join(img_dir, '%06d.png'%i)
            # **渲染图像**
            with stdout_redirected():
                bpy.ops.render.render(write_still=True)

            img_rgb = self.load_and_process_image(self.blender_renderer.filepath)


            images.append(img_rgb)
            self.save_image(img_rgb, self.blender_renderer.filepath)
        
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
        # meshes_to_remove = []
        # for ob in bpy.context.selected_objects:
        #     meshes_to_remove.append(ob.data)

        # bpy.ops.object.delete()

        # Remove the meshes from memory too
        # for mesh in meshes_to_remove:
        #     bpy.data.meshes.remove(mesh)
         # 转换为 numpy 数组
        images_np = np.stack(images, axis=0)  # (N, H, W, 3)
        
        # 也可以转换为 torch.Tensor
        images_tensor = torch.from_numpy(images_np).float()

        return images_tensor  # 返回 PyTorch Tensor 或者 `images_np` 作为 numpy 数组


if __name__=='__main__':
    mesh_fpath = "/mnt/hdd/zhengquan/Shapenet/ShapeNetCore.v2/02691156/1a04e3eab45ca15dd86060f189eb133/models/model_normalized.obj"
    output_dir = "/mnt/hdd/zhengquan/Shapenet/stanford_shapenet_render"

    instance_name = mesh_fpath.split('/')[-3]
    instance_dir = os.path.join(opt.output_dir, instance_name)

    renderer = blender_interface.BlenderInterface(resolution=128)