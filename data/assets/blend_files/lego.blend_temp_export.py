
import bpy

bpy.ops.wm.open_mainfile(filepath=r"/attached/remote-home2/zzq/04-fep-nbv/data/assets/blend_files/lego.blend")

# 只选择第一个 MESH 对象
for obj in bpy.data.objects:
    obj.select_set(False)

for obj in bpy.data.objects:
    if obj.type == 'MESH':
        obj.select_set(True)
        bpy.context.view_layer.objects.active = obj
        break

# 导出选中的对象为 .ply 文件
bpy.ops.export_mesh.ply(filepath=r"/attached/remote-home2/zzq/04-fep-nbv/data/assets/blend_files/lego.blend.ply", use_selection=True)
