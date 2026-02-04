import os

def list_sorted_subfolders(root_dir):
    # 获取 root_dir 下所有的子文件夹
    subfolders = [f for f in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, f))]

    # 按名称排序
    subfolder = subfolders.sort()
    subfolders = subfolders[:105]

    # 返回完整路径
    return [os.path.join(root_dir, f) for f in subfolders]

def count_viewpoint_files(root_dir):
    subfolders = [f for f in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, f))]
    subfolders.sort()

    for folder in subfolders:
        image_dir = os.path.join(root_dir, folder, "images")
        count = 0
        if os.path.exists(image_dir):
            files = os.listdir(image_dir)
            count = len([f for f in files if f.startswith("viewpoint_example")])
        if count!=48:
            print(f"{folder}: {count} files starting with 'viewpoint_example'")

if __name__ == "__main__":
    # 示例路径（你可以替换为自己的路径）
    root = "/attached/remote-home2/zzq/data/shapenet/ShapeNetCore.v2/04530566"
    # count_viewpoint_files(root)

    sorted_folders = list_sorted_subfolders(root)
    for folder in sorted_folders:
        print(folder)