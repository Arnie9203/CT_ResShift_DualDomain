import numpy as np

def generate_ct_volume(file_path: str, height=256, width=256, slices=421):
    """
    生成一个体数据 (height x width x slices) 的随机CT体积，并保存为Numpy文件
    :param file_path: 保存路径，例如 'my_ct_volume.npy'
    :param height: 图像高度
    :param width: 图像宽度
    :param slices: 切片数（必须 >= 421）
    """
    assert slices >= 421, "切片数必须大于等于421"
    volume = np.random.rand(height, width, slices).astype(np.float32)  # 生成0-1随机浮点数据
    np.save(file_path, volume)
    print(f"已生成体数据，shape={volume.shape}，保存至 {file_path}")

if __name__ == "__main__":
    generate_ct_volume("my_ct_volume.npy", height=256, width=256, slices=500)  # 这里的500可修改
