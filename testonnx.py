import onnx

# 安全加载模型结构（不加载外部权重）
model = onnx.load("uploads/yslgjl_go_model.onnx", load_external_data=False)

print("🔍 检查模型中的外部数据引用...")

for tensor in model.graph.initializer:
    if tensor.HasField('data_location') and tensor.data_location == onnx.TensorProto.EXTERNAL:
        # 外部数据信息存储在 external_data 列表中
        external_info = {entry.key: entry.value for entry in tensor.external_data}
        location = external_info.get('location', '未知')
        print(f"Tensor '{tensor.name}' 引用了外部文件: {location}")