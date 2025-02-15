
# # 把每一行中间的制表符替换成空格
with open(input_file_path, 'r') as file:
    lines = file.readlines()
# new_list = []
# for i in range(len(lines)):
#     lines[i] = lines[i].replace('\t', ' ')
#     new_list.append(lines[i])
#
# with open(output_file_path, 'w') as file:
#     for line in new_list:
#         file.write(line)

# # 提取无向边列表并转换为有向边列表
directed_edgelist = []
for line in lines:
    parts = line.strip('\t').split()
    if len(parts) == 2:  # 确保每行有两个节点ID
        try:
            u = int(parts[0])
            v = int(parts[1])
            directed_edgelist.append((u, v))
            directed_edgelist.append((v, u))
        except ValueError:
            # 跳过包含无效数据的行
            continue

# 将有向边列表保存到txt文件中
with open(output_file_path, 'w') as file:
    for edge in directed_edgelist:
        file.write(f"{edge[0]} {edge[1]}\n")

print("Directed edgelist saved to", output_file_path)