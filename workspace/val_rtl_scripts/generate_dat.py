import json

# 設定平行度參數
P = 4

# 輸入與輸出檔名
input_json = '../PWConv_header.json'
output_dat = 'layers_params.dat'

def main():
    try:
        with open(input_json, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"找不到檔案: {input_json}")
        return

    layers = data.get("layers", [])
    if not layers:
        print("在 JSON 檔中找不到 layers 資料！")
        return

    # 準備寫入 .dat 檔案
    with open(output_dat, 'w', encoding='utf-8') as f_out:
        for layer in layers:
            idx = layer.get("layer_index")
            name = layer.get("name")
            fields = layer.get("fields", {})

            # 抓取數值
            H = fields.get("H", {}).get("value", 0)
            W = fields.get("W", {}).get("value", 0)
            C_in = fields.get("C_in", {}).get("value", 0)
            C_out = fields.get("C_out", {}).get("value", 0)

            # 計算 C_out_group_max
            c_out_group_max = C_out // P

            # 寫入檔案 (格式: H W C_in C_out_group_max)
            line = f"{H} {W} {C_in} {c_out_group_max}"
            f_out.write(line + "\n")

            # 同時印在 Console 方便核對
            print(f"Layer {idx:2d} ({name}): H={H}, W={W}, C_in={C_in}, C_out={C_out} -> Write: {line}")

    print(f"\n✅ 成功產生 {output_dat}！(共 {len(layers)} 層)")

if __name__ == '__main__':
    main()