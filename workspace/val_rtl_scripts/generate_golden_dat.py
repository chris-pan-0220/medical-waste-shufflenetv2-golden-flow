import json

def load_json(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"❌ 無法讀取 {filepath}: {e}")
        return None

def main():
    header_data = load_json('../PWConv_header.json')
    weight_data = load_json('../PWConv_weight.json')
    bias_data   = load_json('../PWConv_bias.json')

    if not (header_data and weight_data and bias_data):
        return

    P = 4
    num_layers = header_data.get('num_layers', 15)

    with open('layer_configs.dat', 'w') as f_cfg, open('golden_addrs.dat', 'w') as f_gold:
        for i in range(num_layers):
            # 1. 取得 Header 參數
            fields = header_data['layers'][i]['fields']
            H = fields['H']['value']
            W = fields['W']['value']
            C_in = fields['C_in']['value']
            C_out = fields['C_out']['value']
            w_base = fields['weight_base']['value']
            b_base = fields['bias_base']['value']
            
            c_out_group_max = C_out // P
            
            # 寫入設定檔 (H W C_in C_out_group_max w_base b_base)
            f_cfg.write(f"{H} {W} {C_in} {c_out_group_max} {w_base} {b_base}\n")

            # 2. 建立查詢字典，方便透過 (group, cin) 找位址
            w_words = weight_data['layers'][i]['words']
            b_words = bias_data['layers'][i]['words']
            
            w_dict = {(w['c_out_group'], w['c_in']): w['word_addr'] for w in w_words}
            b_dict = {b['c_out_group']: b['word_addr'] for b in b_words}

            # 3. 按照硬體迴圈順序展開 (先掃 C_in, 再掃 C_out_group)
            for grp in range(c_out_group_max):
                golden_bias_addr = b_dict[grp]
                for cin in range(C_in):
                    golden_weight_addr = w_dict[(grp, cin)]
                    # 寫入答案檔 (weight_addr bias_addr)
                    f_gold.write(f"{golden_weight_addr} {golden_bias_addr}\n")
                    
    print("✅ 成功產生 layer_configs.dat 與 golden_addrs.dat！")

if __name__ == '__main__':
    main()