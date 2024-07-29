
# JSON config methods:
# TODO: extract correct values from json input)
def get_num_attention_heads(json):
    print(json)
    return 32

def get_hidden_size(json):
    return 4096

def get_num_key_value_heads(json):
    return 32

def get_norm_layers(json):
    return ["attn_norm", "mlp_norm"] #TODO: Can we store this in the JSON

def get_num_hidden_layers(json):
    return 32

def get_intermediate_size(json):
    return 11008

def get_vocab_size(json):
    return 32000

def post_process(json,args):
    hiddensize=get_hidden_size(json)
    vocab_size=get_vocab_size(json)
    layers=[]
    for stage in ["prefill", "decode"]:
        layers.append({
            'name': 'lm_head',
            'stage':stage,
            'OPs':args['batchsize']*hiddensize*vocab_size*1,
            'load_weight':hiddensize*vocab_size *args['w_byte'],
            'load_act':hiddensize*args['a_byte'],
            'store_act':vocab_size*args['a_byte'],
        })
    return layers

def get_linear_layers(json):
    hidden_size=get_hidden_size(json)
    intermediate_size=get_intermediate_size(json)
    key_value_heads=get_num_key_value_heads(json)
    attention_heads=get_num_attention_heads(json)
    return {
        "q_proj":[hidden_size, hidden_size],
        "k_proj":[hidden_size, hidden_size*key_value_heads/attention_heads],
        "v_proj":[hidden_size, hidden_size*key_value_heads/attention_heads],
        "out_proj":[hidden_size, hidden_size],
        "gate_proj":[hidden_size, intermediate_size],
        "up_proj":[hidden_size,intermediate_size],
        "down_proj":[intermediate_size, hidden_size],
    }