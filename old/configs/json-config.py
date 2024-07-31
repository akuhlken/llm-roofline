import json

def _get_data(file):
    with open(file, 'r') as f:
        data = json.load(f)
        return data

# JSON config methods:
def get_num_attention_heads(json):
    return _get_data(json)['num_attention_heads']

def get_hidden_size(json):
    return _get_data(json)['hidden_size']

def get_num_key_value_heads(json):
    return _get_data(json)['num_key_value_heads']

def get_norm_layers(json):
    return _get_data(json)['norm_layers']

def get_num_hidden_layers(json):
    return _get_data(json)['num_hidden_layers']

def get_intermediate_size(json):
    return _get_data(json)['intermediate_size']

def get_vocab_size(json):
    return _get_data(json)['vocab_size']

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
        "gate_proj":[hidden_size, intermediate_size], #TODO I belive not all LLM have this
        "up_proj":[hidden_size,intermediate_size],
        "down_proj":[intermediate_size, hidden_size],
    }