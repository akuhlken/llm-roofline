from model_analyzer import ModelAnalyzer

def json_net_analysis(model_json, hardware, batchsize=1, seqlen=512, 
                    w_bit=16, a_bit=16, kv_bit=16, use_flashattention=False):
    analyzer = ModelAnalyzer("foo", hardware, config_file="configs/json-config.py", source=model_json)
    results = analyzer.analyze(
        batchsize=batchsize,
        seqlen=seqlen,
        w_bit=w_bit,
        a_bit=a_bit,
        kv_bit=kv_bit,
        use_flashattention=use_flashattention
    )
    return results
    
def huggingface_net_analysis(model_name, hardware, batchsize=1, seqlen=512, 
                    w_bit=16, a_bit=16, kv_bit=16, use_flashattention=False):
    analyzer = ModelAnalyzer(model_name, hardware, source="huggingface")
    results = analyzer.analyze(
        batchsize=batchsize,
        seqlen=seqlen,
        w_bit=w_bit,
        a_bit=a_bit,
        kv_bit=kv_bit,
        use_flashattention=use_flashattention
    )
    return results

"""
results is a dict with the following format:

{
    "decode": {
            "layer_name": {
                    "OPs": "",
                    "memory_access": "",
                    "arithmetic_intensity": "",
                    "performance": "",
                    "bound": "",
                    "load_weight": "",
                    "load_act": "",
                    "store_act": "",
                    "load_kv_cache": "",
                    "store_kv_cache": "",
                    "inference_time": ""
            }
    },
    "prefill": {
            "layer_name": {
                    "OPs": "",
                    "memory_access": "",
                    "arithmetic_intensity": "",
                    "performance": "",
                    "bound": "",
                    "load_weight": "",
                    "load_act": "",
                    "store_act": "",
                    "load_kv_cache": "",
                    "store_kv_cache": "",
                    "inference_time": ""
            }
    },
    "total_results": {
        "decode": {},
        "prefill": {}
    }
}
"""