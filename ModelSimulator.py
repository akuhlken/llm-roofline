import os
import importlib
import math
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
from utils import str_number, str_number_time, get_bottleneck, get_hardware_info

ALL_DATA_NAMES = [
    "OPs",
    "memory_access",
    "load_weight",
    "load_act",
    "store_act",
    "load_kv_cache",
    "store_kv_cache",
    "inference_time",
]

class ModelSimulator():

    def __init__(self, model, source="huggingface"):
        self.model = model
        self.source = source
        self.setModelParams()

        # temporary variables
        self.results = None
        self.w_bit = None
        self.a_bit = None
        self.kv_bit = None
        self.batchsize = None
        self.seqlen = None

    def simulate(
        self,
        hardware,
        seqlen,
        batchsize,
        w_bit=16,
        a_bit=16,
        kv_bit=None,
        use_flashattention=False,
        kv_token_ratio=1,
    ):
        assert seqlen > 0
        assert batchsize > 0
        self.results = {"decode": {}, "prefill": {}}
        if kv_bit is None:
            kv_bit = a_bit
        self.w_bit = w_bit
        self.a_bit = a_bit
        self.kv_bit = kv_bit
        self.batchsize = batchsize
        self.seqlen = seqlen

        w_byte = self.w_bit / 8
        a_byte = self.a_bit / 8
        kv_byte = self.kv_bit / 8

        config = self.config
        model_params = self.model_params
        num_attention_heads = config.get_num_attention_heads(model_params)
        hidden_size = config.get_hidden_size(model_params)
        num_key_value_heads = config.get_num_key_value_heads(model_params)
        num_hidden_layers = config.get_num_hidden_layers(model_params)

        for name, (ic, oc) in config.get_linear_layers(model_params).items():
            # for linear layers
            is_kv_proj = name in ["k_proj", "v_proj"]
            is_normal_proj = not is_kv_proj
            self._analyze_to_results(
                hardware,
                "decode",
                name,
                OPs=ic * oc * batchsize * 2,
                load_weight=ic * oc * w_byte,
                load_act=ic * batchsize * a_byte,
                store_act=0 if is_kv_proj else oc * batchsize * a_byte,
                load_kv_cache=0,
                store_kv_cache=(0 if is_normal_proj else oc * batchsize * kv_byte),
            )
            # for prefill
            self._analyze_to_results(
                hardware,
                "prefill",
                name,
                OPs=ic * oc * batchsize * seqlen * 2,
                load_weight=ic * oc * w_byte,
                load_act=ic * batchsize * seqlen * a_byte,
                store_act=(0 if is_kv_proj else oc * batchsize * seqlen * a_byte),
                load_kv_cache=0,
                store_kv_cache=(
                    0 if is_normal_proj else oc * batchsize * seqlen * kv_byte
                ),
            )

        # for attention
        head_size = hidden_size // num_attention_heads
        # for decode
        qk_matmul_OPs = seqlen * head_size * num_attention_heads * batchsize * 2
        sv_matmul_OPs = 1 * head_size * seqlen * num_attention_heads * batchsize * 2
        softmax_OPs = batchsize * num_attention_heads * seqlen * 1 * 5
        if use_flashattention:
            name = f"fused_attention"
            bandwidth, max_OPS, onchip_buffer = self.get_hardware_info()
            # flashattention-2 https://arxiv.org/pdf/2307.08691.pdf
            block_size_r = min(
                math.ceil(onchip_buffer / (kv_byte * head_size)), head_size
            )
            n_blocks_r = math.ceil(1 / block_size_r)
            q_numel = (1) * head_size * batchsize * num_attention_heads * a_byte
            o_numel = 1 * seqlen * batchsize * num_attention_heads * a_byte
            self._analyze_to_results(
                hardware,
                "decode",
                name,
                OPs=qk_matmul_OPs + sv_matmul_OPs + softmax_OPs,
                load_weight=0,
                load_act=q_numel,
                store_act=o_numel * 2,  # initialize O and save O
                load_kv_cache=n_blocks_r
                * (seqlen)
                * head_size
                * batchsize
                * num_attention_heads
                * kv_byte,
                store_kv_cache=0,
            )

        else:
            name = f"qk_matmul"
            self._analyze_to_results(
                hardware,
                "decode",
                name,
                OPs=qk_matmul_OPs,
                load_weight=0,
                load_act=(1) * head_size * batchsize * num_attention_heads * a_byte,
                store_act=1 * seqlen * batchsize * num_attention_heads * a_byte,
                load_kv_cache=(seqlen)
                * head_size
                * batchsize
                * num_key_value_heads
                * kv_byte,
                store_kv_cache=0,
            )
            name = f"sv_matmul"
            self._analyze_to_results(
                hardware,
                "decode",
                name,
                OPs=sv_matmul_OPs,
                load_weight=0,
                load_act=(1 * seqlen * batchsize * num_attention_heads) * a_byte,
                store_act=1 * head_size * batchsize * num_attention_heads * a_byte,
                load_kv_cache=(seqlen * head_size * batchsize * num_key_value_heads)
                * kv_byte,
                store_kv_cache=0,
            )

            name = f"softmax"
            # max sub exp sum div
            self._analyze_to_results(
                hardware,
                "decode",
                name,
                OPs=softmax_OPs,
                load_weight=0,
                load_act=batchsize * num_attention_heads * seqlen * 1 * a_byte,
                store_act=batchsize * num_attention_heads * seqlen * 1 * a_byte,
                load_kv_cache=0,
                store_kv_cache=0,
            )

        for name in config.get_norm_layers(model_params):
            # sum sub pow sum div mul add
            self._analyze_to_results(
                hardware,
                "decode",
                name,
                OPs=batchsize * hidden_size * 1 * 7,
                load_weight=0,
                load_act=batchsize * hidden_size * 1 * a_byte,
                store_act=batchsize * hidden_size * 1 * a_byte,
                load_kv_cache=0,
                store_kv_cache=0,
            )

        for name in ["attn_add", "mlp_add"]:
            self._analyze_to_results(
                hardware,
                "decode",
                name,
                OPs=batchsize * hidden_size * 1,
                load_weight=0,
                load_act=batchsize * hidden_size * 1 * a_byte,
                store_act=batchsize * hidden_size * 1 * a_byte,
                load_kv_cache=0,
                store_kv_cache=0,
            )
        for name in ["mlp_act"]:
            self._analyze_to_results(
                hardware,
                "decode",
                name,
                OPs=batchsize * hidden_size * 1 * 2,
                load_weight=0,
                load_act=batchsize * hidden_size * 1 * a_byte * 2,
                store_act=batchsize * hidden_size * 1 * a_byte,
                load_kv_cache=0,
                store_kv_cache=0,
            )

        # for prefill
        qk_matmul_OPs = (
            seqlen * seqlen * head_size * num_attention_heads * batchsize * 2
        )
        sv_matmul_OPs = (
            seqlen * head_size * seqlen * num_attention_heads * batchsize * 2
        )
        softmax_OPs = batchsize * num_attention_heads * seqlen * seqlen * 5
        if use_flashattention:
            name = f"fused_attention"
            bandwidth, max_OPS, onchip_buffer = self.get_hardware_info()
            # flashattention-2 https://arxiv.org/pdf/2307.08691.pdf
            block_size_r = min(
                math.ceil(onchip_buffer / (kv_byte * head_size)), head_size
            )
            n_blocks_r = math.ceil(seqlen / block_size_r)
            q_numel = seqlen * head_size * batchsize * num_attention_heads * a_byte
            o_numel = seqlen * seqlen * batchsize * num_attention_heads * a_byte
            self._analyze_to_results(
                hardware,
                "prefill",
                name,
                OPs=qk_matmul_OPs + sv_matmul_OPs + softmax_OPs,
                load_weight=0,
                load_act=q_numel,
                store_act=o_numel * 2,  # initialize O and save O
                load_kv_cache=n_blocks_r
                * (seqlen)
                * head_size
                * batchsize
                * num_attention_heads
                * kv_byte,
                store_kv_cache=0,
            )
        else:
            name = f"qk_matmul"
            self._analyze_to_results(
                hardware,
                "prefill",
                name,
                OPs=qk_matmul_OPs,
                load_weight=0,
                load_act=seqlen * head_size * batchsize * num_key_value_heads * a_byte,
                store_act=seqlen * seqlen * batchsize * num_attention_heads * a_byte,
                load_kv_cache=seqlen
                * head_size
                * batchsize
                * num_key_value_heads
                * kv_byte,
                store_kv_cache=0,
            )
            name = f"sv_matmul"
            self._analyze_to_results(
                hardware,
                "prefill",
                name,
                OPs=sv_matmul_OPs,
                load_weight=0,
                load_act=seqlen * seqlen * batchsize * num_attention_heads * a_byte,
                store_act=seqlen * head_size * batchsize * num_attention_heads * a_byte,
                load_kv_cache=seqlen
                * head_size
                * batchsize
                * num_key_value_heads
                * kv_byte,
                store_kv_cache=0,
            )
            name = f"softmax"
            self._analyze_to_results(
                hardware,
                "prefill",
                name,
                OPs=softmax_OPs,
                load_weight=0,
                load_act=batchsize * num_attention_heads * seqlen * seqlen * a_byte,
                store_act=batchsize * num_attention_heads * seqlen * seqlen * a_byte,
                load_kv_cache=0,
                store_kv_cache=0,
            )
        for name in config.get_norm_layers(model_params):
            self._analyze_to_results(
                hardware,
                "prefill",
                name,
                OPs=batchsize * hidden_size * seqlen * 7,
                load_weight=0,
                load_act=batchsize * hidden_size * seqlen * a_byte,
                store_act=batchsize * hidden_size * seqlen * a_byte,
                load_kv_cache=0,
                store_kv_cache=0,
            )
        for name in ["attn_add", "mlp_add"]:
            self._analyze_to_results(
                hardware,
                "prefill",
                name,
                OPs=batchsize * hidden_size * seqlen * 1,
                load_weight=0,
                load_act=batchsize * hidden_size * seqlen * a_byte,
                store_act=batchsize * hidden_size * seqlen * a_byte,
                load_kv_cache=0,
                store_kv_cache=0,
            )
        for name in ["mlp_act"]:
            self._analyze_to_results(
                hardware,
                "prefill",
                name,
                OPs=batchsize * hidden_size * seqlen * 1 * 2,
                load_weight=0,
                load_act=batchsize * hidden_size * seqlen * a_byte * 2,
                store_act=batchsize * hidden_size * seqlen * a_byte,
                load_kv_cache=0,
                store_kv_cache=0,
            )

        # compute total
        total_results = {"decode": {}, "prefill": {}}
        for data_name in ALL_DATA_NAMES:
            total_results["decode"][data_name] = 0
            total_results["prefill"][data_name] = 0
        for stage in ["decode", "prefill"]:
            for layer_name, result in self.results[stage].items():
                for data_name in ALL_DATA_NAMES:
                    total_results[stage][data_name] += (
                        result[data_name] * num_hidden_layers
                    )

        # memory footprint
        weight_kv_footprint = (
            total_results["prefill"]["load_weight"]
            + total_results["prefill"]["store_kv_cache"]
        )
        decode_tmp_act = 0
        for layer_name, result in self.results["decode"].items():
            decode_tmp_act += result["store_act"]
        total_results["decode"]["memory_consumption"] = (
            decode_tmp_act + weight_kv_footprint
        )
        total_results["decode"]["memory_consumption_tmp_act"] = decode_tmp_act
        total_results["decode"]["memory_consumption_weight"] = total_results["prefill"][
            "load_weight"
        ]
        total_results["decode"]["memory_consumption_kv_cache"] = total_results[
            "prefill"
        ]["store_kv_cache"]
        prefill_tmp_act = 0
        for layer_name, result in self.results["prefill"].items():
            prefill_tmp_act += result["store_act"]
        total_results["prefill"]["memory_consumption"] = (
            prefill_tmp_act + weight_kv_footprint
        )
        total_results["prefill"]["memory_consumption_tmp_act"] = prefill_tmp_act
        total_results["prefill"]["memory_consumption_weight"] = total_results[
            "prefill"
        ]["load_weight"]
        total_results["prefill"]["memory_consumption_kv_cache"] = total_results[
            "prefill"
        ]["store_kv_cache"]

        # lm_head
        name = "lm_head"
        args={
            'batchsize':batchsize,
            'a_byte':a_byte,
            'w_byte':w_byte
        }
        for layer_info in self.config.post_process(self.model_params, args):
            self._analyze_to_results(hardware,
                **layer_info)
            for data_name in ALL_DATA_NAMES:
                total_results[layer_info['stage']][data_name] += self.results[layer_info['stage']][layer_info['name']][data_name]

        self.results["total_results"] = total_results
        return self.results

    def getRoofline(self):
        pass

    def saveCSV(self, save_path=None):
        pass

    def setModelParams(self):
        #TODO: Add support for more model formats
        if config_file is None:
            # get the current file directory
            current_dir = os.path.dirname(os.path.abspath(__file__))
            # auto search the config
            for file in os.listdir(current_dir + "/configs"):
                if file.endswith(".py") and file.replace(".py", "") in self.model:
                    config_file = "configs/" + file
                # print(f"auto search config file {config_file} {file} {model_id}")
        assert (
            config_file is not None
        ), "config file is not found, please specify it manually."
        print(f"use config file {config_file} for {self.model}")
        if self.source == "huggingface":
            self.model_params = AutoConfig.from_pretrained(
                self.model, trust_remote_code=True
            )
        else:
            if not os.path.exists(f"model_params/{self.source}.py"):
                raise Exception(f"model_params/{self.source}.py is not found")
            # from model_params.DiT import model_params
            module=importlib.import_module(f"model_params.{self.source}")
            self.model_params = module.model_params[self.model]
        self.config = importlib.import_module(
            config_file.replace("/", ".").replace(".py", "")
        )

    def _analyze_to_results(self, hardware, stage, name, OPs=0, load_weight=0, load_act=0,
                    store_act=0, load_kv_cache=0, store_kv_cache=0,
    ):
        bandwidth, max_OPS, onchip_buffer = get_hardware_info(hardware, 
                                            self.w_bit, self.a_bit, self.kv_bit)
        memory_access = (
            load_weight + load_act + store_act + load_kv_cache + store_kv_cache
        )
        arithmetic_intensity, performance, bound = get_bottleneck(
            bandwidth, max_OPS, OPs, memory_access
        )
        inference_time = OPs / performance
        self.results[stage][name] = {
            "OPs": OPs,
            "memory_access": memory_access,
            "arithmetic_intensity": arithmetic_intensity,
            "performance": performance,
            "bound": bound,
            "load_weight": load_weight,
            "load_act": load_act,
            "store_act": store_act,
            "load_kv_cache": load_kv_cache,
            "store_kv_cache": store_kv_cache,
            "inference_time": inference_time,
        }

    