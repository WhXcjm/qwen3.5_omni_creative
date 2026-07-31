该文档用于在qwen3.5模型上适配ascendC GDN算子，经测试，相比triton GDN，端到端单步性能可以提升8%~10%；

## 环境配置
### CANN
要求版本高于8.5.0，下载链接：https://www.hiascend.com/developer/download/community/result?module=cann&cann=8.5.1

### Pytorch Adaptor
对版本严要求，使用26.1.0.beta，下载链接：https://gitcode.com/Ascend/pytorch/releases

## FLA-npu安装及qwen3.5-moe适配
AscendC GDN通过增量编译的方式将相关内容添加到CANN自定义算子及PTA适配层中；

1、源码链接：github.com/flashserve/flash-linear-attention-npu

2、配置环境变量：source $CANN_INSTALL_PATH/ascend-toolkit/set_env.sh

3、编译GDN算子run包并安装
```shell
# 编译命令，注意 --soc=${soc_version} 需指定为当前机器芯片类型 {ascend910b/ascend910_93/ascend950}
bash build.sh --soc=ascend910_93 --pkg --vendor_name=fla_npu

# 安装 run 包（custom 包名：fla-npu-<vendor>_linux-<arch>.run）
./build_out/fla-npu-*.run
```

4、编译torch适配whl包并安装
```shell
cd torch_custom/fla_npu
bash build.sh  # 一键编译安装脚本，先调用torchnpugen自动接入算子，再运行setup编whl包，最后安装whl包
```

5、单算子测试
```shell
# 运行测试
cd torch_custom/fla_npu/test
bash test.sh  # 包含所有GDN单算子的一键测试脚本
```

6、算子调用方式参考
使用torch.ops.npu.npu_{算子名称}()调用对应算子，具体可参考torch_custom/fla_npu/test下面的对应算子测试脚本
例如：
```shell
import torch
import torch_npu
import fla_npu

torch.ops.npu.npu_chunk_bwd_dv_local(...)
```

7、AscendC GDN算子在qwen3.5-moe模型上的适配
当前只适配了qwen3.5-moe模型，需要适配其他模型请自行参考；

适配本体代码：
```shell
mindspeed-mm/fsdp/models/qwen3_5/flash_gated_delta_rule.py
```

模型适配代码：
```shell
mindspeed-mm/fsdp/models/qwen3_5_moe/modeling_qwen3_5_moe.py

...
self.use_triton_gdn = config.use_triton_gdn
self.use_ascend_gdn = config.use_ascend_gdn

if self.use_triton_gdn and IS_NPU_AVAILABLE:
    from mindspeed_mm.fsdp.models.qwen3_5.chunk_gated_delta_rule import chunk_gated_delta_rule
    print_rank(logger.info, "Qwen3_5MoeGatedDeltaNet use NPU fused ops")
    self.chunk_gated_delta_rule = chunk_gated_delta_rule
elif self.use_ascend_gdn and IS_NPU_AVAILABLE:
    from mindspeed_mm.fsdp.models.qwen3_5.flash_gated_delta_rule import flash_gated_delta_rule
    print_rank(logger.info, "Qwen3_5MoeGatedDeltaNet use NPU ascendC fused ops")
    self.chunk_gated_delta_rule = flash_gated_delta_rule
...
```
