#!/usr/bin/env bash
# islet_patchseq 总控：按依赖顺序运行全部分析 stage 与出图脚本。
# 用法：bash run_all.sh [--resume|--force] [--skip=stageA,stageB]
#   --resume（默认）：results/<stage>/PROVENANCE.json 已存在则跳过该 stage
#   --force：全部重跑
# 环境：conda 环境 islet_patchseq（environment.yml）；03d 用 islet_patchseq_fm（environment_fm.yml），03e 的 scVI 部分用
#       islet_patchseq_scvi（environment_scvi.yml），均经 conda run 调用；神经网络对照需要 GPU（CUDA_VISIBLE_DEVICES 指定）。
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"
MODE=resume; SKIP=""
for a in "$@"; do
  case "$a" in
    --force) MODE=force ;;
    --resume) MODE=resume ;;
    --skip=*) SKIP="${a#--skip=}" ;;
  esac
done
CONDA_BASE="${CONDA_BASE:-$(conda info --base 2>/dev/null || echo "$HOME/miniconda3")}"
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate islet_patchseq
mkdir -p logs
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}   # 神经网络对照使用的 GPU

# stage 名 | 结果目录（用于 --resume 判断）| 命令
STAGES=(
  "00_download|00_download|python scripts/stages/00_download.py"
  "01_dataset|01_dataset|python scripts/stages/01_build_dataset.py"
  "02_variance|02_variance|python scripts/stages/02_variance.py"
  "03_benchmark|03_benchmark|python scripts/stages/03_benchmark.py --split donors"
  "03_benchmark_cellsplit|03_benchmark_cellsplit|python scripts/stages/03_benchmark.py --split cells"
  "03b_benchmark_nets|03b_benchmark_nets|python scripts/stages/03b_benchmark_nets.py"
  "03c_summary|03c_summary|python scripts/stages/03c_summarize.py"
  "03d_embed|03d_embed|conda run --no-capture-output -n islet_patchseq_fm python scripts/stages/03d_embed.py"   # 基础模型嵌入，独立环境 environment_fm.yml
  "04_context_tests|04_context_tests|python scripts/stages/04_context_tests.py"
  "04b_robustness|04b_robustness|python scripts/stages/04b_robustness.py"
  "04c_donor_null|04c_donor_null|python scripts/stages/04c_donor_null.py"
  "04d_reliability|04d_reliability|python scripts/stages/04d_reliability.py"
  "04e_training_size|04e_training_size|python scripts/stages/04e_training_size.py"
  "04f_technical|04f_technical|python scripts/stages/04f_technical.py"
  "04g_learning_curve|04g_learning_curve|python scripts/stages/04g_learning_curve.py"   # 按训练供体数的学习曲线（需 04b）
  "04h_centre_null|04h_centre_null|python scripts/stages/04h_centre_null.py"            # 按中心 / series 分层的供体层面置换（需 04b、04c）
  "05_donor_level|05_donor_level|python scripts/stages/05_donor_level.py"
  "06_external_hpap|06_external_hpap|python scripts/stages/06_external_hpap.py"
  "06b_external_ceiling|06b_external_ceiling|python scripts/stages/06b_external_ceiling.py"
  "06c_external_donor_level|06c_external_donor_level|python scripts/stages/06c_external_donor_level.py"
  "06d_calibration|06d_calibration|python scripts/stages/06d_calibration.py"   # 绝对尺度校准与共同供体跨平台比较（只读已存预测）
  "03e_sota|03e_sota|conda run --no-capture-output -n islet_patchseq_scvi python scripts/stages/03e_sota.py --part scvi_dev && python scripts/stages/03e_sota.py --part hvg && python scripts/stages/03e_sota.py --part nets && python scripts/stages/03e_sota.py --part scvi_heads && python scripts/stages/03e_sota.py --part emb && python scripts/stages/03e_sota.py --part summary && conda run --no-capture-output -n islet_patchseq_scvi python scripts/stages/03e_sota.py --part scvi_external && python scripts/stages/03e_sota.py --part external"   # 新增对照（scVI 部分用环境 environment_scvi.yml）；需 03b 预测、03d 嵌入与 06 外部预测，故排在 06c 之后
  "03f_finetune|03f_finetune|conda run --no-capture-output -n islet_patchseq_fm python scripts/stages/03f_finetune.py --part cv && conda run --no-capture-output -n islet_patchseq_fm python scripts/stages/03f_finetune.py --part summary && conda run --no-capture-output -n islet_patchseq_fm python scripts/stages/03f_finetune.py --part external"   # 基础模型微调（GPU，config finetune.gpus）；需 03d 编码、03e 冻结嵌入预测与 06 外部预测
  "04i_multiplicity|04i_multiplicity|python scripts/stages/04i_multiplicity.py"          # 模型比较的 BH 校正（需 03c、03e、03f）
  "03g_ps_faithful|03g_ps_faithful|python scripts/stages/03g_ps_faithful.py --part cv && python scripts/stages/03g_ps_faithful.py --part summary"   # 原始预测基因集 + kNN 的折内忠实重建（GPU，config ps_faithful.gpus）；需 Stage 03 两种划分的预测
  "03h_geneset_rawcounts|03h_geneset_rawcounts|python scripts/stages/03h_geneset_rawcounts.py --part geneset && python scripts/stages/03h_geneset_rawcounts.py --part raw_prep && python scripts/stages/03h_geneset_rawcounts.py --part raw_cv && conda run --no-capture-output -n islet_patchseq_fm python scripts/stages/03h_geneset_rawcounts.py --part raw_embed && python scripts/stages/03h_geneset_rawcounts.py --part raw_heads && python scripts/stages/03h_geneset_rawcounts.py --part summary"   # 同一基因集对照与 GEO 原始计数敏感性；需 03d 嵌入与 03e 预测
  "08_programmes|08_programmes|python scripts/stages/08_programmes.py"
  "08b_programme_qc|08b_programme_qc|python scripts/stages/08b_programme_qc.py"   # 需 08 的基因关联与富集结果
  "09_t2d_function|09_t2d_function|python scripts/stages/09_t2d_function.py"
  "09b_t2d_sensitivity|09b_t2d_sensitivity|python scripts/stages/09b_t2d_sensitivity.py"   # 需 09 的主效应表（等效性界）
  "04j_coupling|04j_coupling|python scripts/stages/04j_coupling.py"                      # Ca2+ 可解释部分与偶联效率的分解（含 T2D 比较）
  "09c_t2d_hidden|09c_t2d_hidden|python scripts/stages/09c_t2d_hidden.py"                # ND 训练预测值中的 T2D 差异
  "09d_t2d_signature|09d_t2d_signature|python scripts/stages/09d_t2d_signature.py"       # 留出供体 T2D 转录特征与中介（需 09c 预测）
  "09e_t2d_conditioned|09e_t2d_conditioned|python scripts/stages/09e_t2d_conditioned.py" # 以转录组为条件的 T2D 效应、阳性对照与半合成检验
  "09f_t2d_robust|09f_t2d_robust|python scripts/stages/09f_t2d_robust.py --force"         # 携带比例的边界：供体间/供体内分解、安慰剂与置换零分布、模拟携带者、外部与重抽样特征（需 09c 预测、外部特征文件）
  "09g_t2d_mechanism|09g_t2d_mechanism|python scripts/stages/09g_t2d_mechanism.py"     # 胞吐/颗粒基因集与 ETV1、Ca2+ 测量误差校正、早晚期胞吐、ND 训练诊断、胰岛胰岛素含量
  "09h_t2d_glucose|09h_t2d_glucose|python scripts/stages/09h_t2d_glucose.py"           # 各葡萄糖浓度内的两部分模型
  "11_islet_secretion|11_islet_secretion|python scripts/stages/11_islet_secretion.py"       # 供体层面：单细胞功能与同一供体胰岛灌流 / GSIS 胰岛素分泌（需 03 逐细胞预测）
  "10_tables|10_tables|python scripts/stages/10_tables.py"
)

for entry in "${STAGES[@]}"; do
  IFS='|' read -r name dir cmd <<< "$entry"
  if [[ ",$SKIP," == *",$name,"* ]]; then echo "[skip] $name"; continue; fi
  if [[ "$MODE" == resume && -f "results/$dir/PROVENANCE.json" ]]; then echo "[done] $name"; continue; fi
  echo "[run ] $name"
  eval "$cmd" 2>&1 | tee "logs/$name.log"
done

echo "[figs] figures -> figures/"
for f in scripts/figures/Fig[0-9]*.py scripts/figures/FigS*.py; do
  [[ -f "$f" ]] || continue
  python "$f" 2>&1 | tee -a logs/figures.log
done
echo "all stages finished"
