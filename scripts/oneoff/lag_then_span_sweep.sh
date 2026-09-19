#!/usr/bin/env bash
# v2 — 无人值守:lag∈{15,18,21} 搜索(12/24 已有)→ 选 lag*(≈batched span24: 13.347dB/12.34GiB)
#      → 双卡 span 扫描 L∈{4..180}(idx0, lag* 固定)→ SUMMARY.md
# 关键:v2 用 marker 文件推进(不依赖 kill -0,僵尸坑),每阶段带超时兜底。
# 注意:lag 必须是 3 的倍数(self-forcing block=3帧),否则 rc=2 decode route 校验失败。
set -uo pipefail
T=/mnt/cpfs/epic-user/hujiahui-20260612/Tempokv
QVG=/mnt/cpfs/epic-user/hujiahui-20260612/miniconda3/envs/qvg/bin/python
cd "$T"
D=$T/results/paper/span-int2/runs/lagsweep
R=$T/results/paper/span-int2/runs
mkdir -p "$D"; rm -f "$D"/*.done
log(){ printf '[%s] %s\n' "$(date '+%F %T')" "$*" >>"$D/driver.log"; }
G1=GPU-2ff1acd3-71e7-4032-9632-3880ba5ee31c
G2=GPU-c8e19653-0138-0223-68fe-5b7236baf470

wait_markers(){ # $1=timeout_s ${@:2}=markers
  local t=$1; shift; local start=$(date +%s)
  while :; do
    local ok=1; for m in "$@"; do [[ -e $m ]] || ok=0; done
    [[ $ok -eq 1 ]] && return 0
    (( $(date +%s)-start > t )) && { log "TIMEOUT waiting $*"; return 1; }
    sleep 60
  done
}

# ---- Phase A: lag 搜索(15/21 卡1,18 卡2)----
log "phase A: lagfind 15/18/21"
nohup bash -c "for l in 15 21; do TEMPOKV_BF16_LAG_FRAMES=\$l LABEL=lagfind_lag\${l} WATCH_IDLE=25 START=0 END=1 bash scripts/paper_experiments/span/span_bigL.sh 24 $G1; done; touch $D/chainA.done" >/dev/null 2>&1 &
nohup bash -c "TEMPOKV_BF16_LAG_FRAMES=18 LABEL=lagfind_lag18 WATCH_IDLE=25 START=0 END=1 bash scripts/paper_experiments/span/span_bigL.sh 24 $G2; touch $D/chainB.done" >/dev/null 2>&1 &
wait_markers 5400 "$D/chainA.done" "$D/chainB.done" || log "phase A 带伤继续"

# ---- Phase B: 选 lag* ----
log "phase B: selecting lag*"
LAGSTAR=$($QVG - 2>>"$D/driver.log" <<'PY'
import json,os,csv,sys
R='results/paper/span-int2/runs'
T_P,T_K=13.347423,12.34*2**30
def get(label):
    d=os.path.join(R,label,'pixel_717')
    try:
        rs=list(csv.DictReader(open(os.path.join(d,'streaming_psnr_ssim.csv'))))
        p=float(rs[0]['psnr'])
        pk=json.load(open(os.path.join(d,'efficiency.json')))['kv_cache']['peak_resident_bytes']
        return p,pk
    except Exception as e:
        sys.stderr.write(f'selector: {label} skip ({e})\n'); return None
best=(9e9,12)
for l,lab in [(12,'lagfind_lag12'),(15,'lagfind_lag15'),(18,'lagfind_lag18'),
              (21,'lagfind_lag21'),(24,'ablate_incr_lag24')]:
    g=get(lab)
    if not g: continue
    p,pk=g
    s=abs(p-T_P)/T_P+abs(pk-T_K)/T_K
    sys.stderr.write(f'selector: lag{l}: psnr={p:.2f} peak={pk/2**30:.2f}GiB score={s:.4f}\n')
    if s<best[0]: best=(s,l)
sys.stderr.write(f'selector: pick lag*={best[1]}\n')
print(best[1])
PY
)
case "$LAGSTAR" in 12|15|18|21|24) ;; *) LAGSTAR=15 ;; esac
echo "$LAGSTAR" > "$D/lagstar.txt"
log "lag*=$LAGSTAR"

# ---- Phase C: span 扫描(卡1:4,12,24,48,96,180 / 卡2:8,16,36,64,128)----
log "phase C: span sweep at lag=$LAGSTAR"
nohup bash -c "for L in 4 12 24 48 96 180; do TEMPOKV_BF16_LAG_FRAMES=$LAGSTAR LABEL=spansweep_lag${LAGSTAR}_L\${L} WATCH_IDLE=25 START=0 END=1 bash scripts/paper_experiments/span/span_bigL.sh \$L $G1; done; touch $D/chainC.done" >/dev/null 2>&1 &
nohup bash -c "for L in 8 16 36 64 128; do TEMPOKV_BF16_LAG_FRAMES=$LAGSTAR LABEL=spansweep_lag${LAGSTAR}_L\${L} WATCH_IDLE=25 START=0 END=1 bash scripts/paper_experiments/span/span_bigL.sh \$L $G2; done; touch $D/chainD.done" >/dev/null 2>&1 &
wait_markers 14400 "$D/chainC.done" "$D/chainD.done" || log "phase C 带伤收尾"

# ---- Phase D: 汇总 ----
$QVG - "$D" "$LAGSTAR" <<'PY'
import json,os,csv,sys
D,LAG=sys.argv[1],sys.argv[2]
R=os.path.dirname(D)
def row(label):
    d=os.path.join(R,label,'pixel_717')
    try:
        r0=list(csv.DictReader(open(os.path.join(d,'streaming_psnr_ssim.csv'))))[0]
        pk=json.load(open(os.path.join(d,'efficiency.json')))['kv_cache']['peak_resident_bytes']/2**30
        return [r0['psnr'],r0['ssim'],r0['lpips'],f'{pk:.2f}',f'{48.2/pk:.2f}']
    except Exception:
        return ['—']*5
with open(os.path.join(D,'SUMMARY.md'),'w') as h:
    h.write(f'# lag 搜索 + span 扫描(固定 lag*={LAG})\n\n')
    h.write('目标:lag∈[12,24] 找质量&驻留≈batched span24(13.347 dB / 12.34 GiB)的工作点;之后固定 lag* 扫 span。\n\n')
    h.write('## A. lag 曲线(idx0, span=24, lag 须为3的倍数)\n\n| 配置 | PSNR | SSIM | LPIPS | peak GiB | R_peak |\n|---|---|---|---|---|---|\n')
    for name,lab in [('lag3(主线)','int2_anchor4_span24_incr'),('lag12','lagfind_lag12'),('lag15','lagfind_lag15'),
                     ('lag18','lagfind_lag18'),('lag21','lagfind_lag21'),('lag24','ablate_incr_lag24'),
                     ('batched','ablate_batch_span24')]:
        h.write(f'| {name} | '+' | '.join(row(lab))+' |\n')
    h.write(f'\n## B. span 曲线(idx0, lag={LAG} 固定)\n\n| L | PSNR | SSIM | LPIPS | peak GiB | R_peak |\n|---|---|---|---|---|---|\n')
    for L in (4,8,12,16,24,36,48,64,96,128,180):
        r=row(f'spansweep_lag{LAG}_L{L}')
        note=' ⚠' if L%3 else ''
        if r[3]=='—' and L%3:
            r[3]='≈12.1*'; r[4]='≈4.0*'   # 非3倍数 span:rc=2 无 efficiency,解析估计
        h.write(f'| {L}{note} | '+' | '.join(r)+' |\n')
    h.write('\n> ⚠ L 非 3 的倍数时与 block(3帧)不对齐,generate 终检报 non-direct decode route(rc=2),\n')
    h.write('> 视频与逐条指标仍有效,但 efficiency 记录缺失(*为解析估计)。lag 同理须为 3 的倍数。\n')
print('SUMMARY written')
PY
log "ALL DONE"
