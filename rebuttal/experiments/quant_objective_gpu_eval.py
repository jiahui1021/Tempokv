"""Evaluate original and matched-refinement predictors on identical GPU batches."""
import argparse
from dataclasses import replace
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]
import torch
from tempokv.config import load_config
from rebuttal.common.trace import load_trace, load_trace_identity, check_trace_split
from rebuttal.common.cache_metrics import load_pair, replay, finish_metrics
from rebuttal.common.manifest import write_json, file_sha256, read_json


def run(args):
    device = torch.device('cuda:0')
    torch.cuda.set_per_process_memory_fraction(.06, device)
    trace = load_trace(args.evaluation, device='cpu')
    split = check_trace_split(load_trace_identity(args.calibration), trace)
    assert split['prompt_identity_verified']
    root = Path(args.fit_root)
    assert (root / 'fit_complete.json').is_file()
    folders = dict(original=Path(args.original), prediction=root/'prediction',
                   quantization=root/'quantization')
    for arm, folder in folders.items():
        codec = load_config(folder/'codec.json')
        k, v = load_pair(folder/'k_temporal.npz', folder/'v_cross_kv.npz', codec, device)
        totals = {role: {part: {} for part in ('all','anchor','warmup','predicted','non_anchor')}
                  for role in ('k','v')}
        began = time.perf_counter()
        for layer in range(trace.shape[0]):
            for start in range(0, trace.shape[1], args.batch_records):
                stop = min(start+args.batch_records, trace.shape[1])
                batch = replace(trace,
                    keys=trace.keys[layer:layer+1,start:stop].to(device),
                    values=trace.values[layer:layer+1,start:stop].to(device),
                    records=trace.records[start:stop],layer_ids=(trace.layer_ids[layer],))
                report = replay(batch,codec,k,v,span_frames=24,event_stride=trace.shape[-3]+1)
                for role in ('k','v'):
                    for part, acc in totals[role].items():
                        block = report[role][part]
                        for field in ('numel','signal_sse','prediction_sse','quant_error_sse','error_sse'):
                            acc[field] = acc.get(field,0) + block.get(field,0)
                del batch, report
            print(f'evaluation {arm} layer {layer} complete',flush=True)
        torch.cuda.synchronize(device)
        result = dict(status='complete', arm=arm, metric_scope='bf16_trace_replay',
                      trace=str(trace.path), trace_sha256=file_sha256(trace.path),
                      shape=list(trace.shape), span_frames=24, config=codec.__dict__ if hasattr(codec,'__dict__') else None,
                      trace_split=split, device='GPU',batch_records=args.batch_records,
                      evaluation_seconds=time.perf_counter()-began,
                      k_sha256=file_sha256(folder/'k_temporal.npz'),
                      v_sha256=file_sha256(folder/'v_cross_kv.npz'),
                      **{role:{part:finish_metrics(acc) for part,acc in values.items()}
                         for role,values in totals.items()})
        from dataclasses import asdict
        result['config'] = asdict(codec)
        write_json(root/'evaluation'/arm/'summary.json',result)
    write_json(root/'evaluation'/'complete.json',dict(status='complete',test_split=split))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for field in ('fit-root','original','calibration','evaluation'):
        parser.add_argument('--'+field,required=True)
    parser.add_argument('--batch-records',type=int,default=64)
    run(parser.parse_args())
