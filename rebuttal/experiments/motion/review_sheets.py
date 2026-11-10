"""Extract temporal contact sheets for visual review; never auto-verify labels."""
import argparse
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[3]
sys.path[:0]=[str(ROOT),str(ROOT/'src')]
from rebuttal.common.manifest import read_json,write_json
from rebuttal.common.video_metrics import VideoReader


def frames(path,indices):
    from PIL import Image
    reader=VideoReader(Path(path),'pyav');wanted=set(indices);found={}
    try:
        for frame in reader:
            if frame.index in wanted:
                found[frame.index]=(Image.fromarray(frame.image[:,:,::-1]).resize((416,240)),frame.timestamp)
            if frame.index>=max(wanted):break
    finally:reader.close()
    assert len(found)==len(wanted)
    return found


def main(args):
    from PIL import Image,ImageDraw
    source=Path(args.source);out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    cases=read_json(source/'difficult_cases.json')['cases']
    chosen=[c for c in cases if c['category']=='occlusion_disocclusion'][:5]
    records=[]
    for case in chosen:
        first,last=round(case['actual_start_seconds']*16),round(case['actual_end_seconds']*16)-1
        indices=sorted({round(first+(last-first)*i/4) for i in range(5)})
        left,right=frames(case['reference'],indices),frames(case['candidate'],indices)
        sheet=Image.new('RGB',(832,24+len(indices)*264),'white');draw=ImageDraw.Draw(sheet)
        draw.text((4,4),f"{case['case_id']} prompt {case['prompt_id']} | BF16 left / TempoKV right",fill='black')
        for row,index in enumerate(indices):
            y=24+row*264
            draw.text((4,y),f'frame {index}, t={left[index][1]:.4f}s',fill='black')
            sheet.paste(left[index][0],(0,y+24));sheet.paste(right[index][0],(416,y+24))
        path=out/f"{case['case_id']}.png";sheet.save(path)
        records.append(dict(case_id=case['case_id'],prompt_id=case['prompt_id'],frames=indices,
            sheet=str(path),status='pending_visual_review',manual_verified=False))
    write_json(out/'review_manifest.json',dict(cases=records,scope='representative occlusion candidates; no automatic ground-truth labels'))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--source',required=True);parser.add_argument('--output',required=True)
    main(parser.parse_args())
