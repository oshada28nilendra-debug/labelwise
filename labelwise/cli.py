import argparse
import json
from pathlib import Path
from .core import parse_csv, benchmark, train_and_queue

def main():
    parser=argparse.ArgumentParser(description="Local active-learning assistant")
    sub=parser.add_subparsers(dest="command",required=True)
    serve=sub.add_parser("serve");serve.add_argument("--port",type=int,default=8768)
    queue=sub.add_parser("queue");queue.add_argument("--seed",type=Path,required=True);queue.add_argument("--pool",type=Path,required=True);queue.add_argument("--size",type=int,default=10);queue.add_argument("--output",type=Path,required=True)
    run=sub.add_parser("benchmark");run.add_argument("--seed",type=Path,required=True);run.add_argument("--pool",type=Path,required=True);run.add_argument("--test",type=Path,required=True);run.add_argument("--budgets",type=int,nargs="+",default=[0,5,10,20,40]);run.add_argument("--repeats",type=int,default=20);run.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    try:
        if args.command=="serve":
            from .web import create_app
            create_app().run(host="127.0.0.1",port=args.port,debug=False,threaded=False);return
        seed=parse_csv(args.seed.read_bytes())
        result=train_and_queue(seed,parse_csv(args.pool.read_bytes(),False),args.size) if args.command=="queue" else benchmark(seed,parse_csv(args.pool.read_bytes()),parse_csv(args.test.read_bytes()),args.budgets,args.repeats)
        args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(result,indent=2)+"\n");print(args.output)
    except (ValueError,OSError) as exc:parser.exit(2,f"Error: {exc}\n")

if __name__=="__main__":main()
