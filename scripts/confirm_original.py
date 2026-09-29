"""Run or resume the frozen original-project confirmation protocol."""
import argparse
from ncd.original_confirmation import run_confirmation

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--protocol',required=True)
    p.add_argument('--resume',action='store_true')
    p.add_argument('--verify',action='store_true')
    p.add_argument('--seconds',type=float)
    args=p.parse_args()
    result=run_confirmation(args.protocol,args.resume,args.seconds,args.verify)
    print(result['state'],result['computed_worlds'],'/',result['declared_worlds'],'worlds; science_passed=',result['science_passed'])
