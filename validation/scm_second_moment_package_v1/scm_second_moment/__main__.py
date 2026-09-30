"""CLI for portable historical Student5 second-moment proofs."""
import argparse
import json
from .proof import generate, verify_bundle


def main():
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest='command', required=True)
    commands.add_parser('prove').add_argument('--config', required=True)
    commands.add_parser('verify-proof').add_argument('bundle')
    args = parser.parse_args()
    result = generate(args.config) if args.command == 'prove' else verify_bundle(args.bundle)
    print(json.dumps(result, sort_keys=True))


if __name__ == '__main__':
    main()