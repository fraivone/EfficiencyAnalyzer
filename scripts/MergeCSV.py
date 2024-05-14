import pandas as pd
import argparse
import time
from Utils import OUTPUT_PATH, mergeEfficiencyCSV

##### Parser
parser = argparse.ArgumentParser(description='Merge csv')
parser.add_argument('inputs', help='Input csv(s)',nargs='*')
parser.add_argument('--output_name', type=str , help="Output csv name",required=False,default="test_merge_csv")
args = parser.parse_args()

output_name = args.output_name+time.strftime("_%-y%m%d%H%M")

# concat multiple dfs
final_df = mergeEfficiencyCSV([pd.read_csv(k, sep=',') for k in args.inputs])

# save to file
final_df.to_csv(OUTPUT_PATH / f"/{output_name}.csv", index=False)
