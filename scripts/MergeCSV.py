import pandas as pd
import argparse
import time
from Utils import BASE_DIR, mergeEfficiencyCSV# concat multiple dfs

##### Parser
parser = argparse.ArgumentParser(description='Merge csv')
parser.add_argument('inputs', help='Input csv(s)',nargs='*')
parser.add_argument('--output_name', type=str , help="Output csv name",required=False,default="test_merge_csv")
args = parser.parse_args()

output_name = args.output_name+time.strftime("_%-y%m%d%H%M")

# concat multiple dfs
final_df = mergeEfficiencyCSV([pd.read_csv(k, sep=',') for k in args.inputs])
# save to file
final_df.to_csv(BASE_DIR /f"data/output/{output_name}.csv",index=False)
with open(BASE_DIR /f"data/output/{output_name}.yml", "w") as text_file: text_file.writelines(line + '\n' for line in args.inputs)