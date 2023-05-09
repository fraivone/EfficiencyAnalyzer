import argparse
from argparse import RawTextHelpFormatter
from Utils import BASE_DIR
from os import system
from os.path import abspath
from config_parser import config
from pathlib import Path

def generateSubFile(outputName,shell_name):
    job_flavour = "tomorrow"
    SubfileName = Path(BASE_DIR / f"condor/JobFiles/SubmitFile_{outputName}.sub")
    with open(SubfileName, 'w') as sout:
        sout.write(f"executable              = {shell_name}\n")
        sout.write("getenv                  = true"+"\n")
        sout.write("arguments               = $(ClusterId) $(ProcId)"+"\n")
        sout.write(f"output                  = ./Logs/out_Run{outputName}.dat"+"\n")
        sout.write(f"error                   = ./Logs/error_Run{outputName}.err"+"\n")
        sout.write(f"log                     = ./Logs/log_Run{outputName}.log"+"\n") #./Logs/log.$(ClusterId)_Run{outputName}.log
        sout.write(f"+JobFlavour             = \"{job_flavour}\""+"\n")
        sout.write("notify_user             = francesco.ivone@cern.ch\n")
        sout.write("notification            = always\n")
        sout.write("queue"+"\n")
    return SubfileName

def generateJobShell(name,absPath_config):
    main_command = f"python Analyzer.py {absPath_config}"    
    main_command = main_command + " \n"

    shell_script_name = Path(BASE_DIR / f"condor/JobFiles/job_Run_{name}.sh")
    print(shell_script_name)
    with open(shell_script_name, 'w') as fout:
        ####### Write the instruction for each job
        fout.write("#!/bin/sh\n")
        fout.write("echo\n")
        fout.write("echo %s_Run"+str(name)+"\n")
        fout.write("echo\n")
        # fout.write("scl -l | grep python \n")
        fout.write("echo 'START---------------'\n")
        fout.write(f"cd {BASE_DIR} \n")
        # ## sourceing the right gcc version to compile the source code
        # fout.write("source /cvmfs/sft.cern.ch/lcg/contrib/gcc/9.1.0/x86_64-centos7/setup.sh"+"\n")
        # fout.write("source /cvmfs/sft.cern.ch/lcg/app/releases/ROOT/6.18.04/x86_64-centos7-gcc48-opt/bin/thisroot.sh"+"\n")
        # activate poetry venv
        # fout.write("source /opt/rh/rh-python38/enable "+"\n")
        fout.write("source /afs/cern.ch/user/f/fivone/.cache/pypoetry/virtualenvs/pfa-columnar-analyzer-5YpECzXB-py3.9/bin/activate"+"\n")
        ## Run the same job interval number of time
        fout.write("ClusterId=$1\n")
        fout.write("ProcId=$2\n")
        fout.write(main_command)
    return shell_script_name
if __name__=='__main__':
    parser = argparse.ArgumentParser(
        description='''Executes the chain Analyzer.py - PlotVFATStatus.py - PlotEfficiency.py - VFATEffPlotter on the CERN Batch Service (HTCondor)''',
        epilog="""Typical exectuion (from this folder)\n\t  python runAnalyzers.py ../data/config/file1.yml ../data/config/file2.yml""",
        formatter_class=RawTextHelpFormatter
        )
    parser.add_argument('config', help='Analysis description file', nargs="*")
    args = parser.parse_args()
    config_list = args.config
    
    for filename in config_list:
        abspath_config = abspath(filename)
        configuration = config(abspath_config)

        shell_name = generateJobShell(configuration.analysis_label,abspath_config)
        SubfileName = generateSubFile(configuration.analysis_label,shell_name)
        condorDAG_file = Path(BASE_DIR / f"condor/condor_DAG/condor_DAG_{configuration.analysis_label}.dag")

        with open(condorDAG_file, "w") as DAG_file:
            DAG_file.write(
                """
                JOB A {analysis_submit}
                """.format(analysis_submit=SubfileName))
        
        system(f"condor_submit_dag -dont_suppress_notification {condorDAG_file}")

