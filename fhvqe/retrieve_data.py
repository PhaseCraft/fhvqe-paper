#   Copyright 2021 Phasecraft Ltd.
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.

import cirq

import os
import json
import logging
import numpy as np
import pandas as pd

logger = logging.getLogger("fhvqe.retrieve_data")

# Constants
CIRCUIT_JSON = "circuit.json"
RESULTS_JSON = "results.json"
PROJECT_ID = "fermi-hubbard-vqe"
RESULTS_FOLDER = "results"

project_id = PROJECT_ID
engine = cirq.google.Engine(project_id=project_id)


program_id_filename = "data/program_ids.csv"
full_program_ids = []
with open(program_id_filename, "r") as pids:
    full_program_ids = [pid[:-1] for pid in pids]

def retrieve_historical_data(program_id, prefix=""):
    """Retrieve historical data for a given program_id

    Args:
        program_id -- The program_id as saved in jobs_xxx.json
        prefix -- The job prefix, e.g. "tflo-", "noise-", "givens" or ...

    Returns:
        (jobs, results) A list `jobs` containing all jobs with the 
        given program_id and a list containing all the `result`s of 
        these jobs.
    """
    full_pids = list(filter(lambda pid: prefix+program_id in pid, full_program_ids))
    jobs = []
    print(f"Program id: {program_id}. Prefix: {prefix}. Full pids: {list(full_pids)}")
    for full_pid in full_pids:
        jobs += engine.get_program(program_id=full_pid).list_jobs()
    results = [job.results() for job in jobs]
    return jobs, results


def retrieve_historical_samples(program_id, prefix=""):
    """Retrieve historical samples for a given program_id

    Args:
        program_id -- The program_id as saved in jobs_xxx.json
        prefix -- The job prefix, e.g. "tflo-", "noise-", "givens" or ...

    Returns:
        `samples` A list of int16-matrices containing the samples ordered s.t. 
        they are compatible with the observables in `scripts/analyse_results.py`
    """
    jobs, results = retrieve_historical_data(program_id, prefix=prefix)
    print(f"Retrieved {len(jobs)} jobs for program id {program_id} prefix {prefix}")
    samples = results[0]
    samples = [s.measurements["x"].astype(np.int16).T for s in samples]
    return samples


def load_samples_txt(filename):
    """Load a samples.txt file to a list of numpy arrays

    Args:
        filename -- The `samples.txt` (or similar) filename containing the samples

    Returns:
        `samples` A list of int16-matrices containing the samples ordered s.t. 
        they are compatible with the observables in `scripts/analyse_results.py`.
    """
    with open(filename) as file:
        samples_blocks = file.read().split("\n\n\n")
    # delete last block if it is empty
    if len(samples_blocks[-1]) == 1: samples_blocks = samples_blocks[:-1]
    samples_list = [block.splitlines() for block in samples_blocks]
    samples = []
    for sample in samples_list:
        if len(sample) != 0:
            arr = np.empty((len(sample[0]), len(sample)), dtype=np.int16)
            for i in range(len(sample[0])):
                for j in range(len(sample)):
                    arr[i,j] = sample[j][i]
            samples.append(arr)
    
    return samples

# Allow this file to be run directly as a separate script.
if __name__ == "__main__":
    # Change here:
    folder = "heatmaps/4x1"
    filename = "jobs_4x1_2.json"

    full_filename = os.path.join(folder, filename)
    results_folder = os.path.join(folder, RESULTS_FOLDER)
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)

    with open(full_filename) as json_file:
        jobs_data = json.load(json_file)

    for j, job_data in enumerate(jobs_data):
        print(f"[{j}/{len(jobs_data)-1}]\r", end="")
        job_id = job_data["job_id"]
        prog_id = job_data["prog_id"]
        job = engine.get_program(program_id=prog_id).get_job(job_id=job_id)
        if job.status() == "SUCCESS":
            job_results = job.results()
            prog_folder = os.path.join(results_folder, prog_id)
            if not os.path.exists(prog_folder):
                os.makedirs(prog_folder)
            prog_data = cirq.to_json(job.program().get_circuit())
            with open(os.path.join(prog_folder, CIRCUIT_JSON), 'w') as outfile:
                json.dump(prog_data, outfile)
            with open(os.path.join(prog_folder, RESULTS_JSON), 'w') as outfile:
                json.dump(cirq.to_json(job_results), outfile)

